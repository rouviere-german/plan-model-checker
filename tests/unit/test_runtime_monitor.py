"""The incremental runtime monitor against a naive reference, plus its cost and budgets.

The monitor only steps spec instances whose letter can have changed, grows the set of instances
one resource at a time, and replays the history of a new instance only at the positions that
concern it. The property test checks all of that against the obvious implementation: instantiate
every spec over the resources seen so far and fold each compiled monitor over the whole trace.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from hypothesis import given
from hypothesis import strategies as st

from plan_model_checker.checker import GroundCall
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.props import END, START, GroundProp
from plan_model_checker.runtime import (
    EffectReportError,
    MonitorBudgetExceededError,
    RuntimeMonitor,
    SensorError,
)
from plan_model_checker.spec.monitor import compile_monitor
from plan_model_checker.spec.specs import InstanceBudgetExceededError, SpecInstance, SpecSet

pytestmark = pytest.mark.unit

POOLS = {"path": ("/a", "/b", "/c"), "service": ("svc:a", "svc:b"), "job": ("job-a",)}
# Specs whose new instances read propositions that already changed before the instance existed:
# several variables, a constant resource, the start event, Since and weak yesterday.
MIXED_SPECS = """spec pair: G(delete[r] -> O archive[q]);
spec shared_lock: G(mutate[r] -> Y !lock_held["/c"]);
spec after_start: G(retry[j] -> (O check[j] & Z !start));
spec drained_since: G(restart[s] -> Y(drained[s] & (!undrain[s] S drain[s])));
spec released: G(end -> (!lock_held[r] | O release[r]));
"""
Snapshot = dict[GroundProp, bool]
Step = tuple[GroundCall, tuple[tuple[GroundProp, bool], ...]]


def gc(tool: str, **args: str) -> GroundCall:
    return GroundCall(tool=tool, args=tuple(args.items()), path="test")


class Reference:
    """Every instance over the resources seen so far, folded over the whole trace every time."""

    def __init__(
        self, registry: EffectRegistry, specs: SpecSet, initial: Mapping[GroundProp, bool]
    ) -> None:
        self.registry = registry
        self.specs = specs
        self.initial = dict(initial)
        self.universe: dict[str, set[str]] = {}
        self.values: Snapshot = {}
        self.trace: list[tuple[GroundProp, Snapshot]] = []

    def value(self, prop: GroundProp, snapshot: Mapping[GroundProp, bool] | None = None) -> bool:
        source = self.values if snapshot is None else snapshot
        if prop in source:
            return source[prop]
        return self.initial.get(prop, self.registry.initial(prop))

    def append(self, event: GroundProp, updates: Mapping[GroundProp, bool] | None = None) -> None:
        self.values.update(updates or {})
        self.trace.append((event, dict(self.values)))

    def add_resources(self, call: GroundCall) -> None:
        sorts = self.registry.param_sorts(call.tool)
        for name, value in call.args:
            self.universe.setdefault(sorts[name], set()).add(value)

    def instances(self) -> list[SpecInstance]:
        return self.specs.instances(
            {sort: tuple(sorted(items)) for sort, items in self.universe.items()}
        )

    def fails(
        self, instance: SpecInstance, extra: tuple[GroundProp, Snapshot] | None = None
    ) -> bool:
        monitor = compile_monitor(instance.final)
        positions = self.trace if extra is None else [*self.trace, extra]
        bits = 0
        for index, (event, snapshot) in enumerate(positions):
            true = {event} | {prop for prop in monitor.props if self.value(prop, snapshot)}
            letter = monitor.letter(true)
            bits = monitor.first(letter) if index == 0 else monitor.step(bits, letter)
            if not monitor.holds(bits):
                return True
        return False

    def read_props(self) -> set[GroundProp]:
        return {prop for inst in self.instances() for prop in compile_monitor(inst.final).props}


@st.composite
def histories(draw: st.DrawFn, registry: EffectRegistry) -> list[Step]:
    steps: list[Step] = []
    for _ in range(draw(st.integers(min_value=0, max_value=10))):
        tool = draw(st.sampled_from(registry.tool_names))
        args = tuple(
            (param.name, draw(st.sampled_from(POOLS[param.sort])))
            for param in registry.tool(tool).params
        )
        declared = sorted(registry.declared_updates(tool, dict(args)))
        reported = tuple((prop, draw(st.booleans())) for prop in declared if draw(st.booleans()))
        steps.append((GroundCall(tool=tool, args=args, path="p"), reported))
    return steps


@st.composite
def initial_states(draw: st.DrawFn) -> dict[GroundProp, bool]:
    props = [
        GroundProp(name, (resource,))
        for name in ("lock_held", "backup_ok", "approved")
        for resource in POOLS["path"]
    ] + [GroundProp("drained", (svc,)) for svc in POOLS["service"]]
    return {prop: True for prop in props if draw(st.booleans())}


@given(data=st.data())
def test_the_incremental_monitor_agrees_with_a_naive_reference(
    registry: EffectRegistry, specs: SpecSet, data: st.DataObject
) -> None:
    if data.draw(st.booleans()):
        specs = SpecSet.parse(MIXED_SPECS, registry)
    steps = data.draw(histories(registry))
    with_sensor = data.draw(st.booleans())
    initial = data.draw(initial_states()) if with_sensor else {}
    monitor = RuntimeMonitor(
        registry, specs, sensor=(lambda prop: initial.get(prop, False)) if with_sensor else None
    )
    reference = Reference(registry, specs, initial)
    reported = {v.instance for v in monitor.start()}
    reference.append(START)
    for call, effects in steps:
        already = {inst.key for inst in reference.instances() if reference.fails(inst)}
        reference.add_resources(call)
        args = call.arg_map()
        event = registry.ground_event(call.tool, args)
        expected: set[str] = set()
        for outcome in registry.ground_outcomes(call.tool, args):
            predicted = {**reference.values, **dict(outcome.updates)}
            expected |= {
                inst.key
                for inst in reference.instances()
                if reference.fails(inst, (event, predicted))
            }
        assert {v.instance for v in monitor.precheck(call)} == expected - already
        reported |= {v.instance for v in monitor.commit(call, effects)}
        reference.append(event, dict(effects))
    reported |= {v.instance for v in monitor.finish()}
    reference.append(END)
    assert reported == {inst.key for inst in reference.instances() if reference.fails(inst)}
    for prop in reference.read_props():
        assert monitor.value(prop) == reference.value(prop), prop


def test_registering_resources_costs_work_per_new_instance_not_per_instance_so_far(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    monitor = RuntimeMonitor(registry, specs)
    monitor.start()
    paths = [f"/p/{i}" for i in range(300)]
    for path in paths:
        monitor.commit(gc("archive_dir", path=path), ((GroundProp("archived", (path,)), True),))
    monitor.finish()
    assert len(monitor.instance_keys()) == 6 * len(paths)
    # Stepping every instance at every position would take about 6 * 300 * 300 / 2 = 270,000
    # monitor steps (and replaying each new instance over the whole history as many again).
    assert monitor.work < 12 * len(paths) * 6


def test_instance_and_work_budgets_fail_closed(registry: EffectRegistry, specs: SpecSet) -> None:
    small = RuntimeMonitor(registry, specs, max_instances=10)
    small.start()
    small.commit(gc("archive_dir", path="/a"), ())
    with pytest.raises(InstanceBudgetExceededError, match="10"):
        small.commit(gc("archive_dir", path="/b"), ())
    probe = RuntimeMonitor(registry, specs)
    probe.start()
    for index in range(3):
        probe.commit(gc("archive_dir", path=f"/p{index}"), ())
    busy = RuntimeMonitor(registry, specs, max_work=probe.work)
    busy.start()
    for index in range(3):
        busy.commit(gc("archive_dir", path=f"/p{index}"), ())  # exactly at the budget
    with pytest.raises(MonitorBudgetExceededError, match=str(probe.work)):
        busy.commit(gc("archive_dir", path="/p3"), ())


@pytest.mark.parametrize(
    ("effects", "code"),
    [
        (((GroundProp("nonexistent"), True),), "unknown_proposition"),
        (((GroundProp("delete", ("/a",)), True),), "invalid_effect"),
        (((GroundProp("archived", ("/a", "/b")), True),), "invalid_effect"),
        (((GroundProp("approved", ("/a",)), True),), "undeclared_effect"),
        (((GroundProp("archived", ("/b",)), True),), "undeclared_effect"),
        (
            ((GroundProp("archived", ("/a",)), True), (GroundProp("archived", ("/a",)), False)),
            "conflicting_effect",
        ),
    ],
)
def test_reports_outside_the_calls_declaration_are_rejected(
    registry: EffectRegistry,
    specs: SpecSet,
    effects: tuple[tuple[GroundProp, bool], ...],
    code: str,
) -> None:
    monitor = RuntimeMonitor(registry, specs)
    monitor.start()
    with pytest.raises(EffectReportError) as info:
        monitor.commit(gc("archive_dir", path="/a"), effects)
    assert info.value.code == code
    assert monitor.position == 0, "a rejected report records nothing"


def test_sensed_values_override_reports_and_need_only_be_declared_fluents(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    monitor = RuntimeMonitor(registry, specs)
    monitor.start()
    backup = GroundProp("backup_ok", ("/a",))
    lock = GroundProp("lock_held", ("/a",))
    monitor.commit(gc("snapshot_dir", path="/a"), ((backup, True),), sensed=((backup, False),))
    assert monitor.value(backup) is False
    monitor.commit(gc("archive_dir", path="/a"), (), sensed=((lock, True),))
    assert monitor.value(lock) is True
    with pytest.raises(EffectReportError):
        monitor.commit(
            gc("archive_dir", path="/a"), (), sensed=((GroundProp("delete", ("/a",)), True),)
        )


def test_a_failing_sensor_is_reported_as_a_sensor_error(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    def broken(prop: GroundProp) -> bool:
        raise KeyError(prop.name)

    monitor = RuntimeMonitor(registry, specs, sensor=broken)
    with pytest.raises(SensorError, match="backup_ok"):
        monitor.start([("/a", "path")])
    blind = RuntimeMonitor(registry, specs)
    assert blind.sense([GroundProp("backup_ok", ("/a",))]) == {}


def test_resources_can_only_be_registered_on_a_started_monitor(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    monitor = RuntimeMonitor(registry, specs)
    with pytest.raises(RuntimeError, match="not started"):
        monitor.register_resources([("/a", "path")])
