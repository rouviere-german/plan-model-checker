"""Runtime enforcement: the same compiled specs, applied to what tools actually do.

The static check is only as good as the effect declarations. At runtime :class:`PlanExecutor`
first re-checks the plan itself (it refuses a plan that does not verify, unless an approved
override for exactly that plan covers every violation), then walks it and routes every call
through two layers:

1. :class:`ToolGateway` resolves the call to an executable tool that is registered *and* pinned
   to the exact declaration hash in the signed registry (unregistered or modified tools are
   refused at construction time).
2. :class:`RuntimeMonitor` keeps the observed fluent state and a live monitor per spec instance.
   Before a call runs, it predicts the next position for every declared outcome of the call from
   the *observed* state and blocks the call if any spec would become false. After the call it
   records the effects: a tool may only report fluents its declaration says the call can change,
   and when a sensor is configured the sensor's reading of every fluent the specs depend on
   takes precedence over the tool's own report. A tool whose report the sensor contradicts has
   misreported its effects, and the run stops there.

Without a sensor the monitor has nothing to compare a report against, so it trusts what tools
report (README, Non-goals).

Branch guards on propositions read the observed state, so ``If(approved[p])`` at runtime means
what it meant to the checker. Collections are resolved once, at the start of a run, and refused if
they exceed their bound, are not well-formed resource names, or alias resources the plan names
anywhere (the checker assumed fresh, distinct items).

The monitor is incremental. Spec instances are created as resources appear, and a new instance
replays only the positions of the history that concern its propositions. At each new position it
steps only the instances whose letter can have changed (they read the new or the previous event,
or a fluent that changed) plus those whose state has not settled yet; a settled instance that
sees the same letter again keeps its state. The cost of a step is proportional to the instances
it concerns, not to every instance created so far, and both the instance count and the total
number of monitor steps can be capped (the HTTP API does).
"""

from __future__ import annotations

import bisect
import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import JsonValue

from plan_model_checker.agentic.audit import AuditLog
from plan_model_checker.agentic.envelope import UntrustedData, wrap_tool_result
from plan_model_checker.checker import (
    CheckBudgetExceededError,
    CheckLimits,
    GroundCall,
    ModelChecker,
    collection_bounds,
)
from plan_model_checker.core.canonical import sha256_hex
from plan_model_checker.effects import EffectRegistry, UnknownToolError
from plan_model_checker.ir import (
    ForEach,
    If,
    Node,
    OpaqueGuard,
    Plan,
    PlanError,
    ToolCall,
    is_valid_resource,
    plan_sha256,
)
from plan_model_checker.override import OverrideError, OverrideGate
from plan_model_checker.props import END, START, GroundProp
from plan_model_checker.spec.monitor import CompiledMonitor, compile_monitor
from plan_model_checker.spec.specs import InstanceBudgetExceededError, SpecInstance, SpecSet

__all__ = [
    "EffectReportError",
    "ExecutableTool",
    "MonitorBudgetExceededError",
    "MonitorViolation",
    "Observation",
    "PlanExecutor",
    "RunResult",
    "RuntimeMonitor",
    "SensorError",
    "ToolGateway",
    "ToolPinError",
    "describe_outputs",
]


class ToolPinError(Exception):
    """An executable tool is unregistered or pinned to a different declaration."""


class EffectReportError(ValueError):
    """A reported (or sensed) effect that the call cannot have.

    ``code`` is ``unknown_proposition`` (not declared), ``invalid_effect`` (an event, or the wrong
    number of arguments), ``undeclared_effect`` (a fluent outside the call's declared effects,
    such as an approval reported by a tool that cannot grant one) or ``conflicting_effect`` (two
    values for one fluent).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MonitorBudgetExceededError(Exception):
    """The monitor would perform more monitor steps than its work budget allows."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"monitor work budget of {limit} steps exceeded")
        self.limit = limit


class SensorError(Exception):
    """The sensor could not read a fluent."""


@dataclass(frozen=True, slots=True)
class Observation:
    """What a tool reports after running: an outcome label, observed fluent values, output."""

    outcome: str
    effects: tuple[tuple[GroundProp, bool], ...] = ()
    output: JsonValue = None


class ExecutableTool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def declared_sha256(self) -> str: ...

    def __call__(self, args: Mapping[str, str]) -> Observation: ...


@dataclass(frozen=True, slots=True)
class MonitorViolation:
    spec: str
    instance: str
    position: int
    kind: Literal["blocked", "after_execution", "at_end"]
    step: str
    waived: bool = False

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "spec": self.spec,
            "instance": self.instance,
            "position": self.position,
            "kind": self.kind,
            "step": self.step,
            "waived": self.waived,
        }


@dataclass
class _Live:
    instance: SpecInstance
    monitor: CompiledMonitor
    bits: int = 0
    letter: int = 0
    # True when stepping ``bits`` with ``letter`` again leaves them unchanged.
    stable: bool = False
    violated: bool = False


Sensor = Callable[[GroundProp], bool]


class RuntimeMonitor:
    """Parametric runtime monitor over observed events and fluent values."""

    def __init__(
        self,
        registry: EffectRegistry,
        specs: SpecSet,
        *,
        waived: Iterable[str] = (),
        sensor: Sensor | None = None,
        max_instances: int = 10_000,
        max_work: int | None = None,
    ) -> None:
        self._registry = registry
        self._specs = specs
        self._waived = frozenset(waived)
        self._sensor = sensor
        self._max_instances = max_instances
        self._max_work = max_work
        #: Monitor steps performed so far (replays and live steps); what ``max_work`` caps.
        self.work = 0
        self._events: list[GroundProp] = []
        self._event_positions: dict[GroundProp, list[int]] = {}
        self._values: dict[GroundProp, bool] = {}
        self._seeded: dict[GroundProp, bool] = {}
        self._change_positions: dict[GroundProp, list[int]] = {}
        self._change_values: dict[GroundProp, list[bool]] = {}
        self._universe: dict[str, set[str]] = {}
        self._live: dict[str, _Live] = {}
        self._readers: dict[GroundProp, set[str]] = {}
        self._unstable: set[str] = set()
        self._fluent_names: dict[str, bool] = {}
        self._initialised = False
        self._started = False

    # --- state --------------------------------------------------------------------------------

    @property
    def position(self) -> int:
        return len(self._events) - 1

    @property
    def has_sensor(self) -> bool:
        return self._sensor is not None

    def _initial(self, prop: GroundProp) -> bool:
        seeded = self._seeded.get(prop)
        return seeded if seeded is not None else self._registry.initial(prop)

    def value(self, prop: GroundProp) -> bool:
        current = self._values.get(prop)
        return current if current is not None else self._initial(prop)

    def _value_at(self, prop: GroundProp, position: int) -> bool:
        positions = self._change_positions.get(prop)
        if positions:
            index = bisect.bisect_right(positions, position) - 1
            if index >= 0:
                return self._change_values[prop][index]
        return self._initial(prop)

    def instance_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._live))

    def tracked_fluents(self) -> tuple[GroundProp, ...]:
        """Fluents some spec instance reads (what a sensor should refresh)."""
        return tuple(sorted(prop for prop in self._readers if self._is_fluent(prop)))

    def sense(self, props: Iterable[GroundProp]) -> dict[GroundProp, bool]:
        """Read ``props`` through the sensor (nothing without one). Raises SensorError."""
        sensor = self._sensor
        if sensor is None:
            return {}
        return {prop: _read(sensor, prop) for prop in props}

    def _is_fluent(self, prop: GroundProp) -> bool:
        known = self._fluent_names.get(prop.name)
        if known is None:
            known = (
                self._registry.has_prop(prop.name)
                and self._registry.prop(prop.name).kind == "fluent"
            )
            self._fluent_names[prop.name] = known
        return known

    def _letter(
        self, monitor: CompiledMonitor, event: GroundProp, value: Callable[[GroundProp], bool]
    ) -> int:
        letter = 0
        for index, prop in enumerate(monitor.props):
            if prop == event or (self._is_fluent(prop) and value(prop)):
                letter |= 1 << index
        return letter

    def _letter_at(self, monitor: CompiledMonitor, position: int) -> int:
        return self._letter(
            monitor, self._events[position], functools.partial(self._at, position=position)
        )

    def _at(self, prop: GroundProp, *, position: int) -> bool:
        return self._value_at(prop, position)

    def _charge(self) -> None:
        self.work += 1
        if self._max_work is not None and self.work > self._max_work:
            raise MonitorBudgetExceededError(self._max_work)

    # --- instances ----------------------------------------------------------------------------

    def register_resources(self, resources: Iterable[tuple[str, str]]) -> None:
        """Add ``(resource, sort)`` pairs; new spec instances replay the history so far."""
        self._require_started()
        added: dict[str, set[str]] = {}
        for resource, sort in resources:
            bucket = self._universe.setdefault(sort, set())
            if resource not in bucket:
                bucket.add(resource)
                added.setdefault(sort, set()).add(resource)
        if self._initialised and not added:
            return
        remaining = max(self._max_instances - len(self._live), 0)
        try:
            if self._initialised:
                fresh = self._specs.new_instances(self._universe, added, max_instances=remaining)
            else:
                universe = {sort: tuple(sorted(items)) for sort, items in self._universe.items()}
                fresh = self._specs.instances(universe, max_instances=remaining)
        except InstanceBudgetExceededError as exc:
            raise InstanceBudgetExceededError(self._max_instances) from exc
        self._initialised = True
        created: list[_Live] = []
        for instance in fresh:
            live = _Live(instance=instance, monitor=compile_monitor(instance.final))
            self._live[instance.key] = live
            for prop in live.monitor.props:
                self._readers.setdefault(prop, set()).add(instance.key)
            created.append(live)
        self._seed(created)
        for live in created:
            self._replay(live)
            # Processed at the next position whatever happens, so its verdict is collected.
            self._unstable.add(live.instance.key)

    def _seed(self, created: Sequence[_Live]) -> None:
        """Sense the initial value of fluents that new instances read and nothing set yet.

        The registry's declared initial values are an assumption about the world; with a sensor
        the monitor starts from what is actually there (a lock another process already holds). A
        resource first mentioned mid-run is assumed to have kept its sensed value until then.
        """
        sensor = self._sensor
        if sensor is None:
            return
        for live in created:
            for prop in live.monitor.props:
                if prop in self._seeded or prop in self._values or not self._is_fluent(prop):
                    continue
                self._seeded[prop] = _read(sensor, prop)

    def _relevant_positions(self, props: Sequence[GroundProp]) -> list[int]:
        """Positions > 0 where some prop's truth value may differ from the previous position."""
        last = self.position
        found: set[int] = set()
        for prop in props:
            for position in self._event_positions.get(prop, ()):
                found.add(position)
                if position < last:
                    found.add(position + 1)
            found.update(self._change_positions.get(prop, ()))
        found.discard(0)
        return sorted(found)

    def _replay(self, live: _Live) -> None:
        monitor = live.monitor
        last = self.position
        letter = self._letter_at(monitor, 0)
        bits = monitor.first(letter)
        self._charge()
        stable = False
        relevant = self._relevant_positions(monitor.props)
        cursor = 0
        position = 1
        while position <= last:
            if stable:
                # Until the next relevant position the letter repeats, so the bits do too.
                cursor = bisect.bisect_left(relevant, position, lo=cursor)
                if cursor == len(relevant):
                    break
                position = relevant[cursor]
            new_letter = self._letter_at(monitor, position)
            new_bits = monitor.step(bits, new_letter)
            self._charge()
            stable = new_letter == letter and new_bits == bits
            letter, bits = new_letter, new_bits
            position += 1
        live.bits, live.letter, live.stable = bits, letter, stable

    def acknowledge(self, keys: Iterable[str]) -> None:
        """Mark instances as already reported (a waived violation is logged once, not twice)."""
        for key in keys:
            self._live[key].violated = True

    def _call_resources(self, call: GroundCall) -> list[tuple[str, str]]:
        sorts = self._registry.param_sorts(call.tool)
        return [(value, sorts[name]) for name, value in call.args]

    # --- reports ------------------------------------------------------------------------------

    def _check_fluent(self, prop: GroundProp) -> None:
        if not self._registry.has_prop(prop.name):
            raise EffectReportError(
                "unknown_proposition", f"proposition {prop.name} is not declared"
            )
        decl = self._registry.prop(prop.name)
        if decl.kind != "fluent" or len(decl.params) != len(prop.args):
            raise EffectReportError(
                "invalid_effect", f"{prop} is not a declared fluent with {len(prop.args)} args"
            )

    def check_report(
        self, call: GroundCall, reported: Iterable[tuple[GroundProp, bool]]
    ) -> dict[GroundProp, bool]:
        """Validate what a tool (or an API client) reports for ``call``; return it as a map.

        A call may only report fluents that one of its declared outcomes can set. Anything else
        (an archive tool reporting ``approved[path]``) is rejected, not applied.
        """
        declared = self._registry.declared_updates(call.tool, call.arg_map())
        result: dict[GroundProp, bool] = {}
        for prop, value in reported:
            self._check_fluent(prop)
            if prop not in declared:
                raise EffectReportError(
                    "undeclared_effect",
                    f"{call.label()} reported {prop}, which is not among its declared effects",
                )
            if result.get(prop, bool(value)) != bool(value):
                raise EffectReportError(
                    "conflicting_effect", f"{call.label()} reported two values for {prop}"
                )
            result[prop] = bool(value)
        return result

    # --- protocol -----------------------------------------------------------------------------

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("monitor not started")

    def start(self, resources: Iterable[tuple[str, str]] = ()) -> list[MonitorViolation]:
        if self._started:
            raise RuntimeError("monitor already started")
        self._started = True
        self._events.append(START)
        self._event_positions[START] = [0]
        self.register_resources(resources)
        return self._collect("after_execution", "start", list(self._live))

    def _collect(
        self, kind: Literal["after_execution", "at_end"], step: str, keys: Iterable[str]
    ) -> list[MonitorViolation]:
        found: list[MonitorViolation] = []
        for key in sorted(keys):
            live = self._live[key]
            if live.violated or live.monitor.holds(live.bits):
                continue
            live.violated = True
            found.append(
                MonitorViolation(
                    spec=live.instance.spec,
                    instance=key,
                    position=self.position,
                    kind=kind,
                    step=step,
                    waived=key in self._waived,
                )
            )
        return found

    def _touched(self, event: GroundProp, changed: Iterable[GroundProp]) -> set[str]:
        """Instances whose letter at a new position with ``event`` may differ, plus unsettled."""
        touched = set(self._unstable)
        for prop in (event, self._events[-1], *changed):
            touched |= self._readers.get(prop, set())
        return touched

    def precheck(self, call: GroundCall) -> list[MonitorViolation]:
        """Violations the call would cause under any of its declared outcomes (nothing runs)."""
        self._require_started()
        self.register_resources(self._call_resources(call))
        args = call.arg_map()
        event = self._registry.ground_event(call.tool, args)
        found: dict[str, MonitorViolation] = {}
        for outcome in self._registry.ground_outcomes(call.tool, args):
            predicted = dict(outcome.updates)
            changed = [prop for prop, value in predicted.items() if self.value(prop) != value]
            lookup = functools.partial(self._predicted, predicted=predicted)
            for key in sorted(self._touched(event, changed)):
                live = self._live[key]
                if live.violated or key in found:
                    continue
                bits = live.monitor.step(live.bits, self._letter(live.monitor, event, lookup))
                self._charge()
                if not live.monitor.holds(bits):
                    found[key] = MonitorViolation(
                        spec=live.instance.spec,
                        instance=key,
                        position=self.position + 1,
                        kind="blocked",
                        step=call.label(),
                        waived=key in self._waived,
                    )
        return [found[key] for key in sorted(found)]

    def _predicted(self, prop: GroundProp, *, predicted: Mapping[GroundProp, bool]) -> bool:
        value = predicted.get(prop)
        return value if value is not None else self.value(prop)

    def commit(
        self,
        call: GroundCall,
        reported: Iterable[tuple[GroundProp, bool]],
        *,
        sensed: Iterable[tuple[GroundProp, bool]] = (),
    ) -> list[MonitorViolation]:
        """Record the call with its effects: sensed values override the tool's own report.

        ``reported`` must stay within the call's declared effects (:meth:`check_report`);
        ``sensed`` may be any declared fluent. Raises EffectReportError without recording
        anything when either is invalid.
        """
        self._require_started()
        reported_map = self.check_report(call, reported)
        sensed_map: dict[GroundProp, bool] = {}
        for prop, value in sensed:
            self._check_fluent(prop)
            sensed_map[prop] = bool(value)
        resources = self._call_resources(call)
        for prop in (*reported_map, *sensed_map):
            decl = self._registry.prop(prop.name)
            resources.extend(zip(prop.args, decl.params, strict=True))
        self.register_resources(resources)
        event = self._registry.ground_event(call.tool, call.arg_map())
        processed = self._advance(event, {**reported_map, **sensed_map})
        return self._collect("after_execution", call.label(), processed)

    def _advance(self, event: GroundProp, updates: Mapping[GroundProp, bool]) -> list[str]:
        changed = [prop for prop, value in updates.items() if self.value(prop) != value]
        touched = self._touched(event, changed)
        position = len(self._events)
        self._events.append(event)
        self._event_positions.setdefault(event, []).append(position)
        for prop in changed:
            self._change_positions.setdefault(prop, []).append(position)
            self._change_values.setdefault(prop, []).append(updates[prop])
        self._values.update(updates)
        processed: list[str] = []
        for key in sorted(touched):
            live = self._live[key]
            if live.violated:
                self._unstable.discard(key)
                continue
            letter = self._letter(live.monitor, event, self.value)
            bits = live.monitor.step(live.bits, letter)
            self._charge()
            live.stable = letter == live.letter and bits == live.bits
            live.letter, live.bits = letter, bits
            if live.stable:
                self._unstable.discard(key)
            else:
                self._unstable.add(key)
            processed.append(key)
        return processed

    def finish(self) -> list[MonitorViolation]:
        self._require_started()
        return self._collect("at_end", "end", self._advance(END, {}))


class ToolGateway:
    """Executable tools, admitted only if registered and pinned to their declaration hash."""

    def __init__(self, registry: EffectRegistry, tools: Iterable[ExecutableTool]) -> None:
        self._registry = registry
        self._tools: dict[str, ExecutableTool] = {}
        for tool in tools:
            if not registry.has_tool(tool.name):
                raise ToolPinError(f"tool {tool.name} is not in the signed registry")
            if tool.declared_sha256 != registry.tool_sha256(tool.name):
                raise ToolPinError(
                    f"tool {tool.name} is pinned to a different declaration than the registry"
                )
            if tool.name in self._tools:
                raise ToolPinError(f"tool {tool.name} is provided twice")
            self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, call: GroundCall) -> Observation:
        tool = self._tools.get(call.tool)
        if tool is None:
            raise UnknownToolError(f"no executable implementation for {call.tool}")
        return tool(call.arg_map())


@dataclass
class RunResult:
    status: Literal["completed", "blocked", "violated", "aborted", "refused"]
    plan_sha256: str
    executed: list[str] = field(default_factory=list)
    blocked: MonitorViolation | None = None
    violations: list[MonitorViolation] = field(default_factory=list)
    waived: list[MonitorViolation] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
    misreports: list[str] = field(default_factory=list)
    outputs: list[UntrustedData] = field(default_factory=list)
    reason: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "plan_sha256": self.plan_sha256,
            "executed": list(self.executed),
            "blocked": self.blocked.to_json() if self.blocked else None,
            "violations": [v.to_json() for v in self.violations],
            "waived": [v.to_json() for v in self.waived],
            "divergences": list(self.divergences),
            "misreports": list(self.misreports),
            "reason": self.reason,
        }


class _Stop(Exception):  # noqa: N818 - internal control flow, never escapes run()
    pass


class _Refused(Exception):  # noqa: N818 - internal control flow, never escapes run()
    pass


@dataclass
class _Run:
    """Mutable state of one execution."""

    collections: Mapping[str, list[str]]
    monitor: RuntimeMonitor
    result: RunResult


OpaqueOracle = Callable[[str, str], bool]
CollectionProvider = Callable[[str], Sequence[str]]


def _flag(value: bool) -> str:
    return "true" if value else "false"


class PlanExecutor:
    """Execute a verified plan step by step under the runtime monitor."""

    def __init__(
        self,
        registry: EffectRegistry,
        specs: SpecSet,
        gateway: ToolGateway,
        *,
        opaque: OpaqueOracle,
        collections: CollectionProvider,
        audit: AuditLog | None = None,
        principal: str = "agent:executor",
        sensor: Sensor | None = None,
        overrides: OverrideGate | None = None,
        limits: CheckLimits | None = None,
    ) -> None:
        self._sensor = sensor
        self._registry = registry
        self._specs = specs
        self._gateway = gateway
        self._opaque = opaque
        self._collections = collections
        self._audit = audit
        self._principal = principal
        self._overrides = overrides
        self._checker = ModelChecker(registry, specs, limits)

    def _log(self, action: str, data: Mapping[str, JsonValue]) -> None:
        if self._audit is not None:
            self._audit.append(self._principal, action, dict(data))

    def _authorize(self, plan: Plan, override: str | None) -> frozenset[str]:
        """The static layer: the plan must verify, or an override must cover its violations.

        Returns the waived instance keys. The check runs here, against the executor's own specs
        and registry, so no caller can skip it or hand in a result for another plan; waivers
        only come from consuming an approved override that is bound to this plan's hash.
        """
        try:
            result = self._checker.check(plan)
        except (PlanError, CheckBudgetExceededError) as exc:
            raise _Refused(f"the plan could not be verified: {exc}") from exc
        if result.ok:
            return frozenset()
        violated = ", ".join(result.violated_specs())
        if override is None:
            raise _Refused(
                f"the plan violates {violated}; repair it or run it with an approved override"
            )
        if self._overrides is None:
            raise _Refused("an override was given but this executor has no override gate")
        try:
            grant = self._overrides.consume(plan, override, principal=self._principal)
        except OverrideError as exc:
            raise _Refused(f"override refused: {exc}") from exc
        if grant.plan_sha256 != result.plan_sha256:
            raise _Refused("override refused: it was granted for a different plan")
        uncovered = sorted(set(result.violated_instances()) - grant.waived)
        if uncovered:
            raise _Refused(f"the override does not cover {', '.join(uncovered)}")
        return grant.waived

    def _resolve_collections(self, plan: Plan) -> dict[str, list[str]]:
        bounds = collection_bounds(plan)
        constants = _plan_constants(plan.steps) | set(self._specs.constants())
        resolved: dict[str, list[str]] = {}
        seen: set[str] = set()
        for label in sorted(bounds):
            items = list(self._collections(label))
            if len(items) > bounds[label]:
                raise _Stop(f"collection {label} has {len(items)} items, bound is {bounds[label]}")
            invalid = [item for item in items if not is_valid_resource(item)]
            if invalid:
                raise _Stop(f"collection {label} has an invalid item ({len(invalid)} in total)")
            if len(set(items)) != len(items):
                raise _Stop(f"collection {label} contains duplicates")
            clash = (set(items) & seen) | (set(items) & constants)
            if clash:
                raise _Stop(f"collection {label} aliases other resources: {sorted(clash)}")
            seen |= set(items)
            resolved[label] = items
        return resolved

    def run(self, plan: Plan, *, override: str | None = None) -> RunResult:
        """Verify, then execute ``plan``. ``override`` names an approved override request."""
        result = RunResult(status="completed", plan_sha256=plan_sha256(plan))
        self._log("run.started", {"plan_sha256": result.plan_sha256, "override": override})
        try:
            waived = self._authorize(plan, override)
            monitor = RuntimeMonitor(
                self._registry, self._specs, waived=waived, sensor=self._sensor
            )
            collections = self._resolve_collections(plan)
            resources = self._plan_resources(plan.steps, {}, collections)
            for violation in monitor.start(sorted(resources)):
                self._record(result, violation)
            self._block(plan.steps, "steps", {}, _Run(collections, monitor, result))
            for violation in monitor.finish():
                self._record(result, violation)
        except _Refused as refused:
            result.status = "refused"
            result.reason = str(refused)
        except _Stop as stop:
            if result.status == "completed":
                result.status = "aborted"
            result.reason = str(stop)
        except (SensorError, InstanceBudgetExceededError, MonitorBudgetExceededError) as exc:
            result.status = "aborted"
            result.reason = str(exc)
        if result.status == "completed" and result.violations:
            result.status = "violated"
            result.reason = "end state violates a spec"
        self._log(
            "run.finished",
            {"status": result.status, "reason": result.reason, "executed": len(result.executed)},
        )
        return result

    def _plan_resources(
        self,
        nodes: tuple[Node, ...],
        env: Mapping[str, str],
        collections: Mapping[str, list[str]],
    ) -> set[tuple[str, str]]:
        """Every (resource, sort) the plan mentions, as the checker's active domain."""
        found: set[tuple[str, str]] = set()
        for node in nodes:
            if isinstance(node, ToolCall):
                sorts = self._registry.param_sorts(node.tool)
                found.update((_resolve(arg.value, env), sorts[arg.name]) for arg in node.args)
            elif isinstance(node, If):
                if not isinstance(node.cond, OpaqueGuard) and self._registry.has_prop(
                    node.cond.prop
                ):
                    params = self._registry.prop(node.cond.prop).params
                    for value, sort in zip(node.cond.args, params, strict=False):
                        found.add((_resolve(value, env), sort))
                found |= self._plan_resources(node.then, env, collections)
                found |= self._plan_resources(node.otherwise, env, collections)
            else:
                items = (
                    list(node.items)
                    if node.items is not None
                    else collections[node.collection or ""]
                )
                for item in items:
                    found |= self._plan_resources(node.body, {**env, node.var: item}, collections)
        return found

    def _record(self, result: RunResult, violation: MonitorViolation) -> None:
        if violation.waived:
            result.waived.append(violation)
            self._log("monitor.waived", violation.to_json())
            return
        result.violations.append(violation)
        self._log("monitor.violation", violation.to_json())

    def _block(self, nodes: tuple[Node, ...], path: str, env: Mapping[str, str], run: _Run) -> None:
        for index, node in enumerate(nodes):
            here = f"{path}/{index}"
            if isinstance(node, ToolCall):
                self._call(node, here, env, run)
            elif isinstance(node, If):
                if isinstance(node.cond, OpaqueGuard):
                    taken = self._opaque(node.cond.label, here)
                else:
                    prop = GroundProp(
                        node.cond.prop, tuple(_resolve(value, env) for value in node.cond.args)
                    )
                    taken = run.monitor.value(prop) != node.cond.negate
                branch = "then" if taken else "otherwise"
                body = node.then if taken else node.otherwise
                self._block(body, f"{here}/{branch}", env, run)
            else:
                self._loop(node, here, env, run)

    def _loop(self, node: ForEach, path: str, env: Mapping[str, str], run: _Run) -> None:
        items = (
            list(node.items) if node.items is not None else run.collections[node.collection or ""]
        )
        for item in items:
            self._block(node.body, f"{path}/body", {**env, node.var: item}, run)

    def _ground(self, node: ToolCall, path: str, env: Mapping[str, str]) -> GroundCall:
        # The plan was type-checked against the registry before the run, so the arguments match.
        given = node.arg_map()
        return GroundCall(
            tool=node.tool,
            args=tuple(
                (param.name, _resolve(given[param.name], env))
                for param in self._registry.tool(node.tool).params
            ),
            path=path,
        )

    def _gate(self, call: GroundCall, monitor: RuntimeMonitor, result: RunResult) -> None:
        """Block the call if a declared outcome would violate a spec that is not waived."""
        blocking = monitor.precheck(call)
        for violation in blocking:
            if violation.waived:
                result.waived.append(violation)
                self._log("monitor.waived", violation.to_json())
        monitor.acknowledge(violation.instance for violation in blocking if violation.waived)
        hard = [violation for violation in blocking if not violation.waived]
        if hard:
            result.status = "blocked"
            result.blocked = hard[0]
            result.violations.extend(hard)
            self._log(
                "monitor.blocked",
                {"step": call.label(), "specs": [*sorted({v.spec for v in hard})]},
            )
            raise _Stop(f"blocked {call.label()}: would violate {hard[0].instance}")

    def _observe(
        self, call: GroundCall, observation: Observation, monitor: RuntimeMonitor
    ) -> tuple[dict[GroundProp, bool], dict[GroundProp, bool], list[str]]:
        """The tool's validated report, the sensor's readings, and where the two disagree."""
        reported = monitor.check_report(call, observation.effects)
        sensed: dict[GroundProp, bool] = {}
        if monitor.has_sensor:
            # The sensor wins: re-read every fluent the specs depend on and every fluent the call
            # may change or reported. Changes no tool reported (a lock lease that expired, a
            # concurrent writer) reach the monitor, and so does a report that is not true.
            wanted = (
                set(monitor.tracked_fluents())
                | self._registry.declared_updates(call.tool, call.arg_map())
                | set(reported)
            )
            sensed = monitor.sense(sorted(wanted))
        misreports = [
            f"{call.label()}: {prop} reported {_flag(value)}, sensed {_flag(sensed[prop])}"
            for prop, value in sorted(reported.items())
            if prop in sensed and sensed[prop] != value
        ]
        return reported, sensed, misreports

    def _call(self, node: ToolCall, path: str, env: Mapping[str, str], run: _Run) -> None:
        monitor, result = run.monitor, run.result
        if not self._gateway.has(node.tool):
            result.status = "aborted"
            raise _Stop(f"tool {node.tool} has no pinned implementation")
        call = self._ground(node, path, env)
        self._gate(call, monitor, result)
        try:
            observation = self._gateway.execute(call)
        except Exception as exc:
            result.status = "aborted"
            self._log("tool.failed", {"step": call.label(), "error": type(exc).__name__})
            raise _Stop(f"tool {call.tool} failed") from exc
        try:
            reported, sensed, misreports = self._observe(call, observation, monitor)
        except EffectReportError as exc:
            result.status = "aborted"
            self._log("tool.invalid_report", {"step": call.label(), "error": str(exc)[:300]})
            raise _Stop(str(exc)) from exc
        observed = {**reported, **sensed}
        diverged = bool(misreports) or self._diverged(call, observation, observed, monitor)
        if diverged:
            result.divergences.append(call.label())
        output = wrap_tool_result(call.tool, observation.output)
        result.outputs.append(output)
        result.executed.append(call.label())
        self._log(
            "tool.executed",
            {
                "step": call.label(),
                "outcome": observation.outcome,
                "diverged": diverged,
                "output_sha256": sha256_hex(output.content),
            },
        )
        after = monitor.commit(call, tuple(reported.items()), sensed=tuple(sensed.items()))
        for violation in after:
            self._record(result, violation)
        if misreports:
            result.misreports.extend(misreports)
            self._log("tool.misreported", {"step": call.label(), "misreports": [*misreports]})
        if any(not violation.waived for violation in after):
            result.status = "violated"
            raise _Stop(f"observed effects of {call.label()} violated a spec")
        if misreports:
            result.status = "aborted"
            raise _Stop(
                f"{call.label()} misreported its effects: the sensor contradicts "
                f"{len(misreports)} reported value(s), so its report cannot be trusted"
            )

    def _diverged(
        self,
        call: GroundCall,
        observation: Observation,
        observed: Mapping[GroundProp, bool],
        monitor: RuntimeMonitor,
    ) -> bool:
        """True unless the observed state change is exactly one declared outcome's effects.

        The reported outcome must be declared, every effect it declares must be observed, and
        every other observed fluent must have kept the value the monitor had before the call.
        """
        for outcome in self._registry.ground_outcomes(call.tool, call.arg_map()):
            if outcome.name != observation.outcome:
                continue
            declared = dict(outcome.updates)
            explained = all(observed.get(prop) == value for prop, value in declared.items())
            unchanged = all(
                value == monitor.value(prop)
                for prop, value in observed.items()
                if prop not in declared
            )
            return not (explained and unchanged)
        return True


def _read(sensor: Sensor, prop: GroundProp) -> bool:
    try:
        return bool(sensor(prop))
    except Exception as exc:
        raise SensorError(f"the sensor could not read {prop}") from exc


def _plan_constants(steps: tuple[Node, ...]) -> set[str]:
    """Every resource constant the plan names: call arguments, guard arguments, listed items."""
    found: set[str] = set()
    stack: list[tuple[Node, ...]] = [steps]
    while stack:
        for node in stack.pop():
            if isinstance(node, ToolCall):
                found.update(arg.value for arg in node.args if not arg.value.startswith("$"))
            elif isinstance(node, If):
                if not isinstance(node.cond, OpaqueGuard):
                    found.update(value for value in node.cond.args if not value.startswith("$"))
                stack.extend((node.then, node.otherwise))
            else:
                found.update(node.items or ())
                stack.append(node.body)
    return found


def _resolve(value: str, env: Mapping[str, str]) -> str:
    return env[value[1:]] if value.startswith("$") else value


def describe_outputs(outputs: Iterable[UntrustedData]) -> str:
    """Render tool outputs for a prompt: every one inside its own untrusted-data envelope."""
    return "\n".join(item.render() for item in outputs)
