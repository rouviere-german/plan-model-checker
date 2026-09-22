"""Runtime enforcement: the same compiled specs, applied to what tools actually do.

The static check is only as good as the effect declarations. At runtime :class:`PlanExecutor`
walks the plan and routes every call through two layers:

1. :class:`ToolGateway` resolves the call to an executable tool that is registered *and* pinned
   to the exact declaration hash in the signed registry (unregistered or modified tools are
   refused at construction time).
2. :class:`RuntimeMonitor` keeps the observed fluent state and a live monitor per spec instance
   (instances are created as resources appear and replayed over the history, which keeps
   active-domain semantics exact). Before a call runs, it predicts the next position for every
   declared outcome of the call from the *observed* state and blocks the call if any spec would
   become false. After the call it applies the effects the tool actually reported, flags
   divergences from the declaration, and stops the run if the real effects broke a spec.

Branch guards on propositions read the observed state, so ``If(approved[p])`` at runtime means
what it meant to the checker. Collections are resolved once, at the start of a run, and refused
if they exceed their bound or alias other resources (the checker assumed fresh, distinct items).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import JsonValue

from plan_model_checker.agentic.audit import AuditLog
from plan_model_checker.agentic.envelope import UntrustedData, wrap_tool_result
from plan_model_checker.checker import GroundCall, collection_bounds
from plan_model_checker.core.canonical import sha256_hex
from plan_model_checker.effects import EffectRegistry, UnknownToolError
from plan_model_checker.ir import (
    ForEach,
    If,
    Node,
    OpaqueGuard,
    Plan,
    ToolCall,
    iter_calls,
    plan_sha256,
)
from plan_model_checker.props import END, START, GroundProp
from plan_model_checker.spec.monitor import CompiledMonitor, compile_monitor
from plan_model_checker.spec.specs import SpecInstance, SpecSet

__all__ = [
    "ExecutableTool",
    "MonitorViolation",
    "Observation",
    "PlanExecutor",
    "RunResult",
    "RuntimeMonitor",
    "ToolGateway",
    "ToolPinError",
    "describe_outputs",
]


class ToolPinError(Exception):
    """An executable tool is unregistered or pinned to a different declaration."""


@dataclass(frozen=True, slots=True)
class Observation:
    """What a tool reports after running: an outcome label, observed fluent values, output."""

    outcome: str
    effects: tuple[tuple[GroundProp, bool], ...] = ()
    output: JsonValue = None


class ExecutableTool(Protocol):
    name: str
    declared_sha256: str

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
    bits: int
    violated: bool = False


class RuntimeMonitor:
    """Parametric runtime monitor over observed events and fluent values."""

    def __init__(
        self, registry: EffectRegistry, specs: SpecSet, *, waived: Iterable[str] = ()
    ) -> None:
        self._registry = registry
        self._specs = specs
        self._waived = frozenset(waived)
        self._fluents: dict[GroundProp, bool] = {}
        self._history: list[tuple[GroundProp, dict[GroundProp, bool]]] = []
        self._universe: dict[str, set[str]] = {}
        self._live: dict[str, _Live] = {}
        self._compiled: dict[object, CompiledMonitor] = {}
        self._started = False

    # --- state --------------------------------------------------------------------------------

    @property
    def position(self) -> int:
        return len(self._history) - 1

    def value(self, prop: GroundProp) -> bool:
        return self._fluents.get(prop, self._registry.initial(prop))

    def instance_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._live))

    def tracked_fluents(self) -> tuple[GroundProp, ...]:
        """Fluents some live spec instance reads (what a sensor should refresh)."""
        found = {
            prop
            for live in self._live.values()
            for prop in live.monitor.props
            if self._is_fluent(prop)
        }
        return tuple(sorted(found))

    def _letter(
        self, monitor: CompiledMonitor, event: GroundProp, fluents: Mapping[GroundProp, bool]
    ) -> int:
        value = 0
        for index, prop in enumerate(monitor.props):
            if prop == event or (
                self._is_fluent(prop) and fluents.get(prop, self._registry.initial(prop))
            ):
                value |= 1 << index
        return value

    def _is_fluent(self, prop: GroundProp) -> bool:
        return (
            self._registry.has_prop(prop.name) and self._registry.prop(prop.name).kind == "fluent"
        )

    def _compile(self, instance: SpecInstance) -> CompiledMonitor:
        monitor = self._compiled.get(instance.final)
        if monitor is None:
            monitor = compile_monitor(instance.final)
            self._compiled[instance.final] = monitor
        return monitor

    def register_resources(self, resources: Iterable[tuple[str, str]]) -> None:
        """Add ``(resource, sort)`` pairs; new spec instances replay the history so far."""
        changed = False
        for resource, sort in resources:
            bucket = self._universe.setdefault(sort, set())
            if resource not in bucket:
                bucket.add(resource)
                changed = True
        if not changed and self._live:
            return
        universe = {sort: tuple(sorted(items)) for sort, items in self._universe.items()}
        for instance in self._specs.instances(universe):
            if instance.key in self._live:
                continue
            monitor = self._compile(instance)
            live = _Live(instance=instance, monitor=monitor, bits=0)
            for position, (event, fluents) in enumerate(self._history):
                letter = self._letter(monitor, event, fluents)
                live.bits = (
                    monitor.first(letter) if position == 0 else monitor.step(live.bits, letter)
                )
            self._live[instance.key] = live

    def acknowledge(self, keys: Iterable[str]) -> None:
        """Mark instances as already reported (a waived violation is logged once, not twice)."""
        for key in keys:
            live = self._live.get(key)
            if live is not None:
                live.violated = True

    def _call_resources(self, call: GroundCall) -> list[tuple[str, str]]:
        sorts = self._registry.param_sorts(call.tool)
        return [(value, sorts[name]) for name, value in call.args]

    # --- protocol -----------------------------------------------------------------------------

    def start(self, resources: Iterable[tuple[str, str]] = ()) -> list[MonitorViolation]:
        if self._started:
            raise RuntimeError("monitor already started")
        self._started = True
        self._history.append((START, dict(self._fluents)))
        self.register_resources(resources)
        return self._collect("after_execution", "start")

    def _collect(
        self, kind: Literal["after_execution", "at_end"], step: str
    ) -> list[MonitorViolation]:
        found: list[MonitorViolation] = []
        for key in sorted(self._live):
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

    def precheck(self, call: GroundCall) -> list[MonitorViolation]:
        """Violations the call would cause under any of its declared outcomes (nothing runs)."""
        if not self._started:
            raise RuntimeError("monitor not started")
        self.register_resources(self._call_resources(call))
        args = call.arg_map()
        event = self._registry.ground_event(call.tool, args)
        found: dict[str, MonitorViolation] = {}
        for outcome in self._registry.ground_outcomes(call.tool, args):
            predicted = dict(self._fluents)
            for prop, value in outcome.updates:
                predicted[prop] = value
            for key, live in self._live.items():
                if live.violated or key in found:
                    continue
                bits = live.monitor.step(live.bits, self._letter(live.monitor, event, predicted))
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

    def commit(
        self, call: GroundCall, observed: Sequence[tuple[GroundProp, bool]]
    ) -> list[MonitorViolation]:
        """Record the call with the effects that were actually observed."""
        if not self._started:
            raise RuntimeError("monitor not started")
        resources = self._call_resources(call)
        for prop, _ in observed:
            decl = self._registry.prop(prop.name)
            resources.extend(zip(prop.args, decl.params, strict=True))
        self.register_resources(resources)
        for prop, value in observed:
            self._fluents[prop] = value
        event = self._registry.ground_event(call.tool, call.arg_map())
        self._history.append((event, dict(self._fluents)))
        for live in self._live.values():
            live.bits = live.monitor.step(
                live.bits, self._letter(live.monitor, event, self._fluents)
            )
        return self._collect("after_execution", call.label())

    def finish(self) -> list[MonitorViolation]:
        if not self._started:
            raise RuntimeError("monitor not started")
        self._history.append((END, dict(self._fluents)))
        for live in self._live.values():
            live.bits = live.monitor.step(live.bits, self._letter(live.monitor, END, self._fluents))
        return self._collect("at_end", "end")


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
    status: Literal["completed", "blocked", "violated", "aborted"]
    plan_sha256: str
    executed: list[str] = field(default_factory=list)
    blocked: MonitorViolation | None = None
    violations: list[MonitorViolation] = field(default_factory=list)
    waived: list[MonitorViolation] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)
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
            "reason": self.reason,
        }


class _Stop(Exception):  # noqa: N818 - internal control flow, never escapes run()
    pass


@dataclass
class _Run:
    """Mutable state of one execution."""

    collections: Mapping[str, list[str]]
    monitor: RuntimeMonitor
    result: RunResult


OpaqueOracle = Callable[[str, str], bool]
CollectionProvider = Callable[[str], Sequence[str]]
Sensor = Callable[[GroundProp], bool]


class PlanExecutor:
    """Execute a plan step by step under the runtime monitor."""

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
        waived: Iterable[str] = (),
        sensor: Sensor | None = None,
    ) -> None:
        self._sensor = sensor
        self._registry = registry
        self._specs = specs
        self._gateway = gateway
        self._opaque = opaque
        self._collections = collections
        self._audit = audit
        self._principal = principal
        self._waived = frozenset(waived)

    def _log(self, action: str, data: Mapping[str, JsonValue]) -> None:
        if self._audit is not None:
            self._audit.append(self._principal, action, dict(data))

    def _resolve_collections(self, plan: Plan) -> dict[str, list[str]]:
        bounds = collection_bounds(plan)
        constants = {
            arg.value
            for _, call in iter_calls(plan.steps)
            for arg in call.args
            if not arg.value.startswith("$")
        } | set(self._specs.constants())
        resolved: dict[str, list[str]] = {}
        seen: set[str] = set()
        for label in sorted(bounds):
            items = list(self._collections(label))
            if len(items) > bounds[label]:
                raise _Stop(f"collection {label} has {len(items)} items, bound is {bounds[label]}")
            if len(set(items)) != len(items):
                raise _Stop(f"collection {label} contains duplicates")
            clash = (set(items) & seen) | (set(items) & constants)
            if clash:
                raise _Stop(f"collection {label} aliases other resources: {sorted(clash)}")
            seen |= set(items)
            resolved[label] = items
        return resolved

    def run(self, plan: Plan) -> RunResult:
        result = RunResult(status="completed", plan_sha256=plan_sha256(plan))
        monitor = RuntimeMonitor(self._registry, self._specs, waived=self._waived)
        self._log("run.started", {"plan_sha256": result.plan_sha256})
        try:
            collections = self._resolve_collections(plan)
            resources = self._plan_resources(plan.steps, {}, collections)
            for violation in monitor.start(sorted(resources)):
                self._record(result, violation)
            self._block(plan.steps, "steps", {}, _Run(collections, monitor, result))
            for violation in monitor.finish():
                self._record(result, violation)
        except _Stop as stop:
            if result.status == "completed":
                result.status = "aborted"
            result.reason = str(stop)
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
                if not self._registry.has_tool(node.tool):
                    continue
                sorts = self._registry.param_sorts(node.tool)
                for arg in node.args:
                    if arg.name in sorts:
                        found.add((_resolve(arg.value, env), sorts[arg.name]))
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

    def _call(self, node: ToolCall, path: str, env: Mapping[str, str], run: _Run) -> None:
        monitor, result = run.monitor, run.result
        if not self._registry.has_tool(node.tool) or not self._gateway.has(node.tool):
            result.status = "aborted"
            raise _Stop(f"tool {node.tool} is not registered or has no pinned implementation")
        given = node.arg_map()
        params = [param.name for param in self._registry.tool(node.tool).params]
        if set(given) != set(params):
            result.status = "aborted"
            raise _Stop(f"{path}: arguments do not match the declaration of {node.tool}")
        call = GroundCall(
            tool=node.tool,
            args=tuple((name, _resolve(given[name], env)) for name in params),
            path=path,
        )
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
        try:
            observation = self._gateway.execute(call)
        except Exception as exc:
            result.status = "aborted"
            self._log("tool.failed", {"step": call.label(), "error": type(exc).__name__})
            raise _Stop(f"tool {call.tool} failed") from exc
        observed = dict(observation.effects)
        if self._sensor is not None:
            # Re-read what the specs depend on: changes no tool reported (a lock lease that
            # expired, a concurrent writer) still reach the monitor.
            for prop in monitor.tracked_fluents():
                observed.setdefault(prop, self._sensor(prop))
        diverged = self._diverged(call, observation, observed, monitor)
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
        after = monitor.commit(call, tuple(observed.items()))
        for violation in after:
            self._record(result, violation)
        if any(not violation.waived for violation in after):
            result.status = "violated"
            raise _Stop(f"observed effects of {call.label()} violated a spec")

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


def _resolve(value: str, env: Mapping[str, str]) -> str:
    return env[value[1:]] if value.startswith("$") else value


def describe_outputs(outputs: Iterable[UntrustedData]) -> str:
    """Render tool outputs for a prompt: every one inside its own untrusted-data envelope."""
    return "\n".join(item.render() for item in outputs)
