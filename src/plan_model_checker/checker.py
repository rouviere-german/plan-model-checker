"""Explicit-state model checking of plans against past-time specs, with shortest counterexamples.

Pipeline for one plan:

1. **Limits and typing.** Structural budgets (:func:`~plan_model_checker.ir.check_plan_limits`),
   then every call is resolved against the signed effect registry (unregistered tools, wrong
   parameters and sort clashes are rejected).
2. **Collection sizes.** Each data-dependent collection label gets a size from 0 to its declared
   bound; every combination is checked (a collection is a snapshot, so two loops over the same
   label see the same items).
3. **Control-flow graph.** Loops are unrolled, loop variables substituted, and branches become
   edges: guarded edges for ``If`` on a tracked proposition, unguarded choice edges for opaque
   predicates. Tool calls are action edges; each declared outcome is a separate transition.
4. **Spec instances.** Specs are grounded over the resources the plan mentions. An instance that
   holds on every trace for syntactic reasons is skipped: events no call in the graph emits are
   false everywhere and fluents no call can change keep their initial value, and if folding
   those constants into the instance (:mod:`~plan_model_checker.spec.simplify`) gives ``true``
   no search is needed (``G(delete[r] -> ...)`` for an ``r`` that is never deleted).
5. **Search.** For each instance, a 0-1 breadth-first search over states ``(control point,
   relevant fluents, monitor state)``: choice edges cost 0, actions cost 1, so the first violating
   terminal state popped is a shortest violating trace. States are memoized, so paths that
   reach the same state are explored once; only fluents that the instance or a guard reads are
   tracked (cone of influence). Once a monotone spec is false, its monitor state collapses to a
   single sink.

The result carries, per violated spec, the shortest counterexample: the tool calls with the
outcome each took, the opaque-branch decisions, the collection sizes and the letters seen by
the monitor, so it can be replayed against the plan by an independent interpreter.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from plan_model_checker.effects import EffectRegistry, UnknownToolError
from plan_model_checker.ir import (
    ForEach,
    If,
    Node,
    OpaqueGuard,
    Plan,
    PlanError,
    PlanLimits,
    PropGuard,
    ToolCall,
    check_plan_limits,
    plan_sha256,
)
from plan_model_checker.props import END, START, GroundProp
from plan_model_checker.spec.monitor import CompiledMonitor, compile_monitor
from plan_model_checker.spec.simplify import TRUE, simplify
from plan_model_checker.spec.specs import InstanceBudgetExceededError, SpecInstance, SpecSet

__all__ = [
    "CheckBudgetExceededError",
    "CheckLimits",
    "CheckResult",
    "ControlFlowGraph",
    "Counterexample",
    "Decision",
    "Edge",
    "GroundCall",
    "GroundGuard",
    "ModelChecker",
    "TraceStep",
    "Violation",
    "build_cfg",
    "collection_bounds",
    "fixed_props",
    "size_assignments",
]

_SINK = -1


class CheckBudgetExceededError(Exception):
    """The check would explore more states, instances or size assignments than allowed."""

    def __init__(self, dimension: str, limit: int) -> None:
        super().__init__(f"{dimension} budget of {limit} exceeded")
        self.dimension = dimension
        self.limit = limit


@dataclass(frozen=True, slots=True)
class CheckLimits:
    plan: PlanLimits = field(default_factory=PlanLimits)
    max_states: int = 250_000
    max_instances: int = 5_000
    max_size_assignments: int = 256
    max_collections: int = 3


@dataclass(frozen=True, slots=True)
class GroundCall:
    tool: str
    args: tuple[tuple[str, str], ...]
    path: str

    def arg_map(self) -> dict[str, str]:
        return dict(self.args)

    def label(self) -> str:
        inner = ", ".join(f"{name}={value}" for name, value in self.args)
        return f"{self.tool}({inner})"


@dataclass(frozen=True, slots=True)
class GroundGuard:
    prop: GroundProp
    expected: bool


@dataclass(frozen=True, slots=True)
class Edge:
    target: int
    call: GroundCall | None = None
    guard: GroundGuard | None = None
    decision: str | None = None
    site: str = ""


@dataclass(frozen=True, slots=True)
class ControlFlowGraph:
    edges: tuple[tuple[Edge, ...], ...]
    entry: int
    exit: int
    universe: Mapping[str, tuple[str, ...]]
    guard_props: frozenset[GroundProp]
    sizes: tuple[tuple[str, int], ...]

    @property
    def node_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True, slots=True)
class TraceStep:
    call: GroundCall
    outcome: str

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": self.call.tool,
            "args": dict(self.call.args),
            "path": self.call.path,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    site: str
    choice: str


@dataclass(frozen=True, slots=True)
class Counterexample:
    spec: str
    instance: SpecInstance
    sizes: tuple[tuple[str, int], ...]
    steps: tuple[TraceStep, ...]
    decisions: tuple[Decision, ...]
    letters: tuple[frozenset[GroundProp], ...]
    violation_position: int

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "instance": self.instance.key,
            "binding": dict(self.instance.binding),
            "collection_sizes": dict(self.sizes),
            "steps": [step.to_json() for step in self.steps],
            "decisions": [{"site": d.site, "choice": d.choice} for d in self.decisions],
            "positions": [sorted(str(prop) for prop in letter) for letter in self.letters],
            "violation_position": self.violation_position,
        }


@dataclass(frozen=True, slots=True)
class Violation:
    spec: str
    counterexample: Counterexample
    instance_keys: tuple[str, ...]

    @property
    def instances_violated(self) -> int:
        return len(self.instance_keys)

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "violated_instances": list(self.instance_keys),
            "counterexample": self.counterexample.to_json(),
        }


@dataclass(frozen=True, slots=True)
class CheckResult:
    verdict: Literal["safe", "violation"]
    violations: tuple[Violation, ...]
    states_explored: int
    instances_checked: int
    size_assignments: int
    plan_sha256: str
    seconds: float
    instances_skipped: int = 0

    @property
    def ok(self) -> bool:
        return self.verdict == "safe"

    def violated_specs(self) -> tuple[str, ...]:
        return tuple(violation.spec for violation in self.violations)

    def violated_instances(self) -> tuple[str, ...]:
        return tuple(key for violation in self.violations for key in violation.instance_keys)

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "plan_sha256": self.plan_sha256,
            "states_explored": self.states_explored,
            "instances_checked": self.instances_checked,
            "instances_skipped": self.instances_skipped,
            "size_assignments": self.size_assignments,
            "violations": [violation.to_json() for violation in self.violations],
        }


# --- grounding --------------------------------------------------------------------------------


def collection_bounds(plan: Plan) -> dict[str, int]:
    """Collection label -> declared bound. Loops over one label must declare the same bound."""
    bounds: dict[str, int] = {}
    stack: list[tuple[Node, ...]] = [plan.steps]
    while stack:
        for node in stack.pop():
            if isinstance(node, If):
                stack.extend((node.then, node.otherwise))
            elif isinstance(node, ForEach):
                if node.collection is not None:
                    previous = bounds.setdefault(node.collection, node.max_iterations)
                    if previous != node.max_iterations:
                        raise PlanError(
                            f"loops over collection {node.collection} declare different bounds"
                        )
                stack.append(node.body)
    return bounds


def size_assignments(bounds: Mapping[str, int]) -> list[tuple[tuple[str, int], ...]]:
    labels = sorted(bounds)
    return [
        tuple(zip(labels, sizes, strict=True))
        for sizes in itertools.product(*(range(bounds[label] + 1) for label in labels))
    ]


def symbolic_item(collection: str, index: int) -> str:
    return f"{collection}#{index}"


class _CfgBuilder:
    def __init__(self, registry: EffectRegistry, sizes: Mapping[str, int]) -> None:
        self._registry = registry
        self._sizes = sizes
        self._edges: list[tuple[Edge, ...]] = []
        self._sorts: dict[str, str] = {}
        self._guard_props: set[GroundProp] = set()

    def new_node(self, edges: tuple[Edge, ...] = ()) -> int:
        self._edges.append(edges)
        return len(self._edges) - 1

    def _resolve(self, value: str, env: Mapping[str, str]) -> str:
        if value.startswith("$"):
            return env[value[1:]]
        return value

    def _note_sort(self, resource: str, sort: str) -> None:
        previous = self._sorts.setdefault(resource, sort)
        if previous != sort:
            raise PlanError(f"resource {resource} is used both as a {previous} and as a {sort}")

    def _ground_call(self, node: ToolCall, path: str, env: Mapping[str, str]) -> GroundCall:
        try:
            tool = self._registry.tool(node.tool)
        except UnknownToolError as exc:
            raise PlanError(f"{path}: tool {node.tool} is not registered") from exc
        given = node.arg_map()
        declared = [param.name for param in tool.params]
        if set(given) != set(declared):
            raise PlanError(f"{path}: {node.tool} takes parameters {declared}, got {sorted(given)}")
        args: list[tuple[str, str]] = []
        for param in tool.params:
            resource = self._resolve(given[param.name], env)
            self._note_sort(resource, param.sort)
            args.append((param.name, resource))
        return GroundCall(tool=node.tool, args=tuple(args), path=path)

    def _ground_guard(self, guard: PropGuard, path: str, env: Mapping[str, str]) -> GroundGuard:
        if not self._registry.has_prop(guard.prop):
            raise PlanError(f"{path}: guard proposition {guard.prop} is not declared")
        decl = self._registry.prop(guard.prop)
        if decl.kind != "fluent":
            raise PlanError(f"{path}: guards may only test fluents, {guard.prop} is an event")
        if len(guard.args) != len(decl.params):
            raise PlanError(f"{path}: guard {guard.prop} has the wrong number of arguments")
        values: list[str] = []
        for value, sort in zip(guard.args, decl.params, strict=True):
            resource = self._resolve(value, env)
            self._note_sort(resource, sort)
            values.append(resource)
        prop = GroundProp(guard.prop, tuple(values))
        self._guard_props.add(prop)
        return GroundGuard(prop=prop, expected=not guard.negate)

    def block(self, nodes: tuple[Node, ...], path: str, env: Mapping[str, str], succ: int) -> int:
        for index in reversed(range(len(nodes))):
            succ = self.node(nodes[index], f"{path}/{index}", env, succ)
        return succ

    def node(self, node: Node, path: str, env: Mapping[str, str], succ: int) -> int:
        if isinstance(node, ToolCall):
            call = self._ground_call(node, path, env)
            return self.new_node((Edge(target=succ, call=call),))
        if isinstance(node, If):
            then_entry = self.block(node.then, f"{path}/then", env, succ)
            else_entry = self.block(node.otherwise, f"{path}/otherwise", env, succ)
            if isinstance(node.cond, OpaqueGuard):
                return self.new_node(
                    (
                        Edge(target=then_entry, decision="then", site=path),
                        Edge(target=else_entry, decision="otherwise", site=path),
                    )
                )
            guard = self._ground_guard(node.cond, path, env)
            negated = GroundGuard(prop=guard.prop, expected=not guard.expected)
            return self.new_node(
                (
                    Edge(target=then_entry, guard=guard, site=path),
                    Edge(target=else_entry, guard=negated, site=path),
                )
            )
        if node.items is not None:
            items = list(node.items)
        else:
            if node.collection is None:  # pragma: no cover - excluded by the ForEach validator
                raise PlanError(f"{path}: loop has neither items nor a collection")
            items = [symbolic_item(node.collection, k) for k in range(self._sizes[node.collection])]
        entry = succ
        for item in reversed(items):
            entry = self.block(node.body, f"{path}/body", {**env, node.var: item}, entry)
        return entry

    def build(self, plan: Plan, sizes: tuple[tuple[str, int], ...]) -> ControlFlowGraph:
        exit_pc = self.new_node()
        entry = self.block(plan.steps, "steps", {}, exit_pc)
        universe: dict[str, list[str]] = {}
        for resource, sort in self._sorts.items():
            universe.setdefault(sort, []).append(resource)
        return ControlFlowGraph(
            edges=tuple(self._edges),
            entry=entry,
            exit=exit_pc,
            universe={sort: tuple(sorted(items)) for sort, items in universe.items()},
            guard_props=frozenset(self._guard_props),
            sizes=sizes,
        )


def build_cfg(
    plan: Plan, registry: EffectRegistry, sizes: tuple[tuple[str, int], ...] = ()
) -> ControlFlowGraph:
    return _CfgBuilder(registry, dict(sizes)).build(plan, sizes)


def fixed_props(
    cfg: ControlFlowGraph, registry: EffectRegistry
) -> Callable[[GroundProp], bool | None]:
    """Propositions with one value at every position of every trace of ``cfg`` (else None).

    An event that no call in the graph emits is always false; a fluent that no declared outcome
    of any call can set keeps its initial value.
    """
    events = {START, END}
    changeable: set[GroundProp] = set()
    for edges in cfg.edges:
        for edge in edges:
            if edge.call is not None:
                args = edge.call.arg_map()
                events.add(registry.ground_event(edge.call.tool, args))
                changeable |= registry.declared_updates(edge.call.tool, args)

    def fixed(prop: GroundProp) -> bool | None:
        if registry.prop(prop.name).kind == "event":
            return None if prop in events else False
        return None if prop in changeable else registry.initial(prop)

    return fixed


# --- search -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Transition:
    """Precomputed effect of one (action edge, outcome) on the tracked state."""

    target: int
    set_mask: int
    clear_mask: int
    event_letter: int
    step: TraceStep


class _InstanceSearch:
    """0-1 BFS for one spec instance on one control-flow graph."""

    def __init__(
        self,
        cfg: ControlFlowGraph,
        registry: EffectRegistry,
        instance: SpecInstance,
        monitor: CompiledMonitor,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.instance = instance
        self.monitor = monitor
        fluents = sorted(
            {
                prop
                for prop in monitor.props
                if registry.has_prop(prop.name) and registry.prop(prop.name).kind == "fluent"
            }
            | set(cfg.guard_props)
        )
        self.fluents = fluents
        self.fluent_bit = {prop: 1 << index for index, prop in enumerate(fluents)}
        # (fluent bit position, monitor letter bit) pairs, used to project fluents to a letter.
        self.fluent_to_letter = [
            (fluents.index(prop), 1 << index)
            for index, prop in enumerate(monitor.props)
            if prop in self.fluent_bit
        ]
        self.letter_bit = {prop: 1 << index for index, prop in enumerate(monitor.props)}
        self.initial_fluents = sum(
            bit for prop, bit in self.fluent_bit.items() if registry.initial(prop)
        )
        self.transitions: dict[int, list[_Transition]] = {}
        for pc, edges in enumerate(cfg.edges):
            for edge in edges:
                if edge.call is not None:
                    self.transitions[pc] = self._transitions_for(edge.target, edge.call)

    def _transitions_for(self, target: int, call: GroundCall) -> list[_Transition]:
        args = call.arg_map()
        event = self.registry.ground_event(call.tool, args)
        event_letter = self.letter_bit.get(event, 0)
        result: list[_Transition] = []
        for outcome in self.registry.ground_outcomes(call.tool, args):
            set_mask = clear_mask = 0
            for prop, value in outcome.updates:
                bit = self.fluent_bit.get(prop, 0)
                if value:
                    set_mask |= bit
                    clear_mask &= ~bit
                else:
                    clear_mask |= bit
                    set_mask &= ~bit
            result.append(
                _Transition(
                    target=target,
                    set_mask=set_mask,
                    clear_mask=clear_mask,
                    event_letter=event_letter,
                    step=TraceStep(call=call, outcome=outcome.name),
                )
            )
        return result

    def letter(self, fluents: int, event_letter: int) -> int:
        value = event_letter
        for fluent_index, letter_bit in self.fluent_to_letter:
            if (fluents >> fluent_index) & 1:
                value |= letter_bit
        return value

    def advance(self, state: int, letter: int) -> int:
        if state == _SINK:
            return _SINK
        now = self.monitor.step(state, letter)
        return self.monitor.state(now) if self.monitor.holds(now) else _SINK

    def search(self, budget: list[int]) -> tuple[Counterexample | None, int]:
        cfg = self.cfg
        start_letter = self.letter(self.initial_fluents, self.letter_bit.get(START, 0))
        first = self.monitor.first(start_letter)
        start_state = self.monitor.state(first) if self.monitor.holds(first) else _SINK
        start = (cfg.entry, self.initial_fluents, start_state)
        dist: dict[tuple[int, int, int], int] = {start: 0}
        parent: dict[
            tuple[int, int, int], tuple[tuple[int, int, int], TraceStep | None, Decision | None]
        ] = {}
        queue: deque[tuple[int, int, int]] = deque([start])
        done: set[tuple[int, int, int]] = set()
        explored = 0
        end_bit = self.letter_bit.get(END, 0)
        while queue:
            state = queue.popleft()
            if state in done:
                continue
            done.add(state)
            explored += 1
            budget[0] -= 1
            if budget[0] < 0:
                raise CheckBudgetExceededError("states", budget[1])
            pc, fluents, mstate = state
            base = dist[state]
            if pc == cfg.exit:
                final = self.advance(mstate, self.letter(fluents, end_bit))
                if final == _SINK:
                    return self._counterexample(state, parent), explored
                continue
            action = self.transitions.get(pc)
            if action is not None:
                for transition in action:
                    new_fluents = (fluents | transition.set_mask) & ~transition.clear_mask
                    letter = self.letter(new_fluents, transition.event_letter)
                    successor = (transition.target, new_fluents, self.advance(mstate, letter))
                    if successor not in dist or dist[successor] > base + 1:
                        dist[successor] = base + 1
                        parent[successor] = (state, transition.step, None)
                        queue.append(successor)
                continue
            for edge in cfg.edges[pc]:
                if edge.guard is not None:
                    bit = self.fluent_bit[edge.guard.prop]
                    if bool(fluents & bit) != edge.guard.expected:
                        continue
                decision = (
                    Decision(site=edge.site, choice=edge.decision)
                    if edge.decision is not None
                    else None
                )
                successor = (edge.target, fluents, mstate)
                if successor not in dist or dist[successor] > base:
                    dist[successor] = base
                    parent[successor] = (state, None, decision)
                    queue.appendleft(successor)
        return None, explored

    def _counterexample(
        self,
        state: tuple[int, int, int],
        parent: Mapping[
            tuple[int, int, int], tuple[tuple[int, int, int], TraceStep | None, Decision | None]
        ],
    ) -> Counterexample:
        steps: list[TraceStep] = []
        decisions: list[Decision] = []
        fluent_path: list[int] = [state[1]]
        cursor = state
        while cursor in parent:
            previous, step, decision = parent[cursor]
            if step is not None:
                steps.append(step)
                fluent_path.append(previous[1])
            if decision is not None:
                decisions.append(decision)
            cursor = previous
        steps.reverse()
        decisions.reverse()
        fluent_path.reverse()
        letters: list[frozenset[GroundProp]] = []
        events: list[GroundProp] = [START]
        events.extend(
            self.registry.ground_event(step.call.tool, step.call.arg_map()) for step in steps
        )
        events.append(END)
        positions = [*fluent_path, state[1]]
        for event, fluents in zip(events, positions, strict=True):
            true_fluents = {prop for prop, bit in self.fluent_bit.items() if fluents & bit}
            letters.append(frozenset({event, *true_fluents}))
        return Counterexample(
            spec=self.instance.spec,
            instance=self.instance,
            sizes=self.cfg.sizes,
            steps=tuple(steps),
            decisions=tuple(decisions),
            letters=tuple(letters),
            violation_position=self._first_false(letters),
        )

    def _first_false(self, letters: list[frozenset[GroundProp]]) -> int:
        bits = self.monitor.first(self.monitor.letter(letters[0]))
        if not self.monitor.holds(bits):
            return 0
        for position in range(1, len(letters)):
            bits = self.monitor.step(bits, self.monitor.letter(letters[position]))
            if not self.monitor.holds(bits):
                return position
        return len(letters) - 1  # pragma: no cover - only called on violating traces


class ModelChecker:
    """Check plans against a spec set, relative to a (verified) effect registry."""

    def __init__(
        self,
        registry: EffectRegistry,
        specs: SpecSet,
        limits: CheckLimits | None = None,
        *,
        skip_vacuous: bool = True,
    ) -> None:
        """``skip_vacuous=False`` searches every instance (the benchmark measures the gain)."""
        self.registry = registry
        self.specs = specs
        self.limits = limits or CheckLimits()
        self.skip_vacuous = skip_vacuous
        self._monitors: dict[object, CompiledMonitor] = {}

    def _monitor(self, instance: SpecInstance) -> CompiledMonitor:
        monitor = self._monitors.get(instance.final)
        if monitor is None:
            if len(self._monitors) > 50_000:
                self._monitors.clear()
            monitor = compile_monitor(instance.final)
            self._monitors[instance.final] = monitor
        return monitor

    def check(self, plan: Plan, spec_names: list[str] | None = None) -> CheckResult:
        started = time.perf_counter()
        specs = self.specs.select(spec_names)
        check_plan_limits(plan, self.limits.plan)
        bounds = collection_bounds(plan)
        if len(bounds) > self.limits.max_collections:
            raise PlanError(f"plan uses more than {self.limits.max_collections} collections")
        assignments = size_assignments(bounds)
        if len(assignments) > self.limits.max_size_assignments:
            raise CheckBudgetExceededError("size assignments", self.limits.max_size_assignments)
        budget = [self.limits.max_states, self.limits.max_states]
        best: dict[str, Counterexample] = {}
        violated_instances: dict[str, set[str]] = {}
        explored = 0
        instances_checked = 0
        instances_skipped = 0
        for sizes in assignments:
            cfg = build_cfg(plan, self.registry, sizes)
            remaining = self.limits.max_instances - instances_checked
            try:
                instances = specs.instances(cfg.universe, max_instances=max(remaining, 0))
            except InstanceBudgetExceededError as exc:
                raise CheckBudgetExceededError("instances", self.limits.max_instances) from exc
            instances_checked += len(instances)
            if instances_checked > self.limits.max_instances:
                raise CheckBudgetExceededError("instances", self.limits.max_instances)
            fixed = fixed_props(cfg, self.registry)
            for instance in instances:
                if self.skip_vacuous and simplify(instance.final, fixed) == TRUE:
                    instances_skipped += 1  # holds on every trace of this graph
                    continue
                search = _InstanceSearch(cfg, self.registry, instance, self._monitor(instance))
                counterexample, count = search.search(budget)
                explored += count
                if counterexample is None:
                    continue
                violated_instances.setdefault(instance.spec, set()).add(instance.key)
                current = best.get(instance.spec)
                if current is None or _shorter(counterexample, current):
                    best[instance.spec] = counterexample
        violations = tuple(
            Violation(
                spec=name,
                counterexample=best[name],
                instance_keys=tuple(sorted(violated_instances[name])),
            )
            for name in specs.names
            if name in best
        )
        return CheckResult(
            verdict="violation" if violations else "safe",
            violations=violations,
            states_explored=explored,
            instances_checked=instances_checked,
            size_assignments=len(assignments),
            plan_sha256=plan_sha256(plan),
            seconds=time.perf_counter() - started,
            instances_skipped=instances_skipped,
        )


def _shorter(candidate: Counterexample, current: Counterexample) -> bool:
    return (candidate.length, candidate.instance.key) < (current.length, current.instance.key)
