"""Brute-force oracle: enumerate every trace of a plan and evaluate specs by their definition.

Nothing here is shared with the checker's search: plans are interpreted directly from the IR
(no control-flow graph, no memoization, no cone-of-influence projection), every path is
enumerated explicitly, and formulas are evaluated with the reference semantics in
:mod:`plan_model_checker.spec.semantics` (future G, explicit quantification for O/H/S) instead of
the compiled monitor. The only shared pieces are the effect registry lookups and the spec
instantiation, both of which are plain data transformations.

Uses: the differential test (verdicts must agree on thousands of generated plans and specs) and
counterexample replay (every counterexample the checker returns must be a real trace of the plan
that violates the spec).
"""

from __future__ import annotations

from collections.abc import Container, Iterator, Mapping
from dataclasses import dataclass

from plan_model_checker.checker import (
    Counterexample,
    Decision,
    collection_bounds,
    size_assignments,
    symbolic_item,
)
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.ir import ForEach, If, Node, OpaqueGuard, Plan, ToolCall
from plan_model_checker.props import END, START, GroundProp
from plan_model_checker.spec.ast import Formula
from plan_model_checker.spec.semantics import Evaluator
from plan_model_checker.spec.specs import SpecSet

__all__ = [
    "OracleBudgetExceededError",
    "OracleTrace",
    "OracleVerdict",
    "ReplayError",
    "Valuation",
    "enumerate_traces",
    "oracle_check",
    "plan_universe",
    "replay_counterexample",
    "violates",
]


class OracleBudgetExceededError(Exception):
    """The plan has more traces than the oracle is allowed to enumerate."""


class ReplayError(Exception):
    """A counterexample does not correspond to an execution of the plan."""


@dataclass(frozen=True, slots=True, eq=False)
class Valuation:
    """All propositions true at one position: the event plus every true fluent.

    Fluents that were never assigned take their declared initial value, so the set is infinite
    in principle; membership is computed rather than stored.
    """

    event: GroundProp
    assigned: Mapping[GroundProp, bool]
    registry: EffectRegistry

    def __contains__(self, prop: object) -> bool:
        if not isinstance(prop, GroundProp):
            return False
        if prop == self.event:
            return True
        value = self.assigned.get(prop)
        if value is not None:
            return value
        return self.registry.has_prop(prop.name) and self.registry.initial(prop)


@dataclass(frozen=True, slots=True)
class OracleTrace:
    sizes: tuple[tuple[str, int], ...]
    steps: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...]
    decisions: tuple[Decision, ...]
    letters: tuple[Valuation, ...]


@dataclass(frozen=True, slots=True)
class _Config:
    state: dict[GroundProp, bool]
    steps: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...]
    decisions: tuple[Decision, ...]
    letters: tuple[Valuation, ...]


def _freeze(state: Mapping[GroundProp, bool]) -> dict[GroundProp, bool]:
    return dict(state)


class _Interpreter:
    def __init__(
        self,
        registry: EffectRegistry,
        sizes: Mapping[str, int],
        max_traces: int,
    ) -> None:
        self.registry = registry
        self.sizes = sizes
        self.max_traces = max_traces

    def items(self, node: ForEach) -> list[str]:
        if node.items is not None:
            return list(node.items)
        collection = node.collection or ""
        return [symbolic_item(collection, k) for k in range(self.sizes[collection])]

    @staticmethod
    def _value(value: str, env: Mapping[str, str]) -> str:
        return env[value[1:]] if value.startswith("$") else value

    def call_args(self, node: ToolCall, env: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        given = node.arg_map()
        return tuple(
            (param.name, self._value(given[param.name], env))
            for param in self.registry.tool(node.tool).params
        )

    def apply(self, config: _Config, node: ToolCall, env: Mapping[str, str]) -> list[_Config]:
        args = self.call_args(node, env)
        mapping = dict(args)
        event = self.registry.ground_event(node.tool, mapping)
        results: list[_Config] = []
        for outcome in self.registry.ground_outcomes(node.tool, mapping):
            state = dict(config.state)
            for prop, value in outcome.updates:
                state[prop] = value
            letter = Valuation(event=event, assigned=_freeze(state), registry=self.registry)
            results.append(
                _Config(
                    state=state,
                    steps=(*config.steps, (node.tool, args, outcome.name)),
                    decisions=config.decisions,
                    letters=(*config.letters, letter),
                )
            )
        return results

    def guard(self, config: _Config, node: If, env: Mapping[str, str]) -> bool:
        cond = node.cond
        if isinstance(cond, OpaqueGuard):  # pragma: no cover - callers branch on opaque guards
            raise TypeError("opaque guards have no value")
        prop = GroundProp(cond.prop, tuple(self._value(arg, env) for arg in cond.args))
        value = config.state.get(prop, self.registry.initial(prop))
        return value != cond.negate

    def run_block(
        self, nodes: tuple[Node, ...], path: str, env: Mapping[str, str], configs: list[_Config]
    ) -> list[_Config]:
        for index, node in enumerate(nodes):
            configs = self.run_node(node, f"{path}/{index}", env, configs)
            if len(configs) > self.max_traces:
                raise OracleBudgetExceededError(f"more than {self.max_traces} partial traces")
        return configs

    def run_node(
        self, node: Node, path: str, env: Mapping[str, str], configs: list[_Config]
    ) -> list[_Config]:
        if isinstance(node, ToolCall):
            return [result for config in configs for result in self.apply(config, node, env)]
        if isinstance(node, If):
            if isinstance(node.cond, OpaqueGuard):
                then_configs = [
                    _Config(c.state, c.steps, (*c.decisions, Decision(path, "then")), c.letters)
                    for c in configs
                ]
                else_configs = [
                    _Config(
                        c.state, c.steps, (*c.decisions, Decision(path, "otherwise")), c.letters
                    )
                    for c in configs
                ]
                return self.run_block(node.then, f"{path}/then", env, then_configs) + (
                    self.run_block(node.otherwise, f"{path}/otherwise", env, else_configs)
                )
            taken = [c for c in configs if self.guard(c, node, env)]
            skipped = [c for c in configs if not self.guard(c, node, env)]
            return self.run_block(node.then, f"{path}/then", env, taken) + self.run_block(
                node.otherwise, f"{path}/otherwise", env, skipped
            )
        for item in self.items(node):
            configs = self.run_block(node.body, f"{path}/body", {**env, node.var: item}, configs)
        return configs

    def universe(self, nodes: tuple[Node, ...], env: Mapping[str, str]) -> dict[str, set[str]]:
        found: dict[str, set[str]] = {}
        for node in nodes:
            if isinstance(node, ToolCall):
                sorts = self.registry.param_sorts(node.tool)
                for name, value in self.call_args(node, env):
                    found.setdefault(sorts[name], set()).add(value)
            elif isinstance(node, If):
                if not isinstance(node.cond, OpaqueGuard):
                    decl = self.registry.prop(node.cond.prop)
                    for sort, value in zip(decl.params, node.cond.args, strict=True):
                        found.setdefault(sort, set()).add(self._value(value, env))
                for branch in (node.then, node.otherwise):
                    for sort, values in self.universe(branch, env).items():
                        found.setdefault(sort, set()).update(values)
            else:
                for item in self.items(node):
                    for sort, values in self.universe(node.body, {**env, node.var: item}).items():
                        found.setdefault(sort, set()).update(values)
        return found


def _start_letter(registry: EffectRegistry) -> Valuation:
    return Valuation(event=START, assigned={}, registry=registry)


def enumerate_traces(
    plan: Plan,
    registry: EffectRegistry,
    sizes: tuple[tuple[str, int], ...] = (),
    *,
    max_traces: int = 20_000,
) -> list[OracleTrace]:
    """Every complete trace of the plan for one assignment of collection sizes."""
    interpreter = _Interpreter(registry, dict(sizes), max_traces)
    initial = _Config(state={}, steps=(), decisions=(), letters=(_start_letter(registry),))
    finals = interpreter.run_block(plan.steps, "steps", {}, [initial])
    return [
        OracleTrace(
            sizes=sizes,
            steps=config.steps,
            decisions=config.decisions,
            letters=(
                *config.letters,
                Valuation(event=END, assigned=_freeze(config.state), registry=registry),
            ),
        )
        for config in finals
    ]


def plan_universe(
    plan: Plan, registry: EffectRegistry, sizes: tuple[tuple[str, int], ...] = ()
) -> dict[str, tuple[str, ...]]:
    interpreter = _Interpreter(registry, dict(sizes), max_traces=1)
    return {
        sort: tuple(sorted(values)) for sort, values in interpreter.universe(plan.steps, {}).items()
    }


@dataclass(frozen=True, slots=True)
class OracleVerdict:
    violated: dict[str, bool]
    shortest: dict[str, int]
    traces: int


def oracle_check(
    plan: Plan, registry: EffectRegistry, specs: SpecSet, *, max_traces: int = 20_000
) -> OracleVerdict:
    """Per spec: is any trace violating, and how long is the shortest violating trace."""
    violated = dict.fromkeys(specs.names, False)
    shortest: dict[str, int] = {}
    total = 0
    for sizes in size_assignments(collection_bounds(plan)):
        universe = plan_universe(plan, registry, sizes)
        instances = specs.instances(universe)
        traces = enumerate_traces(plan, registry, sizes, max_traces=max_traces)
        total += len(traces)
        for trace in traces:
            evaluator = Evaluator(trace.letters)
            for instance in instances:
                if evaluator.holds(instance.source, 0):
                    continue
                violated[instance.spec] = True
                length = len(trace.steps)
                if instance.spec not in shortest or length < shortest[instance.spec]:
                    shortest[instance.spec] = length
    return OracleVerdict(violated=violated, shortest=shortest, traces=total)


class _Replayer:
    """Walks the plan once, consuming a counterexample's steps and branch decisions in order."""

    def __init__(
        self,
        interpreter: _Interpreter,
        steps: Iterator[tuple[str, tuple[tuple[str, str], ...], str]],
        decisions: Iterator[Decision],
    ) -> None:
        self.interpreter = interpreter
        self.steps = steps
        self.decisions = decisions

    def call(self, node: ToolCall, here: str, env: Mapping[str, str], config: _Config) -> _Config:
        expected = next(self.steps, None)
        args = self.interpreter.call_args(node, env)
        if expected is None or expected[0] != node.tool or expected[1] != args:
            raise ReplayError(f"{here}: plan calls {node.tool}{dict(args)} here")
        outcomes = {cfg.steps[-1][2]: cfg for cfg in self.interpreter.apply(config, node, env)}
        if expected[2] not in outcomes:
            raise ReplayError(f"{here}: {node.tool} has no outcome {expected[2]}")
        return outcomes[expected[2]]

    def block(
        self, nodes: tuple[Node, ...], path: str, env: Mapping[str, str], config: _Config
    ) -> _Config:
        for index, node in enumerate(nodes):
            here = f"{path}/{index}"
            if isinstance(node, ToolCall):
                config = self.call(node, here, env, config)
            elif isinstance(node, If):
                if isinstance(node.cond, OpaqueGuard):
                    decision = next(self.decisions, None)
                    if decision is None or decision.site != here:
                        raise ReplayError(f"{here}: missing branch decision")
                    branch = decision.choice
                else:
                    branch = "then" if self.interpreter.guard(config, node, env) else "otherwise"
                body = node.then if branch == "then" else node.otherwise
                config = self.block(body, f"{here}/{branch}", env, config)
            else:
                for item in self.interpreter.items(node):
                    config = self.block(node.body, f"{here}/body", {**env, node.var: item}, config)
        return config


def replay_counterexample(
    plan: Plan, registry: EffectRegistry, counterexample: Counterexample
) -> OracleTrace:
    """Re-execute the plan following the counterexample's decisions; return the full trace."""
    interpreter = _Interpreter(registry, dict(counterexample.sizes), max_traces=1)
    expected_steps = [
        (step.call.tool, step.call.args, step.outcome) for step in counterexample.steps
    ]
    steps = iter(expected_steps)
    decisions = iter(counterexample.decisions)
    initial = _Config(state={}, steps=(), decisions=(), letters=(_start_letter(registry),))
    final = _Replayer(interpreter, steps, decisions).block(plan.steps, "steps", {}, initial)
    if next(steps, None) is not None or next(decisions, None) is not None:
        raise ReplayError("counterexample has more steps or decisions than the plan executes")
    return OracleTrace(
        sizes=counterexample.sizes,
        steps=final.steps,
        decisions=counterexample.decisions,
        letters=(
            *final.letters,
            Valuation(event=END, assigned=_freeze(final.state), registry=registry),
        ),
    )


def violates(trace: OracleTrace, formula: Formula) -> bool:
    """True when the trace falsifies the ground, G-form formula (reference semantics)."""
    letters: tuple[Container[GroundProp], ...] = trace.letters
    return not Evaluator(letters).holds(formula, 0)
