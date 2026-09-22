"""Turn a counterexample into a repair hint: which step is missing, and where it goes.

For the violated term ``G ψ`` of the spec, the explainer takes the first position ``k`` at which
``ψ`` is false and asks, recursively over the formula, which literal changes would make ``ψ``
true there (``Y p`` needs ``p`` at ``k-1``, ``O p`` needs ``p`` at some earlier position, ``a S b``
needs ``b`` now or ``a`` now and the since-formula one step earlier, and so on). Only fixes that
an *insertion* can achieve are kept:

- a literal on the state immediately before the violating step: insert a step that establishes
  it right before that step;
- a fluent literal on the final position (``G(end -> ...)`` rules): insert a step at the end of
  the block that opened the obligation (after the last node of the block containing the step that
  last set the fluent the other way, so a lock acquired in a loop body is released in that body).

The registry then names tools whose event or declared effects establish the literal, with their
arguments mapped back to the plan's own terms (loop variables included). The hint is advice: the
repaired plan is always re-checked, and nothing is inserted without passing the checker.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal

from plan_model_checker.checker import Counterexample
from plan_model_checker.effects import EffectDecl, EffectRegistry, ToolEffects
from plan_model_checker.ir import ForEach, If, Node, Plan, PlanError, ToolCall, node_at
from plan_model_checker.props import RESERVED_PROPS, GroundProp
from plan_model_checker.spec.ast import (
    Always,
    And,
    Atom,
    Bool,
    Formula,
    Historically,
    Iff,
    Implies,
    Not,
    Once,
    Or,
    WeakYesterday,
    Yesterday,
    ground_prop,
)
from plan_model_checker.spec.semantics import Evaluator

__all__ = ["RepairHint", "Requirement", "Suggestion", "explain"]

_MAX_ALTERNATIVES = 32
_MAX_LITERALS = 3

Lit = tuple[GroundProp, int, bool]
Alt = frozenset[Lit]


@dataclass(frozen=True, slots=True)
class Requirement:
    prop: GroundProp
    value: bool

    def __str__(self) -> str:
        return str(self.prop) if self.value else f"!{self.prop}"


@dataclass(frozen=True, slots=True)
class Suggestion:
    tool: str
    args: tuple[tuple[str, str], ...]

    def to_ir(self) -> ToolCall:
        return ToolCall.model_validate(
            {"tool": self.tool, "args": [{"name": n, "value": v} for n, v in self.args]}
        )

    def label(self) -> str:
        return f"{self.tool}({', '.join(f'{n}={v}' for n, v in self.args)})"


@dataclass(frozen=True, slots=True)
class RepairHint:
    spec: str
    instance: str
    requirements: tuple[Requirement, ...]
    placement: Literal["before", "after", "append"]
    anchor_path: str | None
    suggestions: tuple[Suggestion, ...]

    def describe(self) -> str:
        where = {
            "before": f"immediately before the call at {self.anchor_path}",
            "after": f"immediately after the step at {self.anchor_path}",
            "append": "at the end of the plan",
        }[self.placement]
        needed = " and ".join(str(requirement) for requirement in self.requirements)
        tools = ", ".join(s.label() for s in self.suggestions)
        return f"{needed} must hold {where}; candidate steps: {tools}"

    def to_json(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "instance": self.instance,
            "requires": [
                {**requirement.prop.to_json(), "value": requirement.value}
                for requirement in self.requirements
            ],
            "placement": self.placement,
            "anchor_path": self.anchor_path,
            "suggestions": [
                {"tool": s.tool, "args": [{"name": n, "value": v} for n, v in s.args]}
                for s in self.suggestions
            ],
        }


def _combine(left: list[Alt], right: list[Alt]) -> list[Alt]:
    merged = {a | b for a, b in itertools.product(left, right)}
    return sorted(merged, key=_alt_key)[:_MAX_ALTERNATIVES]


def _union(*groups: list[Alt]) -> list[Alt]:
    merged = {alt for group in groups for alt in group}
    return sorted(merged, key=_alt_key)[:_MAX_ALTERNATIVES]


def _alt_key(alt: Alt) -> tuple[int, list[tuple[str, int, bool]]]:
    return (len(alt), sorted((str(p), -pos, v) for p, pos, v in alt))


class _Fixer:
    def __init__(self, evaluator: Evaluator) -> None:
        self.ev = evaluator

    def fixes(self, formula: Formula, i: int, want: bool) -> list[Alt]:  # noqa: PLR0911, PLR0912
        if self.ev.holds(formula, i) == want:
            return [frozenset()]
        if isinstance(formula, Atom):
            return [frozenset({(ground_prop(formula), i, want)})]
        if isinstance(formula, Bool | Always):
            return []
        if isinstance(formula, Not):
            return self.fixes(formula.arg, i, not want)
        if isinstance(formula, And | Or):
            conjunctive = isinstance(formula, And) == want
            left = self.fixes(formula.left, i, want)
            right = self.fixes(formula.right, i, want)
            return _combine(left, right) if conjunctive else _union(left, right)
        if isinstance(formula, Implies):
            return self.fixes(Or(Not(formula.left), formula.right), i, want)
        if isinstance(formula, Iff):
            same = Or(And(formula.left, formula.right), And(Not(formula.left), Not(formula.right)))
            return self.fixes(same, i, want)
        if isinstance(formula, Yesterday | WeakYesterday):
            if i == 0:
                return []
            return self.fixes(formula.arg, i - 1, want)
        if isinstance(formula, Once | Historically):
            existential = isinstance(formula, Once) == want
            groups = [self.fixes(formula.arg, j, want) for j in range(i, -1, -1)]
            if existential:
                return _union(*groups)
            combined: list[Alt] = [frozenset()]
            for group in groups:
                combined = _combine(combined, group)
            return combined
        # Since: only the "make it true" direction is useful for insertion repairs.
        if not want:
            return []
        now = self.fixes(formula.right, i, True)
        if i == 0:
            return now
        keep = _combine(self.fixes(formula.left, i, True), self.fixes(formula, i - 1, True))
        return _union(now, keep)


def _violated_term(source: Formula, ev: Evaluator, position: int) -> Formula | None:
    stack = [source]
    while stack:
        node = stack.pop()
        if isinstance(node, Always) and not ev.holds(Historically(node.arg), position):
            return node.arg
        if isinstance(node, And | Or):
            stack.extend((node.right, node.left))
    return None


def _first_false(term: Formula, ev: Evaluator, upto: int) -> int:
    for position in range(upto + 1):
        if not ev.holds(term, position):
            return position
    return upto


def _ir_terms(plan: Plan, counterexample: Counterexample, anchor: int | None) -> dict[str, str]:
    """Ground resource -> plan term, from the anchor call's own arguments."""
    if anchor is None:
        return {}
    call = counterexample.steps[anchor].call
    try:
        node = node_at(plan, call.path)
    except PlanError:
        return {}
    if not isinstance(node, ToolCall):
        return {}
    given = node.arg_map()
    return {value: given[name] for name, value in call.args if name in given}


def _bind(
    registry: EffectRegistry, tool: ToolEffects, requirements: tuple[Requirement, ...]
) -> dict[str, str] | None:
    """Parameter values under which one call of ``tool`` establishes every requirement."""
    params = {param.name for param in tool.params}
    for outcome in tool.outcomes:
        mapping = _bind_outcome(registry, tool, outcome.effects, requirements, {})
        if mapping is not None and set(mapping) == params:
            return mapping
    return None


def _bind_outcome(
    registry: EffectRegistry,
    tool: ToolEffects,
    effects: tuple[EffectDecl, ...],
    requirements: tuple[Requirement, ...],
    mapping: dict[str, str],
) -> dict[str, str] | None:
    if not requirements:
        return mapping
    first, rest = requirements[0], requirements[1:]
    decl = registry.prop(first.prop.name)
    options: list[tuple[str, ...]] = []
    if decl.kind == "event":
        if not first.value:
            # Satisfied unless the inserted call itself emits exactly this event.
            result = _bind_outcome(registry, tool, effects, rest, mapping)
            if result is not None and tool.event.prop == first.prop.name:
                emitted = tuple(result.get(param, "") for param in tool.event.args)
                if emitted == first.prop.args:
                    return None
            return result
        if tool.event.prop == first.prop.name:
            options.append(tool.event.args)
    else:
        options.extend(
            effect.args
            for effect in effects
            if effect.prop == first.prop.name and effect.value == first.value
        )
    for params in options:
        candidate = dict(mapping)
        if all(
            candidate.setdefault(param, value) == value
            for param, value in zip(params, first.prop.args, strict=True)
        ):
            result = _bind_outcome(registry, tool, effects, rest, candidate)
            if result is not None:
                return result
    return None


def _always(
    registry: EffectRegistry, tool_name: str, requirements: tuple[Requirement, ...]
) -> bool:
    tool = registry.tool(tool_name)
    return all(
        _bind_outcome(registry, tool, outcome.effects, requirements, {}) is not None
        for outcome in tool.outcomes
    )


def _suggest(
    registry: EffectRegistry, requirements: tuple[Requirement, ...], terms: dict[str, str]
) -> tuple[Suggestion, ...]:
    suggestions: list[Suggestion] = []
    for name in registry.tool_names:
        tool = registry.tool(name)
        mapping = _bind(registry, tool, requirements)
        if mapping is None:
            continue
        args = [
            (param.name, terms.get(mapping[param.name], mapping[param.name]))
            for param in tool.params
        ]
        if any("#" in value for _, value in args):
            continue  # a symbolic loop item with no name in the plan at this point
        suggestions.append(Suggestion(tool=name, args=tuple(args)))
    suggestions.sort(key=lambda s: (not _always(registry, s.tool, requirements), s.tool))
    return tuple(suggestions)


def _placement_class(registry: EffectRegistry, alternative: Alt, k: int, last: int) -> int | None:
    """0: every literal just before step k; 1: fluents at the final position; 2: fluents at k."""
    positions = {position for _, position, _ in alternative}
    props = [prop for prop, _, _ in alternative]
    if any(prop.name in RESERVED_PROPS or not registry.has_prop(prop.name) for prop in props):
        return None
    fluents = all(registry.prop(prop.name).kind == "fluent" for prop in props)
    if positions == {k - 1} and 1 <= k <= last:
        return 0
    if positions == {k} and k == last and fluents:
        return 1
    if positions == {k} and 1 <= k < last and fluents:
        return 2
    return None


def explain(
    plan: Plan, registry: EffectRegistry, counterexample: Counterexample
) -> RepairHint | None:
    """A single-insertion repair hint for the counterexample, or None if none applies."""
    ev = Evaluator(counterexample.letters)
    last = len(counterexample.letters) - 1
    term = _violated_term(counterexample.instance.source, ev, last)
    if term is None:  # pragma: no cover - counterexamples always falsify a G-term
        return None
    k = _first_false(term, ev, last)
    candidates: list[tuple[int, int, Alt]] = []
    for alternative in _Fixer(ev).fixes(term, k, True):
        if not alternative or len(alternative) > _MAX_LITERALS:
            continue
        placement_class = _placement_class(registry, alternative, k, last)
        if placement_class is not None:
            candidates.append((placement_class, len(alternative), alternative))
    candidates.sort(key=lambda item: (item[0], item[1], _alt_key(item[2])))
    for _, _, alternative in candidates:
        requirements = tuple(
            Requirement(prop=prop, value=value)
            for prop, _, value in sorted(alternative, key=lambda lit: (str(lit[0]), lit[2]))
        )
        placement: Literal["before", "after", "append"]
        if k == last:
            # An end-of-plan obligation: close it at the end of the block that opened it (the
            # step that last set the fluent the other way), e.g. release a lock at the end of
            # the loop body that acquired it.
            anchor = _opening_step(registry, counterexample, requirements)
            placement = "after" if anchor is not None else "append"
            anchor_path = (
                _block_end(plan, counterexample.steps[anchor].call.path)
                if (anchor is not None)
                else None
            )
        else:
            anchor = k - 1
            placement = "before"
            anchor_path = counterexample.steps[anchor].call.path
        suggestions = _suggest(registry, requirements, _ir_terms(plan, counterexample, anchor))
        if not suggestions:
            continue
        return RepairHint(
            spec=counterexample.spec,
            instance=counterexample.instance.key,
            requirements=requirements,
            placement=placement,
            anchor_path=anchor_path,
            suggestions=suggestions,
        )
    return None


def _opening_step(
    registry: EffectRegistry, counterexample: Counterexample, requirements: tuple[Requirement, ...]
) -> int | None:
    """Index of the last step whose outcome set a required fluent to the opposite value.

    Falls back to the last step of the counterexample (None for an empty trace).
    """
    wanted = {(r.prop, not r.value) for r in requirements}
    for index in range(len(counterexample.steps) - 1, -1, -1):
        step = counterexample.steps[index]
        for outcome in registry.ground_outcomes(step.call.tool, step.call.arg_map()):
            if outcome.name == step.outcome and wanted & set(outcome.updates):
                return index
    return len(counterexample.steps) - 1 if counterexample.steps else None


def _block_end(plan: Plan, path: str) -> str:
    """Path of the last node in the block that contains the node at ``path``."""
    parts = path.split("/")
    parent = parts[:-1]
    if len(parent) == 1:
        size = len(plan.steps)
    else:
        container = node_at(plan, "/".join(parent[:-1]))
        size = len(dict(_blocks(container))[parent[-1]])
    return "/".join([*parent, str(size - 1)])


def _blocks(node: ToolCall | If | ForEach) -> list[tuple[str, tuple[Node, ...]]]:
    if isinstance(node, If):
        return [("then", node.then), ("otherwise", node.otherwise)]
    if isinstance(node, ForEach):
        return [("body", node.body)]
    return []
