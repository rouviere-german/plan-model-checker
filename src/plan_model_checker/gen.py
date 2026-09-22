"""Random plans and specs for differential testing against the brute-force oracle.

The generators draw from a :class:`random.Random` supplied by the caller, so the same code serves
the seeded benchmark (5,000 cases) and the hypothesis property tests (``st.randoms()``, which lets
hypothesis control and shrink every draw).

The synthetic registry is small but exercises every feature the checker handles: nondeterministic
outcomes (``touch`` may fail, ``flip`` has two outcomes), a fluent that starts true, 0-arity and
unary propositions, a two-parameter tool, and effects that set and clear the same fluent family.
Formulas use every operator (Y Z O H S, all boolean connectives) under a top-level G, and
sometimes combine two G-terms with & or |.
"""

from __future__ import annotations

import random
from typing import Any

from plan_model_checker.effects import (
    EffectDecl,
    EffectRegistry,
    EffectRegistryModel,
    EventDecl,
    OutcomeDecl,
    ParamDecl,
    PropDecl,
    ToolEffects,
)
from plan_model_checker.ir import Plan, unrolled_call_count
from plan_model_checker.spec.ast import (
    Always,
    And,
    Atom,
    Bool,
    Const,
    Formula,
    Historically,
    Iff,
    Implies,
    Not,
    Once,
    Or,
    Since,
    Term,
    Var,
    WeakYesterday,
    Yesterday,
)
from plan_model_checker.spec.specs import SpecSet, build_spec

__all__ = ["CONSTANTS", "diff_registry", "random_formula", "random_plan", "random_specs"]

CONSTANTS = ("r1", "r2")
_UNARY_FLUENTS = ("p", "q")
_UNARY_EVENTS = ("a", "b")
_NULLARY = ("s", "c", "start", "end")


def diff_registry() -> EffectRegistry:
    def param(name: str) -> ParamDecl:
        return ParamDecl(name=name, sort="r")

    def effect(prop: str, value: bool, *args: str) -> EffectDecl:
        return EffectDecl(prop=prop, args=args, value=value)

    model = EffectRegistryModel(
        name="differential",
        sorts=("r",),
        props=(
            PropDecl(name="p", kind="fluent", params=("r",)),
            PropDecl(name="q", kind="fluent", params=("r",)),
            PropDecl(name="s", kind="fluent", initial=True),
            PropDecl(name="a", kind="event", params=("r",)),
            PropDecl(name="b", kind="event", params=("r",)),
            PropDecl(name="c", kind="event"),
        ),
        tools=(
            ToolEffects(
                name="touch",
                params=(param("x"),),
                event=EventDecl(prop="a", args=("x",)),
                outcomes=(
                    OutcomeDecl(name="ok", effects=(effect("p", True, "x"),)),
                    OutcomeDecl(name="failed", effects=()),
                ),
            ),
            ToolEffects(
                name="swap",
                params=(param("x"),),
                event=EventDecl(prop="b", args=("x",)),
                outcomes=(
                    OutcomeDecl(
                        name="ok", effects=(effect("p", False, "x"), effect("q", True, "x"))
                    ),
                ),
            ),
            ToolEffects(
                name="flip",
                params=(),
                event=EventDecl(prop="c"),
                outcomes=(
                    OutcomeDecl(name="off", effects=(effect("s", False),)),
                    OutcomeDecl(name="on", effects=(effect("s", True),)),
                ),
            ),
            ToolEffects(
                name="link",
                params=(param("x"), param("y")),
                event=EventDecl(prop="b", args=("y",)),
                outcomes=(OutcomeDecl(name="ok", effects=(effect("q", True, "x"),)),),
            ),
            ToolEffects(
                name="clear",
                params=(param("x"),),
                event=EventDecl(prop="a", args=("x",)),
                outcomes=(
                    OutcomeDecl(
                        name="ok", effects=(effect("q", False, "x"), effect("p", False, "x"))
                    ),
                ),
            ),
        ),
    )
    return EffectRegistry(model)


def _resource(rng: random.Random, variables: list[str]) -> str:
    options = [*CONSTANTS, *(f"${var}" for var in variables)]
    return rng.choice(options)


def _call(rng: random.Random, variables: list[str]) -> dict[str, Any]:
    tool = rng.choice(("touch", "swap", "flip", "link", "clear"))
    params = {"touch": ["x"], "swap": ["x"], "flip": [], "link": ["x", "y"], "clear": ["x"]}[tool]
    return {
        "kind": "call",
        "tool": tool,
        "args": [{"name": name, "value": _resource(rng, variables)} for name in params],
    }


def _block(
    rng: random.Random, depth: int, budget: list[int], variables: list[str], max_depth: int
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for _ in range(rng.randint(0 if depth > 1 else 1, 3)):
        if budget[0] <= 0:
            break
        roll = rng.random()
        if depth < max_depth and roll < 0.2:
            cond: dict[str, Any]
            if rng.random() < 0.5:
                cond = {"kind": "opaque", "label": rng.choice(("hot", "cold"))}
            else:
                prop = rng.choice(("p", "q", "s"))
                args = [] if prop == "s" else [_resource(rng, variables)]
                cond = {"kind": "prop", "prop": prop, "args": args, "negate": rng.random() < 0.5}
            steps.append(
                {
                    "kind": "if",
                    "cond": cond,
                    "then": _block(rng, depth + 1, budget, variables, max_depth),
                    "otherwise": _block(rng, depth + 1, budget, variables, max_depth),
                }
            )
        elif depth < max_depth and roll < 0.32 and len(variables) < 2:
            var = f"v{len(variables)}"
            loop: dict[str, Any] = {"kind": "foreach", "var": var}
            if rng.random() < 0.5:
                items = rng.sample(CONSTANTS, rng.randint(1, 2))
                loop.update(items=items, max_iterations=len(items))
            else:
                loop.update(collection="items", max_iterations=2)
            loop["body"] = _block(rng, depth + 1, budget, [*variables, var], max_depth)
            steps.append(loop)
        else:
            budget[0] -= 1
            steps.append(_call(rng, variables))
    return steps


def random_plan(
    rng: random.Random, *, max_depth: int = 6, max_calls: int = 6, max_unrolled: int = 8
) -> Plan:
    """A well-typed plan over the synthetic registry (nesting depth at most ``max_depth``).

    Plans whose unrolled size exceeds ``max_unrolled`` call sites are redrawn, which keeps the
    brute-force oracle's trace enumeration small.
    """
    while True:
        budget = [max_calls]
        steps = _block(rng, 1, budget, [], max_depth)
        plan = Plan.model_validate({"goal": "generated", "steps": steps})
        if unrolled_call_count(plan.steps) <= max_unrolled:
            return plan


def _term(rng: random.Random, variables: tuple[str, ...]) -> Term:
    if variables and rng.random() < 0.7:
        return Var(rng.choice(variables))
    return Const(rng.choice(CONSTANTS))


def _atom(rng: random.Random, variables: tuple[str, ...]) -> Formula:
    roll = rng.random()
    if roll < 0.05:
        return Bool(rng.random() < 0.5)
    if roll < 0.3:
        return Atom(rng.choice(_NULLARY))
    name = rng.choice(_UNARY_FLUENTS + _UNARY_EVENTS)
    return Atom(name, (_term(rng, variables),))


def random_formula(
    rng: random.Random, *, depth: int = 4, variables: tuple[str, ...] = ("x", "y")
) -> Formula:
    """A pure past-time formula over the synthetic registry's propositions."""
    if depth <= 0 or rng.random() < 0.25:
        return _atom(rng, variables)
    kind = rng.randrange(11)
    sub = depth - 1
    if kind == 0:
        return Not(random_formula(rng, depth=sub, variables=variables))
    if kind in (1, 2, 3, 4):
        left = random_formula(rng, depth=sub, variables=variables)
        right = random_formula(rng, depth=sub, variables=variables)
        return (And, Or, Implies, Iff)[kind - 1](left, right)
    if kind in (5, 6, 7, 8):
        arg = random_formula(rng, depth=sub, variables=variables)
        return (Yesterday, WeakYesterday, Once, Historically)[kind - 5](arg)
    return Since(
        random_formula(rng, depth=sub, variables=variables),
        random_formula(rng, depth=sub, variables=variables),
    )


def _term_formula(rng: random.Random, variables: tuple[str, ...]) -> Formula:
    body = random_formula(rng, variables=variables)
    if rng.random() < 0.5:
        # Resource-protocol shape: "whenever this event happens, the past must look like ...".
        trigger = rng.choice(("a", "b", "c", "end"))
        event = (
            Atom(trigger, (_term(rng, variables),)) if trigger in _UNARY_EVENTS else Atom(trigger)
        )
        return Implies(event, body)
    return body


def random_spec(rng: random.Random) -> Formula:
    """``G ψ`` with ψ pure past-time, sometimes combined with a second G-term by & or |."""
    variables = ("x",) if rng.random() < 0.8 else ("x", "y")
    first = Always(_term_formula(rng, variables))
    roll = rng.random()
    if roll < 0.15:
        return And(first, Always(_term_formula(rng, variables)))
    if roll < 0.25:
        return Or(first, Always(_term_formula(rng, variables)))
    return first


def random_specs(rng: random.Random, registry: EffectRegistry, count: int = 2) -> SpecSet:
    return SpecSet(
        tuple(build_spec(f"spec_{index}", random_spec(rng), registry) for index in range(count))
    )
