"""Reference semantics of the spec language, written straight from the textbook definitions.

This evaluator is deliberately naive and independent of :mod:`~plan_model_checker.spec.monitor`:
G is interpreted as a *future* operator over the remaining trace, O/H/S quantify over explicit
position ranges, and nothing is incremental. The brute-force oracle uses it to validate the
checker, and the repair explainer uses it to find which propositions a violation hinges on.
"""

from __future__ import annotations

from collections.abc import Container, Sequence

from plan_model_checker.props import GroundProp
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
    Since,
    WeakYesterday,
    Yesterday,
    ground_prop,
)

__all__ = ["Evaluator", "satisfies"]

Letter = Container[GroundProp]


class Evaluator:
    """Memoized ``(w, i) ⊨ φ`` for one finite trace ``w``."""

    def __init__(self, trace: Sequence[Letter]) -> None:
        if not trace:
            raise ValueError("traces have at least one position")
        self._trace = tuple(trace)
        self._memo: dict[tuple[Formula, int], bool] = {}

    @property
    def length(self) -> int:
        return len(self._trace)

    def holds(self, formula: Formula, i: int) -> bool:  # noqa: PLR0912
        key = (formula, i)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        result: bool
        if isinstance(formula, Atom):
            result = ground_prop(formula) in self._trace[i]
        elif isinstance(formula, Bool):
            result = formula.value
        elif isinstance(formula, Not):
            result = not self.holds(formula.arg, i)
        elif isinstance(formula, And):
            result = self.holds(formula.left, i) and self.holds(formula.right, i)
        elif isinstance(formula, Or):
            result = self.holds(formula.left, i) or self.holds(formula.right, i)
        elif isinstance(formula, Implies):
            result = (not self.holds(formula.left, i)) or self.holds(formula.right, i)
        elif isinstance(formula, Iff):
            result = self.holds(formula.left, i) == self.holds(formula.right, i)
        elif isinstance(formula, Yesterday):
            result = i > 0 and self.holds(formula.arg, i - 1)
        elif isinstance(formula, WeakYesterday):
            result = i == 0 or self.holds(formula.arg, i - 1)
        elif isinstance(formula, Once):
            result = any(self.holds(formula.arg, j) for j in range(i + 1))
        elif isinstance(formula, Historically):
            result = all(self.holds(formula.arg, j) for j in range(i + 1))
        elif isinstance(formula, Since):
            result = any(
                self.holds(formula.right, j)
                and all(self.holds(formula.left, k) for k in range(j + 1, i + 1))
                for j in range(i + 1)
            )
        elif isinstance(formula, Always):
            result = all(self.holds(formula.arg, j) for j in range(i, len(self._trace)))
        else:  # pragma: no cover - exhaustive over Formula
            raise TypeError(f"unknown formula node {type(formula).__name__}")
        self._memo[key] = result
        return result


def satisfies(formula: Formula, trace: Sequence[Letter]) -> bool:
    """``w ⊨ φ``: the formula evaluated at position 0 (standard LTLf semantics)."""
    return Evaluator(trace).holds(formula, 0)
