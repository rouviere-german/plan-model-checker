"""Constant folding of ground past-time formulas, for skipping instances that cannot fail.

On a given plan many spec instances are vacuous: ``G(delete[r] -> Y backup_ok[r])`` for a
directory the plan never deletes, or ``G(end -> !lock_held[r])`` for a path whose lock no call
touches. The checker would still run a full search for each of them. Here the caller says which
propositions are *fixed* on every trace of the plan (an event no call emits is always false; a
fluent no call can change keeps its initial value), and :func:`simplify` folds them in with rules
that are exact at every position of every trace:

====================  ===================================================
``O c``, ``H c``      ``c`` (both include the current position)
``Y false``           ``false``     (``Y true`` is false at position 0: kept)
``Z true``            ``true``      (``Z false`` is true at position 0: kept)
``p S false``         ``false``     ``p S true`` is ``true``
``false S q``         ``q``         ``true S q`` is ``O q``
====================  ===================================================

plus the usual boolean identities. If the final form of an instance folds to ``true``, it holds
on every trace and needs no search. The differential test against the brute-force oracle covers
this shortcut like the rest of the checker.
"""

from __future__ import annotations

from collections.abc import Callable

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

__all__ = ["TRUE", "simplify"]

TRUE = Bool(True)
FALSE = Bool(False)

Fixed = Callable[[GroundProp], bool | None]


def _const(formula: Formula) -> bool | None:
    return formula.value if isinstance(formula, Bool) else None


def _and(left: Formula, right: Formula) -> Formula:
    a, b = _const(left), _const(right)
    if a is False or b is False:
        return FALSE
    if a is True:
        return right
    return left if b is True else And(left, right)


def _or(left: Formula, right: Formula) -> Formula:
    a, b = _const(left), _const(right)
    if a is True or b is True:
        return TRUE
    if a is False:
        return right
    return left if b is False else Or(left, right)


def _implies(left: Formula, right: Formula) -> Formula:
    a, b = _const(left), _const(right)
    if a is False or b is True:
        return TRUE
    if a is True:
        return right
    return Not(left) if b is False else Implies(left, right)


def _iff(left: Formula, right: Formula) -> Formula:
    a, b = _const(left), _const(right)
    return Bool(a == b) if a is not None and b is not None else Iff(left, right)


_BOOLEAN = {And: _and, Or: _or, Implies: _implies, Iff: _iff}


def _since(left: Formula, right: Formula) -> Formula:
    a, b = _const(left), _const(right)
    if b is not None:
        return Bool(b)
    if a is False:
        return right
    if a is True:
        return Once(right)
    return Since(left, right)


def _unary(
    formula: Not | Yesterday | WeakYesterday | Once | Historically | Always, inner: Formula
) -> Formula:
    value = _const(inner)
    if isinstance(formula, Not):
        return Not(inner) if value is None else Bool(not value)
    if isinstance(formula, Yesterday):
        return FALSE if value is False else Yesterday(inner)
    if isinstance(formula, WeakYesterday):
        return TRUE if value is True else WeakYesterday(inner)
    return inner if value is not None else type(formula)(inner)


def simplify(formula: Formula, fixed: Fixed) -> Formula:
    """``formula`` with fixed propositions replaced by constants and folded (exact)."""
    if isinstance(formula, Atom):
        value = fixed(ground_prop(formula))
        return formula if value is None else Bool(value)
    if isinstance(formula, Bool):
        return formula
    if isinstance(formula, And | Or | Implies | Iff):
        fold = _BOOLEAN[type(formula)]
        return fold(simplify(formula.left, fixed), simplify(formula.right, fixed))
    if isinstance(formula, Since):
        return _since(simplify(formula.left, fixed), simplify(formula.right, fixed))
    return _unary(formula, simplify(formula.arg, fixed))
