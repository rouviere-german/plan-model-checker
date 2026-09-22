"""Abstract syntax of the past-time LTL spec language.

Operators (Y, Z, O, H, S are past-time; G is the only future operator and may appear only at the
top level, outside every temporal operator and under positive polarity):

====  ===============  ======================================================
Y φ   yesterday        φ held at the previous position (false at position 0)
Z φ   weak yesterday   φ held at the previous position (true at position 0)
O φ   once             φ held at some position up to now
H φ   historically     φ held at every position up to now
φSψ   since            ψ held at some position, and φ at every position after it
G φ   always           φ holds at every position of the trace
====  ===============  ======================================================

References: Manna & Pnueli, "Temporal Verification of Reactive Systems: Safety" (1995) for the
past operators; Havelund & Roşu, "Synthesizing Monitors for Safety Properties" (TACAS 2002) for
the incremental evaluation used by :mod:`plan_model_checker.spec.monitor`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from plan_model_checker.props import GroundProp

__all__ = [
    "Always",
    "And",
    "Atom",
    "Bool",
    "Const",
    "Formula",
    "Historically",
    "Iff",
    "Implies",
    "Not",
    "Once",
    "Or",
    "Since",
    "Term",
    "Var",
    "WeakYesterday",
    "Yesterday",
    "atoms",
    "children",
    "depth",
    "ground_prop",
    "render",
    "size",
    "subformulas_postorder",
    "substitute",
    "variables",
]


@dataclass(frozen=True, slots=True)
class Var:
    name: str


@dataclass(frozen=True, slots=True)
class Const:
    value: str


Term = Var | Const


@dataclass(frozen=True, slots=True)
class Atom:
    name: str
    args: tuple[Term, ...] = ()


@dataclass(frozen=True, slots=True)
class Bool:
    value: bool


@dataclass(frozen=True, slots=True)
class Not:
    arg: Formula


@dataclass(frozen=True, slots=True)
class And:
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Or:
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Implies:
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Iff:
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Yesterday:
    arg: Formula


@dataclass(frozen=True, slots=True)
class WeakYesterday:
    arg: Formula


@dataclass(frozen=True, slots=True)
class Once:
    arg: Formula


@dataclass(frozen=True, slots=True)
class Historically:
    arg: Formula


@dataclass(frozen=True, slots=True)
class Since:
    left: Formula
    right: Formula


@dataclass(frozen=True, slots=True)
class Always:
    arg: Formula


Formula = (
    Atom
    | Bool
    | Not
    | And
    | Or
    | Implies
    | Iff
    | Yesterday
    | WeakYesterday
    | Once
    | Historically
    | Since
    | Always
)

_UNARY_SYMBOL: dict[type, str] = {
    Not: "!",
    Yesterday: "Y",
    WeakYesterday: "Z",
    Once: "O",
    Historically: "H",
    Always: "G",
}
_BINARY_SYMBOL: dict[type, str] = {And: "&", Or: "|", Implies: "->", Iff: "<->", Since: "S"}


def children(formula: Formula) -> tuple[Formula, ...]:
    if isinstance(formula, Atom | Bool):
        return ()
    if isinstance(formula, And | Or | Implies | Iff | Since):
        return (formula.left, formula.right)
    return (formula.arg,)


def subformulas_postorder(formula: Formula) -> list[Formula]:
    """Distinct subformulas, children before parents (iterative; shared subterms kept once)."""
    order: list[Formula] = []
    seen: set[Formula] = set()
    stack: list[tuple[Formula, bool]] = [(formula, False)]
    while stack:
        node, expanded = stack.pop()
        if node in seen:
            continue
        if expanded:
            seen.add(node)
            order.append(node)
            continue
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(children(node)))
    return order


def size(formula: Formula) -> int:
    count = 0
    stack = [formula]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(children(node))
    return count


def depth(formula: Formula) -> int:
    best = 0
    stack = [(formula, 1)]
    while stack:
        node, level = stack.pop()
        best = max(best, level)
        stack.extend((child, level + 1) for child in children(node))
    return best


def atoms(formula: Formula) -> Iterator[Atom]:
    stack = [formula]
    while stack:
        node = stack.pop()
        if isinstance(node, Atom):
            yield node
        stack.extend(children(node))


def variables(formula: Formula) -> set[str]:
    return {term.name for atom in atoms(formula) for term in atom.args if isinstance(term, Var)}


def substitute(formula: Formula, binding: Mapping[str, str]) -> Formula:
    """Replace variables by resource constants (unbound variables are left in place)."""
    if isinstance(formula, Atom):
        return Atom(
            formula.name,
            tuple(
                Const(binding[term.name])
                if isinstance(term, Var) and term.name in binding
                else term
                for term in formula.args
            ),
        )
    if isinstance(formula, Bool):
        return formula
    if isinstance(formula, And | Or | Implies | Iff | Since):
        return type(formula)(substitute(formula.left, binding), substitute(formula.right, binding))
    return type(formula)(substitute(formula.arg, binding))


def ground_prop(atom: Atom) -> GroundProp:
    values: list[str] = []
    for term in atom.args:
        if isinstance(term, Var):
            raise TypeError(f"atom {atom.name} still has variable {term.name}")
        values.append(term.value)
    return GroundProp(atom.name, tuple(values))


def _render_term(term: Term) -> str:
    if isinstance(term, Var):
        return term.name
    escaped = term.value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render(formula: Formula) -> str:
    """Concrete syntax that parses back to an equal formula (binary operators parenthesized)."""
    if isinstance(formula, Atom):
        if not formula.args:
            return formula.name
        return f"{formula.name}[{', '.join(_render_term(term) for term in formula.args)}]"
    if isinstance(formula, Bool):
        return "true" if formula.value else "false"
    if isinstance(formula, And | Or | Implies | Iff | Since):
        symbol = _BINARY_SYMBOL[type(formula)]
        return f"({render(formula.left)} {symbol} {render(formula.right)})"
    symbol = _UNARY_SYMBOL[type(formula)]
    inner = render(formula.arg)
    separator = "" if symbol == "!" or inner.startswith("(") else " "
    return f"{symbol}{separator}{inner}"
