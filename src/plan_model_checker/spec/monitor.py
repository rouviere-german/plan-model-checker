"""Compile a ground past-time formula into a per-step boolean monitor.

Following Havelund & Roşu (TACAS 2002), each subformula gets one bit. At every trace position the
bits are recomputed in post-order from the current letter (a bitmask of true propositions) and
the previous position's bits:

    Y φ   : pre[φ]                    (false at position 0)
    Z φ   : pre[φ]                    (true at position 0)
    O φ   : now[φ] or pre[Oφ]         (now[φ] at position 0)
    H φ   : now[φ] and pre[Hφ]        (now[φ] at position 0)
    φ S ψ : now[ψ] or (now[φ] and pre[φSψ])   (now[ψ] at position 0)

Only the bits read through ``pre`` have to survive between steps (``memory_mask``), so the
monitor *state* is ``now & memory_mask``. That state is finite, which makes explicit-state model
checking over (control point, abstract state, monitor state) terminate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
    subformulas_postorder,
)

__all__ = ["CompiledMonitor", "compile_monitor"]

_ATOM, _TRUE, _FALSE, _NOT, _AND, _OR, _IMPLIES, _IFF, _Y, _Z, _O, _H, _S = range(13)


@dataclass(frozen=True, slots=True)
class CompiledMonitor:
    """A compiled formula. Letters are bitmasks over :attr:`props`."""

    props: tuple[GroundProp, ...]
    code: tuple[tuple[int, int, int], ...]
    memory_mask: int
    root: int

    def prop_bits(self) -> dict[GroundProp, int]:
        return {prop: 1 << index for index, prop in enumerate(self.props)}

    def letter(self, true_props: Iterable[GroundProp]) -> int:
        bits = self.prop_bits()
        value = 0
        for prop in true_props:
            value |= bits.get(prop, 0)
        return value

    def first(self, letter: int) -> int:
        """Subformula bits at position 0."""
        return self._run(0, letter, initial=True)

    def step(self, pre: int, letter: int) -> int:
        """Subformula bits at a position > 0, given the previous position's bits."""
        return self._run(pre, letter, initial=False)

    def holds(self, bits: int) -> bool:
        return bool((bits >> self.root) & 1)

    def state(self, bits: int) -> int:
        """The part of ``bits`` the next step depends on."""
        return bits & self.memory_mask

    def _run(self, pre: int, letter: int, *, initial: bool) -> int:  # noqa: PLR0912
        now = 0
        for index, (op, a, b) in enumerate(self.code):
            if op == _ATOM:
                value = (letter >> a) & 1
            elif op == _TRUE:
                value = 1
            elif op == _FALSE:
                value = 0
            elif op == _NOT:
                value = 1 - ((now >> a) & 1)
            elif op == _AND:
                value = (now >> a) & (now >> b) & 1
            elif op == _OR:
                value = ((now >> a) | (now >> b)) & 1
            elif op == _IMPLIES:
                value = (((now >> a) & 1) ^ 1) | ((now >> b) & 1)
            elif op == _IFF:
                value = 1 - (((now >> a) ^ (now >> b)) & 1)
            elif op == _Y:
                value = 0 if initial else (pre >> a) & 1
            elif op == _Z:
                value = 1 if initial else (pre >> a) & 1
            elif op == _O:
                value = (now >> a) & 1 if initial else ((now >> a) | (pre >> index)) & 1
            elif op == _H:
                value = (now >> a) & 1 if initial else (now >> a) & (pre >> index) & 1
            else:  # _S
                right = (now >> b) & 1
                value = right if initial else right | ((now >> a) & (pre >> index) & 1)
            now |= value << index
        return now


def compile_monitor(formula: Formula) -> CompiledMonitor:
    """Compile a ground, pure past-time formula (no G, no variables)."""
    nodes = subformulas_postorder(formula)
    index_of = {node: index for index, node in enumerate(nodes)}
    props: list[GroundProp] = []
    prop_index: dict[GroundProp, int] = {}
    code: list[tuple[int, int, int]] = []
    memory = 0
    for index, node in enumerate(nodes):
        if isinstance(node, Atom):
            prop = ground_prop(node)
            if prop not in prop_index:
                prop_index[prop] = len(props)
                props.append(prop)
            code.append((_ATOM, prop_index[prop], 0))
        elif isinstance(node, Bool):
            code.append((_TRUE if node.value else _FALSE, 0, 0))
        elif isinstance(node, Not):
            code.append((_NOT, index_of[node.arg], 0))
        elif isinstance(node, And | Or | Implies | Iff):
            op = {And: _AND, Or: _OR, Implies: _IMPLIES, Iff: _IFF}[type(node)]
            code.append((op, index_of[node.left], index_of[node.right]))
        elif isinstance(node, Yesterday | WeakYesterday):
            arg = index_of[node.arg]
            code.append((_Y if isinstance(node, Yesterday) else _Z, arg, 0))
            memory |= 1 << arg
        elif isinstance(node, Once | Historically):
            code.append((_O if isinstance(node, Once) else _H, index_of[node.arg], 0))
            memory |= 1 << index
        elif isinstance(node, Since):
            code.append((_S, index_of[node.left], index_of[node.right]))
            memory |= 1 << index
        elif isinstance(node, Always):
            raise TypeError("G must be rewritten to H before compiling (see final_form)")
    return CompiledMonitor(
        props=tuple(props),
        code=tuple(code),
        memory_mask=memory,
        root=index_of[formula],
    )
