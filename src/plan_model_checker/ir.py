"""Typed plan IR: the only thing a planner may produce.

A plan is a tree of four node kinds:

- ``ToolCall``: one call to a registered tool. Arguments are resource constants or ``$var``
  references to an enclosing ``ForEach`` variable.
- ``If``: a branch on either a tracked proposition (``PropGuard``, evaluated against the abstract
  state statically and against the observed state at runtime) or an opaque, data-dependent
  predicate (``OpaqueGuard``), for which the checker explores both branches.
- ``ForEach``: a loop with a declared bound. With ``items`` it iterates over listed constants.
  With ``collection`` it iterates over a data-dependent collection of at most ``max_iterations``
  items; the checker explores every iteration count from 0 to the bound, binding the variable to
  fresh symbolic resources ``<collection>#<k>``.
- ``Plan``: the root, a goal plus a list of steps.

Structural limits (node count, nesting depth, loop bounds, unrolled size) are enforced by
:func:`check_plan_limits` before anything else looks at a plan, so a hostile plan cannot make the
checker allocate unbounded memory.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from plan_model_checker.core.canonical import canonical_json, sha256_hex

__all__ = [
    "IDENT_PATTERN",
    "RESOURCE_PATTERN",
    "TOOL_PATTERN",
    "ArgValue",
    "ForEach",
    "Guard",
    "If",
    "Node",
    "OpaqueGuard",
    "Plan",
    "PlanError",
    "PlanLimits",
    "PropGuard",
    "ResourceValue",
    "ToolArg",
    "ToolCall",
    "check_plan_limits",
    "insert_node",
    "is_valid_resource",
    "iter_calls",
    "node_at",
    "plan_sha256",
    "unrolled_call_count",
]

IDENT_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
TOOL_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
# Constants may not contain "#" (reserved for symbolic loop resources) or start with "$".
RESOURCE_PATTERN = r"^[A-Za-z0-9_./:@-][A-Za-z0-9_./:@-]{0,127}$"
_RESOURCE = re.compile(RESOURCE_PATTERN)
_VAR_REF = re.compile(r"^\$([a-z][a-z0-9_]{0,31})$")
_ARG_PATTERN = r"^(\$[a-z][a-z0-9_]{0,31}|[A-Za-z0-9_./:@-][A-Za-z0-9_./:@-]{0,127})$"
_DOT_SEGMENTS = frozenset({".", ".."})


def is_valid_resource(value: str) -> bool:
    """A resource constant: the allowed characters, and no ``.`` or ``..`` path segment.

    The checker treats resources as opaque names, so ``/srv/../etc`` and ``/etc`` would be two
    different resources to it and one directory to a file system. Dot segments are refused
    everywhere a resource can enter (plans, runtime collections, monitor reports, spec constants)
    so that this kind of aliasing cannot be expressed. Real tools must still resolve paths
    against an allowed root; see docs/SOUNDNESS.md.
    """
    return bool(_RESOURCE.fullmatch(value)) and not any(
        segment in _DOT_SEGMENTS for segment in value.split("/")
    )


def _no_dot_segments(value: str) -> str:
    if not value.startswith("$") and any(segment in _DOT_SEGMENTS for segment in value.split("/")):
        raise ValueError("resource names may not contain '.' or '..' path segments")
    return value


ArgValue = Annotated[str, Field(pattern=_ARG_PATTERN), AfterValidator(_no_dot_segments)]
ResourceValue = Annotated[str, Field(pattern=RESOURCE_PATTERN), AfterValidator(_no_dot_segments)]


class PlanError(ValueError):
    """A plan is malformed, exceeds a limit, or does not type-check against the registry."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolArg(_Frozen):
    """One named argument. ``value`` is a resource constant or a ``$var`` reference."""

    name: str = Field(pattern=IDENT_PATTERN)
    value: ArgValue


class PropGuard(_Frozen):
    kind: Literal["prop"] = "prop"
    prop: str = Field(pattern=IDENT_PATTERN)
    args: tuple[ArgValue, ...] = Field(default=(), max_length=4)
    negate: bool = False


class OpaqueGuard(_Frozen):
    kind: Literal["opaque"] = "opaque"
    label: str = Field(pattern=IDENT_PATTERN)


Guard = Annotated[PropGuard | OpaqueGuard, Field(discriminator="kind")]


class ToolCall(_Frozen):
    kind: Literal["call"] = "call"
    tool: str = Field(pattern=TOOL_PATTERN)
    args: tuple[ToolArg, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def _unique_argument_names(self) -> ToolCall:
        names = [arg.name for arg in self.args]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate argument names in call to {self.tool}")
        return self

    def arg_map(self) -> dict[str, str]:
        return {arg.name: arg.value for arg in self.args}


class If(_Frozen):
    kind: Literal["if"] = "if"
    cond: Guard
    then: tuple[Node, ...] = ()
    otherwise: tuple[Node, ...] = ()


class ForEach(_Frozen):
    kind: Literal["foreach"] = "foreach"
    var: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    items: tuple[ResourceValue, ...] | None = None
    collection: str | None = Field(default=None, pattern=IDENT_PATTERN)
    max_iterations: int = Field(ge=1)
    body: tuple[Node, ...] = ()

    @model_validator(mode="after")
    def _items_xor_collection(self) -> ForEach:
        if (self.items is None) == (self.collection is None):
            raise ValueError("ForEach needs exactly one of 'items' or 'collection'")
        if self.items is not None and len(self.items) > self.max_iterations:
            raise ValueError("ForEach lists more items than its max_iterations bound")
        return self


Node = Annotated[ToolCall | If | ForEach, Field(discriminator="kind")]


class Plan(_Frozen):
    version: Literal[1] = 1
    goal: str = Field(default="", max_length=500)
    steps: tuple[Node, ...] = ()


If.model_rebuild()
ForEach.model_rebuild()
Plan.model_rebuild()


@dataclass(frozen=True, slots=True)
class PlanLimits:
    """Structural budgets applied before checking (DoS protection)."""

    max_nodes: int = 200
    max_depth: int = 8
    max_loop_bound: int = 8
    max_unrolled_calls: int = 2_000


def plan_sha256(plan: Plan) -> str:
    return sha256_hex(canonical_json(plan.model_dump(mode="json")))


def _children(node: ToolCall | If | ForEach) -> Iterator[tuple[str, tuple[Node, ...]]]:
    if isinstance(node, If):
        yield "then", node.then
        yield "otherwise", node.otherwise
    elif isinstance(node, ForEach):
        yield "body", node.body


def iter_calls(steps: tuple[Node, ...], prefix: str = "steps") -> Iterator[tuple[str, ToolCall]]:
    """Every ToolCall in the tree with its path, in source order (loops not unrolled)."""
    stack: list[tuple[str, tuple[Node, ...]]] = [(prefix, steps)]
    order: list[tuple[str, ToolCall]] = []
    while stack:
        path, block = stack.pop()
        for index, node in enumerate(block):
            here = f"{path}/{index}"
            if isinstance(node, ToolCall):
                order.append((here, node))
            for name, child in _children(node):
                stack.append((f"{here}/{name}", child))
    order.sort(key=lambda item: _path_key(item[0]))
    yield from order


def _path_key(path: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in path.split("/"))


def node_at(plan: Plan, path: str) -> ToolCall | If | ForEach:
    """Resolve a path such as ``steps/2/then/0`` to its node."""
    parts = path.split("/")
    if not parts or parts[0] != "steps" or len(parts) % 2 != 0:
        raise PlanError(f"invalid node path {path!r}")
    block: tuple[Node, ...] = plan.steps
    node: ToolCall | If | ForEach | None = None
    for position in range(1, len(parts), 2):
        index_text = parts[position]
        if not index_text.isdigit() or int(index_text) >= len(block):
            raise PlanError(f"invalid node path {path!r}")
        node = block[int(index_text)]
        if position + 1 < len(parts):
            field = parts[position + 1]
            children = dict(_children(node))
            if field not in children:
                raise PlanError(f"invalid node path {path!r}")
            block = children[field]
    if node is None:
        raise PlanError(f"invalid node path {path!r}")
    return node


def unrolled_call_count(steps: tuple[Node, ...]) -> int:
    """Tool calls in the plan with every loop unrolled to its bound (both branches counted).

    This is the number of call sites in the unrolled control-flow graph, an upper bound on the
    length of any execution.
    """
    total = 0
    for node in steps:
        if isinstance(node, ToolCall):
            total += 1
        elif isinstance(node, If):
            total += unrolled_call_count(node.then) + unrolled_call_count(node.otherwise)
        else:
            iterations = len(node.items) if node.items is not None else node.max_iterations
            total += iterations * unrolled_call_count(node.body)
    return total


def check_plan_limits(plan: Plan, limits: PlanLimits) -> None:
    """Raise PlanError when the plan exceeds a structural budget or misuses variables.

    Iterative, so a deeply nested plan cannot exhaust the Python stack here.
    """
    nodes = 0
    stack: list[tuple[tuple[Node, ...], int, frozenset[str]]] = [(plan.steps, 1, frozenset())]
    while stack:
        block, depth, bound_vars = stack.pop()
        if block and depth > limits.max_depth:
            raise PlanError(f"plan nesting depth exceeds {limits.max_depth}")
        for node in block:
            nodes += 1
            if nodes > limits.max_nodes:
                raise PlanError(f"plan has more than {limits.max_nodes} nodes")
            if isinstance(node, ToolCall):
                _check_refs([arg.value for arg in node.args], bound_vars)
            elif isinstance(node, If):
                if isinstance(node.cond, PropGuard):
                    _check_refs(list(node.cond.args), bound_vars)
                stack.append((node.then, depth + 1, bound_vars))
                stack.append((node.otherwise, depth + 1, bound_vars))
            else:
                if node.max_iterations > limits.max_loop_bound:
                    raise PlanError(
                        f"loop bound {node.max_iterations} exceeds {limits.max_loop_bound}"
                    )
                if node.var in bound_vars:
                    raise PlanError(f"loop variable ${node.var} shadows an enclosing loop")
                stack.append((node.body, depth + 1, bound_vars | {node.var}))
    if unrolled_call_count(plan.steps) > limits.max_unrolled_calls:
        raise PlanError(f"unrolled plan exceeds {limits.max_unrolled_calls} tool calls")


def _check_refs(values: list[str], bound_vars: frozenset[str]) -> None:
    for value in values:
        match = _VAR_REF.fullmatch(value)
        if match is not None and match.group(1) not in bound_vars:
            raise PlanError(f"variable {value} is not bound by an enclosing ForEach")


def insert_node(plan: Plan, anchor_path: str | None, new: Node, *, after: bool = False) -> Plan:
    """A copy of ``plan`` with ``new`` inserted next to the node at ``anchor_path``.

    ``anchor_path=None`` appends to the top-level steps. The result is re-validated.
    """
    if anchor_path is None:
        steps: tuple[Node, ...] = (*plan.steps, new)
    else:
        node_at(plan, anchor_path)  # validates the path
        steps = _insert(plan.steps, anchor_path.split("/")[1:], new, after=after)
    return Plan.model_validate(
        {**plan.model_dump(mode="json"), "steps": [n.model_dump(mode="json") for n in steps]}
    )


def _insert(
    block: tuple[Node, ...], rest: list[str], new: Node, *, after: bool
) -> tuple[Node, ...]:
    index = int(rest[0])
    if len(rest) == 1:
        position = index + 1 if after else index
        return (*block[:position], new, *block[position:])
    child = block[index]
    field = rest[1]
    inner = dict(_children(child))[field]
    updated = child.model_copy(update={field: _insert(inner, rest[2:], new, after=after)})
    return (*block[:index], updated, *block[index + 1 :])
