"""A flat, non-recursive encoding of plans for LLM structured output.

Strict structured outputs and strict tool schemas work best with closed, non-recursive schemas in
which every property is required. The tree IR is recursive, so planners emit this flat form
instead: a list of nodes with integer ids, where ``If`` and ``ForEach`` refer to their children by
id and ``root`` lists the top-level steps. :func:`wire_to_plan` rebuilds the tree and rejects
anything that is not one: unknown or duplicate ids, a node referenced twice, a cycle, or an
unreachable node. The result is then validated as an ordinary :class:`~plan_model_checker.ir.Plan`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from plan_model_checker.ir import ForEach, If, Node, OpaqueGuard, Plan, PlanError, ToolCall

__all__ = [
    "WIRE_MAX_NODES",
    "WireArg",
    "WireCond",
    "WireNode",
    "WirePlan",
    "plan_to_wire",
    "wire_schema",
    "wire_to_plan",
]

WIRE_MAX_NODES = 200


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WireArg(_Strict):
    name: str
    value: str


class WireCond(_Strict):
    kind: Literal["prop", "opaque"]
    prop: str | None
    args: list[str]
    negate: bool
    label: str | None


class WireNode(_Strict):
    id: int = Field(ge=0)
    kind: Literal["call", "if", "foreach"]
    tool: str | None
    args: list[WireArg]
    cond: WireCond | None
    then: list[int]
    otherwise: list[int]
    var: str | None
    items: list[str] | None
    collection: str | None
    max_iterations: int | None
    body: list[int]


class WirePlan(_Strict):
    goal: str
    root: list[int]
    nodes: list[WireNode] = Field(max_length=WIRE_MAX_NODES)


def wire_schema() -> dict[str, JsonValue]:
    schema: dict[str, JsonValue] = WirePlan.model_json_schema()
    return schema


def _cond_json(cond: WireCond) -> dict[str, Any]:
    if cond.kind == "opaque":
        return {"kind": "opaque", "label": cond.label}
    return {"kind": "prop", "prop": cond.prop, "args": cond.args, "negate": cond.negate}


def wire_to_plan(wire: WirePlan | dict[str, Any]) -> Plan:
    """Rebuild and validate the tree. Raises PlanError on any structural problem."""
    try:
        parsed = wire if isinstance(wire, WirePlan) else WirePlan.model_validate(wire)
    except ValidationError as exc:
        raise PlanError("plan does not match the wire schema") from exc
    nodes = {node.id: node for node in parsed.nodes}
    if len(nodes) != len(parsed.nodes):
        raise PlanError("duplicate node ids")
    used: set[int] = set()

    def take(ids: list[int]) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        for node_id in ids:
            if node_id not in nodes:
                raise PlanError(f"reference to unknown node {node_id}")
            if node_id in used:
                raise PlanError(f"node {node_id} is referenced more than once")
            used.add(node_id)
            built.append(build(nodes[node_id]))
        return built

    def build(node: WireNode) -> dict[str, Any]:
        if node.kind == "call":
            if node.tool is None:
                raise PlanError(f"call node {node.id} has no tool")
            return {
                "kind": "call",
                "tool": node.tool,
                "args": [{"name": a.name, "value": a.value} for a in node.args],
            }
        if node.kind == "if":
            if node.cond is None:
                raise PlanError(f"if node {node.id} has no condition")
            return {
                "kind": "if",
                "cond": _cond_json(node.cond),
                "then": take(node.then),
                "otherwise": take(node.otherwise),
            }
        return {
            "kind": "foreach",
            "var": node.var,
            "items": node.items,
            "collection": node.collection,
            "max_iterations": node.max_iterations,
            "body": take(node.body),
        }

    try:
        steps = take(parsed.root)
    except RecursionError as exc:  # pragma: no cover - bounded by WIRE_MAX_NODES
        raise PlanError("plan is nested too deeply") from exc
    if used != set(nodes):
        raise PlanError("plan contains unreachable nodes")
    try:
        return Plan.model_validate({"goal": parsed.goal[:500], "steps": steps})
    except ValidationError as exc:
        raise PlanError("plan failed validation") from exc


def plan_to_wire(plan: Plan) -> WirePlan:
    nodes: list[WireNode] = []

    def emit(block: tuple[Node, ...]) -> list[int]:
        ids: list[int] = []
        for node in block:
            node_id = len(nodes)
            nodes.append(_placeholder(node_id))
            ids.append(node_id)
            nodes[node_id] = convert(node, node_id)
        return ids

    def convert(node: Node, node_id: int) -> WireNode:
        empty: dict[str, Any] = {
            "id": node_id,
            "tool": None,
            "args": [],
            "cond": None,
            "then": [],
            "otherwise": [],
            "var": None,
            "items": None,
            "collection": None,
            "max_iterations": None,
            "body": [],
        }
        if isinstance(node, ToolCall):
            return WireNode.model_validate(
                {
                    **empty,
                    "kind": "call",
                    "tool": node.tool,
                    "args": [{"name": a.name, "value": a.value} for a in node.args],
                }
            )
        if isinstance(node, If):
            cond = node.cond
            wire_cond = (
                {"kind": "opaque", "prop": None, "args": [], "negate": False, "label": cond.label}
                if isinstance(cond, OpaqueGuard)
                else {
                    "kind": "prop",
                    "prop": cond.prop,
                    "args": list(cond.args),
                    "negate": cond.negate,
                    "label": None,
                }
            )
            return WireNode.model_validate(
                {
                    **empty,
                    "kind": "if",
                    "cond": wire_cond,
                    "then": emit(node.then),
                    "otherwise": emit(node.otherwise),
                }
            )
        loop: ForEach = node
        return WireNode.model_validate(
            {
                **empty,
                "kind": "foreach",
                "var": loop.var,
                "items": list(loop.items) if loop.items is not None else None,
                "collection": loop.collection,
                "max_iterations": loop.max_iterations,
                "body": emit(loop.body),
            }
        )

    root = emit(plan.steps)
    return WirePlan(goal=plan.goal, root=root, nodes=nodes)


def _placeholder(node_id: int) -> WireNode:
    return WireNode(
        id=node_id,
        kind="call",
        tool="placeholder",
        args=[],
        cond=None,
        then=[],
        otherwise=[],
        var=None,
        items=None,
        collection=None,
        max_iterations=None,
        body=[],
    )
