from __future__ import annotations

import pytest
from pydantic import ValidationError

from plan_model_checker.corpus import call, each, each_of, if_opaque, if_prop, plan
from plan_model_checker.ir import (
    Plan,
    PlanError,
    PlanLimits,
    ToolCall,
    check_plan_limits,
    insert_node,
    iter_calls,
    node_at,
    plan_sha256,
    unrolled_call_count,
)

pytestmark = pytest.mark.unit


def nested(depth: int) -> Plan:
    step = call("snapshot_dir", path="/a")
    for _ in range(depth):
        step = if_opaque("x", [step])
    return plan("deep", step)


def test_plans_are_closed_and_validated() -> None:
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [], "specs": ["waived"]})
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [{"kind": "call", "tool": "Bad Tool"}]})
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [call("snapshot_dir", path="has#hash")]})
    with pytest.raises(ValidationError):
        Plan.model_validate(
            {"steps": [{"kind": "call", "tool": "t", "args": [{"name": "a", "value": "x"}] * 2}]}
        )


def test_foreach_needs_items_xor_collection_within_its_bound() -> None:
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [{"kind": "foreach", "var": "d", "max_iterations": 1}]})
    with pytest.raises(ValidationError):
        Plan.model_validate(
            {"steps": [{"kind": "foreach", "var": "d", "items": ["/a", "/b"], "max_iterations": 1}]}
        )


def test_structural_limits_are_enforced_before_checking() -> None:
    limits = PlanLimits(max_nodes=5, max_depth=3, max_loop_bound=2, max_unrolled_calls=4)
    check_plan_limits(nested(2), limits)
    with pytest.raises(PlanError, match="depth"):
        check_plan_limits(nested(3), limits)
    with pytest.raises(PlanError, match="nodes"):
        check_plan_limits(plan("wide", *[call("snapshot_dir", path="/a")] * 6), limits)
    with pytest.raises(PlanError, match="loop bound"):
        check_plan_limits(plan("loop", each_of("d", "c", 3, [])), limits)
    with pytest.raises(PlanError, match="unrolled"):
        check_plan_limits(
            plan("big", each_of("d", "c", 2, [call("snapshot_dir", path="$d")] * 3)), limits
        )


def test_variables_must_be_bound_and_not_shadowed() -> None:
    with pytest.raises(PlanError, match="not bound"):
        check_plan_limits(plan("x", call("snapshot_dir", path="$d")), PlanLimits())
    with pytest.raises(PlanError, match="not bound"):
        check_plan_limits(plan("x", if_prop("approved", "$d", [])), PlanLimits())
    shadow = plan("x", each("d", ["/a"], [each("d", ["/b"], [])]))
    with pytest.raises(PlanError, match="shadows"):
        check_plan_limits(shadow, PlanLimits())


def test_unrolled_size_counts_loops_and_both_branches() -> None:
    subject = plan(
        "x",
        each_of("d", "c", 3, [call("snapshot_dir", path="$d"), if_opaque("o", [call("a", x="1")])]),
        each("e", ["/a", "/b"], [call("b", x="$e")]),
    )
    assert unrolled_call_count(subject.steps) == 3 * 2 + 2


def test_paths_resolve_and_insertions_land_next_to_the_anchor() -> None:
    subject = plan(
        "x",
        call("request_approval", path="/a"),
        if_prop("approved", "/a", [call("archive_dir", path="/a"), call("delete_dir", path="/a")]),
    )
    assert [path for path, _ in iter_calls(subject.steps)] == [
        "steps/0",
        "steps/1/then/0",
        "steps/1/then/1",
    ]
    anchor = node_at(subject, "steps/1/then/1")
    assert isinstance(anchor, ToolCall)
    assert anchor.tool == "delete_dir"
    snapshot = ToolCall.model_validate(call("snapshot_dir", path="/a"))
    before = insert_node(subject, "steps/1/then/1", snapshot)
    assert [n.tool for _, n in iter_calls(before.steps)] == [
        "request_approval",
        "archive_dir",
        "snapshot_dir",
        "delete_dir",
    ]
    after = insert_node(subject, "steps/0", snapshot, after=True)
    assert [n.tool for _, n in iter_calls(after.steps)][:2] == ["request_approval", "snapshot_dir"]
    appended = insert_node(subject, None, snapshot)
    assert [n.tool for _, n in iter_calls(appended.steps)][-1] == "snapshot_dir"
    assert plan_sha256(subject) != plan_sha256(before)
    for bad in ("steps/9", "steps/1/body/0", "other/0", "steps/x", "steps/1/then"):
        with pytest.raises(PlanError):
            node_at(subject, bad)


def test_plan_hash_is_canonical() -> None:
    first = plan("goal", call("snapshot_dir", path="/a"))
    second = Plan.model_validate_json(first.model_dump_json())
    assert plan_sha256(first) == plan_sha256(second)


@pytest.mark.parametrize("value", ["..", ".", "../etc", "/srv/../etc", "/srv/./logs", "/srv/.."])
def test_dot_segments_are_not_valid_resources(value: str) -> None:
    """``/srv/../etc`` and ``/etc`` would be different resources to the checker, one to a disk."""
    from plan_model_checker.ir import is_valid_resource

    assert not is_valid_resource(value)
    with pytest.raises(ValidationError):
        plan("x", call("archive_dir", path=value))
    with pytest.raises(ValidationError):
        plan("x", each("d", [value], [call("archive_dir", path="$d")]))
    with pytest.raises(ValidationError):
        plan("x", if_prop("approved", value, []))


@pytest.mark.parametrize("value", ["/srv/app..v2", "/srv/.hidden", "a.b", "svc:a", "job-1"])
def test_dots_inside_segment_names_are_fine(value: str) -> None:
    from plan_model_checker.ir import is_valid_resource

    assert is_valid_resource(value)
    plan("x", call("archive_dir", path=value))


@pytest.mark.parametrize("value", ["", "has space", "a;b", "$d", "x#1", "a" * 129])
def test_malformed_resources_are_invalid(value: str) -> None:
    from plan_model_checker.ir import is_valid_resource

    assert not is_valid_resource(value)
