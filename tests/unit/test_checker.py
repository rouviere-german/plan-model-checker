from __future__ import annotations

import pytest

from plan_model_checker.checker import (
    CheckBudgetExceededError,
    CheckLimits,
    ModelChecker,
    build_cfg,
    collection_bounds,
    size_assignments,
)
from plan_model_checker.corpus import call, each, each_of, if_opaque, if_prop, plan
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.ir import Plan, PlanError, ToolCall
from plan_model_checker.oracle import oracle_check
from plan_model_checker.props import GroundProp
from plan_model_checker.spec.specs import SpecSet

pytestmark = pytest.mark.unit

PATH = "/data/x"


def approved_delete(*before_delete: dict[str, object], path: str = PATH) -> Plan:
    return plan(
        "delete",
        call("request_approval", path=path),
        if_prop(
            "approved",
            path,
            [call("archive_dir", path=path), *before_delete, call("delete_dir", path=path)],
        ),
    )


def test_a_delete_without_backup_is_rejected_with_a_replayable_counterexample(
    checker: ModelChecker,
) -> None:
    result = checker.check(approved_delete())
    assert result.verdict == "violation"
    assert result.violated_specs() == ("backup_before_delete",)
    (violation,) = result.violations
    cex = violation.counterexample
    assert [(s.call.tool, s.outcome) for s in cex.steps] == [
        ("request_approval", "granted"),
        ("archive_dir", "ok"),
        ("delete_dir", "ok"),
    ]
    assert cex.instance.binding == (("r", PATH),)
    assert cex.violation_position == 3
    assert GroundProp("delete", (PATH,)) in cex.letters[3]
    assert violation.instance_keys == ("backup_before_delete[r=/data/x]",)
    assert result.to_json()["violations"][0]["counterexample"]["steps"][2]["tool"] == "delete_dir"


def test_guards_on_tracked_propositions_are_path_sensitive(checker: ModelChecker) -> None:
    guarded = approved_delete(call("snapshot_dir", path=PATH))
    assert checker.check(guarded).ok
    unguarded = plan(
        "delete",
        call("request_approval", path=PATH),
        call("archive_dir", path=PATH),
        call("snapshot_dir", path=PATH),
        call("delete_dir", path=PATH),
    )
    result = checker.check(unguarded)
    # request_approval may stay pending, so without the guard the delete can run unapproved.
    assert result.violated_specs() == ("approval_delete_dir",)
    assert result.violations[0].counterexample.steps[0].outcome == "pending"


def test_both_branches_of_an_opaque_predicate_are_explored(checker: ModelChecker) -> None:
    subject = plan(
        "compact",
        call("acquire_lock", path=PATH),
        if_opaque("big", [call("compact_dir", path=PATH)], [call("release_lock", path=PATH)]),
    )
    result = checker.check(subject)
    assert result.violated_specs() == ("release_all_locks",)
    cex = result.violations[0].counterexample
    assert [(d.site, d.choice) for d in cex.decisions] == [("steps/1", "then")]


def test_counterexamples_are_shortest(checker: ModelChecker) -> None:
    # Two ways to violate release_all_locks: a long branch and a short one.
    subject = plan(
        "compact",
        call("acquire_lock", path=PATH),
        if_opaque(
            "slow",
            [call("compact_dir", path=PATH)] * 4,
            [],
        ),
    )
    cex = checker.check(subject).violations[0].counterexample
    assert cex.length == 1
    assert [(d.site, d.choice) for d in cex.decisions] == [("steps/1", "otherwise")]


def test_collection_sizes_from_zero_to_the_bound_are_all_checked(
    checker: ModelChecker,
) -> None:
    subject = plan(
        "retry every job",
        each_of(
            "j",
            "jobs",
            2,
            [call("submit_job", job="$j"), call("retry_job", job="$j")],
        ),
    )
    result = checker.check(subject)
    assert result.size_assignments == 3
    cex = result.violations[0].counterexample
    assert cex.sizes == (("jobs", 1),)
    assert cex.instance.binding == (("j", "jobs#0"),)
    assert collection_bounds(subject) == {"jobs": 2}
    assert size_assignments({"b": 1, "a": 1}) == [
        (("a", 0), ("b", 0)),
        (("a", 0), ("b", 1)),
        (("a", 1), ("b", 0)),
        (("a", 1), ("b", 1)),
    ]


def test_loops_over_one_collection_see_the_same_items(checker: ModelChecker) -> None:
    subject = plan(
        "back up everything, then delete everything",
        each_of("d", "stale", 2, [call("snapshot_dir", path="$d")]),
        each_of(
            "d",
            "stale",
            2,
            [
                call("archive_dir", path="$d"),
                call("request_approval", path="$d"),
                if_prop("approved", "$d", [call("delete_dir", path="$d")]),
            ],
        ),
    )
    result = checker.check(subject)
    assert result.ok, result.violated_specs()
    with pytest.raises(PlanError, match="different bounds"):
        collection_bounds(plan("x", each_of("d", "c", 1, []), each_of("e", "c", 2, [])))


def test_memoization_keeps_states_linear_while_paths_grow_exponentially(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    diamonds = [
        if_opaque(f"b{i}", [call("check_status", job="j")], [call("check_status", job="j")])
        for i in range(10)
    ]
    subject = plan("diamonds", call("submit_job", job="j"), *diamonds, call("retry_job", job="j"))
    result = ModelChecker(registry, specs).check(subject)
    assert result.ok
    paths = 2**10
    assert result.states_explored < 200 < paths
    assert oracle_check(subject, registry, specs, max_traces=10_000).traces == paths


def test_plans_are_typed_against_the_registry(checker: ModelChecker) -> None:
    cases = {
        "not registered": plan("x", call("rm_rf", path=PATH)),
        "takes parameters": plan("x", call("snapshot_dir", target=PATH)),
        "used both as": plan(
            "x", call("snapshot_dir", path="shared"), call("drain_service", service="shared")
        ),
        "only test fluents": plan("x", if_prop("delete", PATH, [])),
        "not declared": plan("x", if_prop("nonexistent", PATH, [])),
        "wrong number": plan("x", {"kind": "if", "cond": {"kind": "prop", "prop": "approved"}}),
    }
    for message, subject in cases.items():
        with pytest.raises(PlanError, match=message):
            checker.check(subject)


def test_budgets_bound_states_collections_and_size_assignments(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    subject = approved_delete(call("snapshot_dir", path=PATH))
    with pytest.raises(CheckBudgetExceededError) as info:
        ModelChecker(registry, specs, CheckLimits(max_states=5)).check(subject)
    assert info.value.dimension == "states"
    many = plan(
        "x", *(each_of("d", f"c{i}", 1, [call("snapshot_dir", path="$d")]) for i in range(4))
    )
    with pytest.raises(PlanError, match="collections"):
        ModelChecker(registry, specs).check(many)
    with pytest.raises(CheckBudgetExceededError, match="size assignments"):
        ModelChecker(registry, specs, CheckLimits(max_size_assignments=3)).check(
            plan("x", each_of("d", "c", 3, []))
        )
    with pytest.raises(CheckBudgetExceededError, match="instances"):
        ModelChecker(registry, specs, CheckLimits(max_instances=3)).check(subject)


def test_spec_selection_keeps_the_implicit_approval_rule(checker: ModelChecker) -> None:
    unapproved = plan("x", call("snapshot_dir", path=PATH), call("delete_dir", path=PATH))
    result = checker.check(unapproved, ["release_all_locks"])
    assert result.violated_specs() == ("approval_delete_dir",)


def test_control_flow_graph_shape(registry: EffectRegistry) -> None:
    subject = plan(
        "x",
        each("d", ["/a", "/b"], [call("snapshot_dir", path="$d")]),
        if_opaque("o", [call("archive_dir", path="/a")]),
    )
    cfg = build_cfg(subject, registry)
    calls = [edge.call for edges in cfg.edges for edge in edges if edge.call is not None]
    assert [c.label() for c in calls if c is not None] == [
        "archive_dir(path=/a)",
        "snapshot_dir(path=/b)",
        "snapshot_dir(path=/a)",
    ]
    assert cfg.universe == {"path": ("/a", "/b")}
    assert cfg.node_count == 5
    assert cfg.edges[cfg.exit] == ()


def test_empty_plans_are_checked_at_the_start_and_end_positions(
    registry: EffectRegistry,
) -> None:
    specs = SpecSet.parse("spec never_end: G(!end);", registry)
    result = ModelChecker(registry, specs).check(Plan())
    assert result.violated_specs() == ("never_end",)
    assert result.violations[0].counterexample.length == 0
    assert ToolCall.model_validate(call("snapshot_dir", path="/a")).arg_map() == {"path": "/a"}
