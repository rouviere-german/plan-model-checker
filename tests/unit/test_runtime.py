from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStore
from plan_model_checker.agentic.audit import AuditLog, verify_file
from plan_model_checker.agentic.errors import ApprovalError
from plan_model_checker.checker import GroundCall, ModelChecker
from plan_model_checker.corpus import call, each, each_of, if_opaque, if_prop, plan
from plan_model_checker.effects import EffectRegistry, UnknownToolError
from plan_model_checker.ir import Plan
from plan_model_checker.override import OverrideError, OverrideGate
from plan_model_checker.props import GroundProp
from plan_model_checker.runtime import (
    Observation,
    PlanExecutor,
    RuntimeMonitor,
    ToolGateway,
    ToolPinError,
    describe_outputs,
)
from plan_model_checker.spec.specs import SpecSet
from plan_model_checker.world import FileMaintenanceWorld, ScopedWorld, build_tools

pytestmark = pytest.mark.unit

PATH = "/data/x"


def approve(record: ApprovalRecord, store: ApprovalStore) -> None:
    store.decide(record.id, approver="human:ops", approve=True)


def deny(record: ApprovalRecord, store: ApprovalStore) -> None:
    store.decide(record.id, approver="human:ops", approve=False)


def executor(
    registry: EffectRegistry,
    specs: SpecSet,
    world: FileMaintenanceWorld,
    *,
    audit: AuditLog | None = None,
    waived: frozenset[str] = frozenset(),
    sensor: bool = True,
) -> PlanExecutor:
    return PlanExecutor(
        registry,
        specs,
        ToolGateway(registry, build_tools(world, registry)),
        opaque=world.decide,
        collections=world.collection,
        audit=audit,
        waived=waived,
        sensor=world.sense if sensor else None,
    )


def safe_delete(path: str = PATH) -> Plan:
    return plan(
        "delete",
        call("request_approval", path=path),
        if_prop(
            "approved",
            path,
            [
                call("archive_dir", path=path),
                call("snapshot_dir", path=path),
                call("delete_dir", path=path),
            ],
        ),
    )


def gc(tool: str, **args: str) -> GroundCall:
    return GroundCall(tool=tool, args=tuple(args.items()), path="test")


# --- monitor ------------------------------------------------------------------------------------


def test_monitor_blocks_before_the_step_and_reports_violations_once(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    monitor = RuntimeMonitor(registry, specs)
    assert monitor.start() == []
    blocked = monitor.precheck(gc("delete_dir", path=PATH))
    assert {v.spec for v in blocked} == {
        "backup_before_delete",
        "archive_before_delete",
        "approval_delete_dir",
    }
    assert all(v.kind == "blocked" and v.position == 1 for v in blocked)
    assert monitor.position == 0  # nothing was committed
    after = monitor.commit(gc("delete_dir", path=PATH), ((GroundProp("deleted", (PATH,)), True),))
    assert {v.kind for v in after} == {"after_execution"}
    assert monitor.commit(gc("archive_dir", path=PATH), ()) == []  # already reported
    assert monitor.value(GroundProp("deleted", (PATH,))) is True
    assert GroundProp("backup_ok", (PATH,)) in monitor.tracked_fluents()


def test_new_resources_replay_the_history_so_far(registry: EffectRegistry) -> None:
    specs = SpecSet.parse("spec restart_after_start: G(restart[s] -> O drain[s]);", registry)
    monitor = RuntimeMonitor(registry, specs)
    monitor.start()
    monitor.commit(gc("drain_service", service="svc:a"), ())
    assert monitor.instance_keys() == ("restart_after_start[s=svc:a]",)
    # svc:b appears only now; its instance must see that it was never drained.
    assert [v.instance for v in monitor.precheck(gc("restart_service", service="svc:b"))] == [
        "restart_after_start[s=svc:b]"
    ]
    assert monitor.precheck(gc("restart_service", service="svc:a")) == []


def test_end_rules_are_evaluated_at_finish(registry: EffectRegistry, specs: SpecSet) -> None:
    monitor = RuntimeMonitor(registry, specs)
    monitor.start()
    monitor.commit(gc("acquire_lock", path=PATH), ((GroundProp("lock_held", (PATH,)), True),))
    (violation,) = monitor.finish()
    assert violation.spec == "release_all_locks"
    assert violation.kind == "at_end"


def test_monitor_protocol_order_is_enforced(registry: EffectRegistry, specs: SpecSet) -> None:
    monitor = RuntimeMonitor(registry, specs)
    actions: list[Callable[[], object]] = [
        lambda: monitor.precheck(gc("archive_dir", path=PATH)),
        lambda: monitor.commit(gc("archive_dir", path=PATH), ()),
        monitor.finish,
    ]
    for action in actions:
        with pytest.raises(RuntimeError):
            action()
    monitor.start()
    with pytest.raises(RuntimeError):
        monitor.start()


# --- gateway ------------------------------------------------------------------------------------


@dataclass
class _Tool:
    name: str
    declared_sha256: str

    def __call__(self, args: Mapping[str, str]) -> Observation:
        return Observation(outcome="ok")


def test_gateway_admits_only_registered_tools_pinned_to_their_declaration(
    registry: EffectRegistry,
) -> None:
    good = _Tool("archive_dir", registry.tool_sha256("archive_dir"))
    gateway = ToolGateway(registry, [good])
    assert gateway.execute(gc("archive_dir", path=PATH)).outcome == "ok"
    with pytest.raises(UnknownToolError):
        gateway.execute(gc("snapshot_dir", path=PATH))
    with pytest.raises(ToolPinError, match="not in the signed registry"):
        ToolGateway(registry, [_Tool("rm_rf", "0" * 64)])
    with pytest.raises(ToolPinError, match="different declaration"):
        ToolGateway(registry, [_Tool("archive_dir", registry.tool_sha256("delete_dir"))])
    with pytest.raises(ToolPinError, match="twice"):
        ToolGateway(registry, [good, good])


def test_world_tools_refuse_a_registry_that_changed_their_declaration(
    registry: EffectRegistry,
) -> None:
    world = FileMaintenanceWorld()
    pins = {name: registry.tool_sha256(name) for name in registry.tool_names}
    pins["delete_dir"] = "f" * 64  # implementation written against another declaration
    with pytest.raises(ToolPinError):
        ToolGateway(registry, build_tools(world, registry, pins=pins))


# --- executor -----------------------------------------------------------------------------------


def test_a_verified_plan_runs_and_the_audit_chain_records_it(
    registry: EffectRegistry, specs: SpecSet, tmp_path: Path
) -> None:
    world = FileMaintenanceWorld(dirs={PATH: 1}, reviewer=approve)
    audit = AuditLog(tmp_path / "audit.jsonl")
    result = executor(registry, specs, world, audit=audit).run(safe_delete())
    assert result.status == "completed", result.reason
    assert world.deleted == {PATH}
    assert result.divergences == []
    assert len(result.outputs) == 4
    assert "untrusted-data" in describe_outputs(result.outputs)
    actions = [json.loads(line)["action"] for line in audit.path.read_text().splitlines()]
    assert actions[0] == "run.started"
    assert actions[-1] == "run.finished"
    assert verify_file(audit.path).ok


def test_denied_approval_skips_the_delete(registry: EffectRegistry, specs: SpecSet) -> None:
    world = FileMaintenanceWorld(dirs={PATH: 1}, reviewer=deny)
    result = executor(registry, specs, world).run(safe_delete())
    assert result.status == "completed"
    assert world.deleted == set()
    assert result.executed == [f"request_approval(path={PATH})"]


def test_a_silent_snapshot_failure_is_a_divergence_and_the_delete_is_blocked(
    registry: EffectRegistry, specs: SpecSet, tmp_path: Path
) -> None:
    world = FileMaintenanceWorld(dirs={PATH: 1}, reviewer=approve, faults={"snapshot_dir": "noop"})
    audit = AuditLog(tmp_path / "audit.jsonl")
    result = executor(registry, specs, world, audit=audit).run(safe_delete())
    assert result.status == "blocked"
    assert result.blocked is not None
    assert result.blocked.spec == "backup_before_delete"
    assert result.divergences == [f"snapshot_dir(path={PATH})"]
    assert world.deleted == set()
    assert "monitor.blocked" in audit.path.read_text()


def test_unreported_state_changes_are_caught_by_the_sensor(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    subject = plan(
        "compact twice",
        call("acquire_lock", path=PATH),
        call("compact_dir", path=PATH),
        call("compact_dir", path=PATH),
        call("release_lock", path=PATH),
    )
    faults = {"compact_dir": "lock_lost"}
    with_sensor = executor(registry, specs, FileMaintenanceWorld(faults=faults)).run(subject)
    assert with_sensor.status == "blocked"
    # Without re-sensing, the lost lease is invisible and the second compaction runs.
    blind = executor(registry, specs, FileMaintenanceWorld(faults=faults), sensor=False)
    assert blind.run(subject).status == "completed"


def test_effects_observed_after_a_step_can_still_violate(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    world = FileMaintenanceWorld(faults={"release_lock": "noop"})
    subject = plan("x", call("acquire_lock", path=PATH), call("release_lock", path=PATH))
    result = executor(registry, specs, world).run(subject)
    assert result.status == "violated"
    assert [v.kind for v in result.violations] == ["at_end"]


def test_collections_are_bounded_distinct_and_fresh(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    subject = plan(
        "x",
        call("archive_dir", path="/fixed"),
        each_of("d", "stale", 2, [call("archive_dir", path="$d")]),
    )
    cases = {
        "bound is": ["/a", "/b", "/c"],
        "duplicates": ["/a", "/a"],
        "aliases": ["/fixed"],
    }
    for reason, items in cases.items():
        world = FileMaintenanceWorld(collections={"stale": items})
        result = executor(registry, specs, world).run(subject)
        assert result.status == "aborted"
        assert reason in result.reason
    ok = executor(registry, specs, FileMaintenanceWorld(collections={"stale": ["/a"]}))
    assert ok.run(subject).executed == ["archive_dir(path=/fixed)", "archive_dir(path=/a)"]


def test_tool_failures_and_undeclared_calls_abort_the_run(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    world = FileMaintenanceWorld(dirs={PATH: 1})
    world.deleted.add(PATH)
    failed = executor(registry, specs, world).run(plan("x", call("write_file", path=PATH)))
    assert failed.status == "aborted"
    assert "failed" in failed.reason
    unknown = executor(registry, specs, world).run(plan("x", call("rm_rf", path=PATH)))
    assert unknown.status == "aborted"
    bad_args = executor(registry, specs, world).run(plan("x", call("write_file", target=PATH)))
    assert bad_args.status == "aborted"


def test_branches_and_loops_follow_the_observed_world(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    subject = plan(
        "x",
        each("d", ["/a", "/b"], [call("archive_dir", path="$d")]),
        if_opaque("big", [call("archive_dir", path="/big")], [call("archive_dir", path="/small")]),
    )
    world = FileMaintenanceWorld(opaque={"big": True})
    result = executor(registry, specs, world).run(subject)
    assert result.executed[-1] == "archive_dir(path=/big)"
    assert world.archive == {"/a", "/b", "/big"}


# --- override -----------------------------------------------------------------------------------


def test_an_approved_override_waives_exactly_the_listed_instances_once(
    registry: EffectRegistry, specs: SpecSet, checker: ModelChecker, tmp_path: Path
) -> None:
    unsafe = plan(
        "delete without backup",
        call("request_approval", path=PATH),
        if_prop("approved", PATH, [call("archive_dir", path=PATH), call("delete_dir", path=PATH)]),
    )
    result = checker.check(unsafe)
    audit = AuditLog(tmp_path / "audit.jsonl")
    gate = OverrideGate(ApprovalStore(tmp_path / "approvals.json"), audit=audit)
    with pytest.raises(OverrideError, match="justification"):
        gate.request(unsafe, result, justification="too short", requested_by="agent:a")
    with pytest.raises(OverrideError, match="only waive"):
        gate.request(
            unsafe,
            result,
            justification="the directory is a disposable cache",
            requested_by="agent:a",
            waive=["release_all_locks[r=/data/x]"],
        )
    record = gate.request(
        unsafe,
        result,
        justification="the directory is a disposable cache rebuilt nightly",
        requested_by="agent:a",
    )
    with pytest.raises(OverrideError, match="not approved"):
        gate.consume(unsafe, record.id, principal="agent:a")
    with pytest.raises(OverrideError, match="separation of duties"):
        gate.decide(record.id, approver="agent:a", approve=True)
    gate.decide(record.id, approver="human:lead", approve=True, reason="accepted risk")
    with pytest.raises(OverrideError, match="different plan"):
        gate.consume(plan("other", call("archive_dir", path="/z")), record.id, principal="agent:a")
    with pytest.raises(OverrideError, match="not yours"):
        gate.consume(unsafe, record.id, principal="agent:b")
    waived = gate.consume(unsafe, record.id, principal="agent:a")
    assert waived == frozenset({"backup_before_delete[r=/data/x]"})
    with pytest.raises(OverrideError):
        gate.consume(unsafe, record.id, principal="agent:a")
    with pytest.raises(OverrideError, match="unknown"):
        gate.consume(unsafe, "0" * 32, principal="agent:a")

    world = FileMaintenanceWorld(dirs={PATH: 1}, reviewer=approve)
    run = executor(registry, specs, world, audit=audit, waived=waived).run(unsafe)
    assert run.status == "completed"
    assert [v.spec for v in run.waived] == ["backup_before_delete"]
    text = audit.path.read_text()
    for action in ("override.requested", "override.decided", "override.consumed", "monitor.waived"):
        assert action in text
    assert "disposable cache rebuilt nightly" in text


def test_overrides_are_refused_for_safe_or_mismatched_results(
    checker: ModelChecker, tmp_path: Path
) -> None:
    gate = OverrideGate(ApprovalStore())
    safe = plan("x", call("archive_dir", path=PATH))
    with pytest.raises(OverrideError, match="nothing to override"):
        gate.request(safe, checker.check(safe), justification="x" * 30, requested_by="a")
    unsafe = plan("x", call("acquire_lock", path=PATH))
    with pytest.raises(OverrideError, match="different plan"):
        gate.request(safe, checker.check(unsafe), justification="x" * 30, requested_by="a")
    with pytest.raises(OverrideError):
        gate.decide("f" * 32, approver="b", approve=True)
    del tmp_path


# --- world --------------------------------------------------------------------------------------


def test_tools_only_get_the_operations_they_were_granted() -> None:
    world = FileMaintenanceWorld(dirs={PATH: 1})
    snapshot_only = ScopedWorld(world, frozenset({"read_dir", "write_backup"}))
    snapshot_only.snapshot(PATH)
    for action in (
        lambda: snapshot_only.delete(PATH),
        lambda: snapshot_only.write(PATH),
        lambda: snapshot_only.lock(PATH, held=True),
        lambda: snapshot_only.request_approval(PATH),
    ):
        with pytest.raises(PermissionError):
            action()
    deleter = ScopedWorld(world, frozenset({"delete_dir"}))
    with pytest.raises(PermissionError, match="approval"):
        deleter.delete(PATH)  # the world itself refuses deletion without a granted approval


def test_approvals_are_bound_to_the_exact_destructive_call() -> None:
    store = ApprovalStore()
    world = FileMaintenanceWorld(dirs={"/a": 1, "/b": 1}, approvals=store)
    approvals = ScopedWorld(world, frozenset({"approvals"}))
    assert approvals.request_approval("/a") is False  # pending, nobody reviewed it
    (pending,) = store.pending()
    assert pending.args == {"path": "/a"}
    store.decide(pending.id, approver="human:ops", approve=True)
    assert approvals.request_approval("/b") is False  # approval for /a does not cover /b
    assert approvals.request_approval("/a") is True
    assert world.granted == {"/a"}
    with pytest.raises(ApprovalError):
        store.decide(pending.id, approver="human:ops", approve=True)
    with pytest.raises(KeyError):
        world.sense(GroundProp("unknown"))
