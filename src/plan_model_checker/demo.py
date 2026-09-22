"""Offline end-to-end demo (no API key, no network).

1. The effect registry is signed with an Ed25519 key and verified; a tampered copy is refused.
2. A scripted planner proposes a plan that deletes a directory without backing it up.
3. The checker rejects it with a shortest counterexample; the explainer names the missing step.
4. The scripted repairer inserts that step; the repaired plan verifies.
5. The repaired plan runs against mock tools: the directory is backed up, then deleted.
6. It runs again in a world where the snapshot tool silently fails: the runtime monitor blocks
   the deletion, and nothing is deleted.
7. And again with a snapshot tool that silently fails *and reports success*: the sensor reading
   contradicts the report, the run stops at the misreport, and nothing is deleted.
8. The hash-chained audit log verifies; a tampered copy does not.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStore
from plan_model_checker.agentic.audit import AuditLog, verify_file
from plan_model_checker.agentic.budget import BudgetLedger, BudgetLimits
from plan_model_checker.agentic.providers.fake import FakeProvider
from plan_model_checker.catalog import PROTOCOL_SPECS, file_maintenance_registry_model
from plan_model_checker.checker import ModelChecker
from plan_model_checker.corpus import call, if_prop, plan
from plan_model_checker.effects import EffectRegistry, EffectRegistryModel
from plan_model_checker.explain import explain
from plan_model_checker.ir import Plan, iter_calls
from plan_model_checker.planner import (
    Planner,
    RepairLoop,
    ScriptedPlannerPolicy,
    ScriptedRepairPolicy,
)
from plan_model_checker.runtime import PlanExecutor, RunResult, ToolGateway
from plan_model_checker.signing import (
    RegistryIntegrityError,
    generate_signing_key,
    key_id_of,
    sign_registry,
    verify_signed_registry,
)
from plan_model_checker.spec.specs import SpecSet
from plan_model_checker.world import FileMaintenanceWorld, build_tools

__all__ = ["DEMO_TARGET", "run_demo"]

DEMO_TARGET = "/srv/logs/2025"


def _unsafe_plan() -> Plan:
    return plan(
        "Free disk space by deleting last year's logs.",
        call("request_approval", path=DEMO_TARGET),
        if_prop(
            "approved",
            DEMO_TARGET,
            [call("archive_dir", path=DEMO_TARGET), call("delete_dir", path=DEMO_TARGET)],
        ),
    )


def _approve_all(record: ApprovalRecord, store: ApprovalStore) -> None:
    store.decide(record.id, approver="human:ops-oncall", approve=True, reason="demo approval")


def _execute(
    registry: EffectRegistry,
    specs: SpecSet,
    audit: AuditLog,
    subject: Plan,
    world: FileMaintenanceWorld,
    *,
    misreporting: frozenset[str] = frozenset(),
) -> RunResult:
    return PlanExecutor(
        registry,
        specs,
        ToolGateway(registry, build_tools(world, registry, misreporting=misreporting)),
        opaque=world.decide,
        collections=world.collection,
        audit=audit,
        principal=world.agent,
        sensor=world.sense,
    ).run(subject)


def run_demo(workdir: Path) -> dict[str, JsonValue]:  # noqa: PLR0915 - a linear narrative
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in ("audit.jsonl", "audit.tampered.jsonl", "budget.json", "approvals.json"):
        (workdir / stale).unlink(missing_ok=True)
    story: list[str] = []
    audit = AuditLog(workdir / "audit.jsonl")

    # 1. Signed, verified registry; a tampered bundle is refused.
    key = generate_signing_key()
    trusted = {key_id_of(key.public_key()): key.public_key()}
    bundle = sign_registry(file_maintenance_registry_model(), key)
    registry = verify_signed_registry(bundle, trusted)
    tampered_model = bundle.registry.model_dump(mode="json")
    for tool in tampered_model["tools"]:
        if tool["name"] == "delete_dir":
            tool["destructive"] = False
            tool["approval"] = None
    tampered = bundle.model_copy(
        update={"registry": EffectRegistryModel.model_validate(tampered_model)}
    )
    try:
        verify_signed_registry(tampered, trusted)
        tamper_refused = False
    except RegistryIntegrityError:
        tamper_refused = True
    story.append(
        f"registry {registry.sha256[:12]} verified; tampered copy refused={tamper_refused}"
    )
    audit.append("demo", "registry.verified", {"sha256": registry.sha256})

    specs = SpecSet.parse(PROTOCOL_SPECS, registry)
    checker = ModelChecker(registry, specs)
    budget = BudgetLedger(
        BudgetLimits.of(steps=10, tokens=200_000, usd=0.5), workdir / "budget.json"
    )

    # 2. The scripted planner proposes a plan that skips the backup.
    planner = Planner(
        provider=FakeProvider(ScriptedPlannerPolicy([_unsafe_plan()])),
        registry=registry,
        specs=specs,
        budget=budget,
        audit=audit,
    )
    task = f"Free disk space: delete {DEMO_TARGET} once a human approves."
    proposed = planner.plan(task)

    # 3. The checker rejects it with a shortest counterexample.
    initial = checker.check(proposed)
    violation = initial.violations[0]
    counterexample = violation.counterexample
    hint = explain(proposed, registry, counterexample)
    story.append(
        f"check: {initial.verdict} ({', '.join(initial.violated_specs())}); counterexample: "
        + " -> ".join(f"{s.call.label()}[{s.outcome}]" for s in counterexample.steps)
    )
    if hint is not None:
        story.append(f"hint: {hint.describe()}")

    # 4. Counterexample-guided repair (scripted repairer inserts the named step).
    loop = RepairLoop(
        checker=checker,
        provider=FakeProvider(ScriptedRepairPolicy()),
        budget=budget,
        audit=audit,
    )
    outcome = loop.run(task, proposed)
    repaired = outcome.plan
    story.append(
        f"repair: {outcome.status} after {len(outcome.attempts)} attempt(s); plan now runs "
        + ", ".join(node.tool for _, node in iter_calls(repaired.steps))
    )

    # 5. Healthy run: backup, then delete.
    healthy = FileMaintenanceWorld(dirs={DEMO_TARGET: 7}, reviewer=_approve_all)
    healthy_run = _execute(registry, specs, audit, repaired, healthy)
    story.append(
        f"run (healthy tools): {healthy_run.status}; deleted={sorted(healthy.deleted)}; "
        f"backups={sorted(healthy.backups)}"
    )

    # 6. Divergent run: the snapshot silently does nothing; the monitor blocks the delete.
    faults = {"snapshot_dir": "noop"}
    broken = FileMaintenanceWorld(dirs={DEMO_TARGET: 7}, reviewer=_approve_all, faults=faults)
    broken_run = _execute(registry, specs, audit, repaired, broken)
    blocked = broken_run.blocked
    story.append(
        f"run (snapshot silently fails): {broken_run.status}; blocked "
        f"{blocked.step if blocked else None} by {blocked.spec if blocked else None}; "
        f"deleted={sorted(broken.deleted)}"
    )

    # 7. Lying run: the snapshot does nothing but reports success; the sensor contradicts it.
    lying = FileMaintenanceWorld(dirs={DEMO_TARGET: 7}, reviewer=_approve_all, faults=faults)
    lying_run = _execute(
        registry, specs, audit, repaired, lying, misreporting=frozenset({"snapshot_dir"})
    )
    story.append(
        f"run (snapshot fails but reports success): {lying_run.status}; "
        f"misreport: {'; '.join(lying_run.misreports)}; deleted={sorted(lying.deleted)}"
    )

    # 8. Audit chain.
    verified = verify_file(audit.path, expect_head=audit.head)
    lines = audit.path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = lines[1].replace('"plan.proposed"', '"plan.approved"', 1)
    tampered_path = workdir / "audit.tampered.jsonl"
    tampered_path.write_text("".join(lines), encoding="utf-8", newline="\n")
    tampered_log = verify_file(tampered_path)
    story.append(
        f"audit: {verified.entries} entries verified={verified.ok}; tampered copy detected="
        f"{not tampered_log.ok}"
    )

    return {
        "story": list(story),
        "registry_tamper_refused": tamper_refused,
        "initial_verdict": initial.verdict,
        "initial_violations": list(initial.violated_specs()),
        "counterexample": counterexample.to_json(),
        "hint": hint.to_json() if hint is not None else None,
        "repair_status": outcome.status,
        "repair_attempts": len(outcome.attempts),
        "repaired_verdict": outcome.result.verdict,
        "repaired_plan": repaired.model_dump(mode="json"),
        "healthy_run": healthy_run.to_json(),
        "divergent_run": broken_run.to_json(),
        "deleted_in_divergent_run": [*sorted(broken.deleted)],
        "lying_run": lying_run.to_json(),
        "deleted_in_lying_run": [*sorted(lying.deleted)],
        "audit_ok": verified.ok,
        "audit_entries": verified.entries,
        "audit_head": verified.head,
        "tampered_audit_detected": not tampered_log.ok,
        "budget_state": budget.state.model_dump(mode="json"),
    }
