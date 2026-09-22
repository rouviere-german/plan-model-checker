"""Human override of a counterexample, with a recorded justification.

When repair fails, the plan can still run only if a human accepts the specific violations. An
override request names the exact plan (by SHA-256), the exact spec instances it waives and a
written justification; all three are hashed into one approval record in the template's
:class:`~plan_model_checker.agentic.approval.ApprovalStore`, so:

- a different principal must approve it (separation of duties), and it expires;
- it is single-use and only the requesting principal can consume it;
- it cannot be moved to another plan, widened to other specs, or re-justified after approval,
  because any change alters the hash.

A consumed override yields an :class:`OverrideGrant`: the plan hash and the waived instance keys.
Only :class:`~plan_model_checker.runtime.PlanExecutor` consumes overrides, and it re-checks that
the grant belongs to the plan it is about to run and covers every violation the checker finds.
The runtime monitor still evaluates the waived instances and records each violation as *waived*
in the audit log; every other spec keeps blocking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStatus, ApprovalStore
from plan_model_checker.agentic.audit import AuditLog
from plan_model_checker.agentic.errors import ApprovalError
from plan_model_checker.agentic.providers.base import ToolUseBlock
from plan_model_checker.agentic.tools import Capability, ToolDefinition, ToolRegistry, ValidatedCall
from plan_model_checker.checker import CheckResult
from plan_model_checker.ir import Plan, plan_sha256

__all__ = ["OVERRIDE_TOOL", "OverrideArgs", "OverrideError", "OverrideGate", "OverrideGrant"]

OVERRIDE_TOOL = "override_counterexample"
MIN_JUSTIFICATION = 20


class OverrideError(Exception):
    """The override request or its use is invalid."""


@dataclass(frozen=True, slots=True)
class OverrideGrant:
    """A consumed override: which plan it was granted for and which instances it waives."""

    request_id: str
    plan_sha256: str
    waived: frozenset[str]


class OverrideArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    waived: list[str] = Field(min_length=1, max_length=50)
    justification: str = Field(min_length=MIN_JUSTIFICATION, max_length=2_000)


def _registry() -> ToolRegistry:
    registry = ToolRegistry({Capability.WRITE_DATA})
    registry.register(
        ToolDefinition(
            name=OVERRIDE_TOOL,
            description="Run a plan despite the listed counterexamples.",
            args_model=OverrideArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=lambda args: None,
        )
    )
    return registry


class OverrideGate:
    def __init__(self, store: ApprovalStore, *, audit: AuditLog | None = None) -> None:
        self._store = store
        self._audit = audit
        self._registry = _registry()

    def _log(self, actor: str, action: str, data: Mapping[str, JsonValue]) -> None:
        if self._audit is not None:
            self._audit.append(actor, action, dict(data))

    def _validated(self, arguments: Mapping[str, JsonValue]) -> ValidatedCall:
        call = ToolUseBlock(id="override", name=OVERRIDE_TOOL, arguments=dict(arguments))
        return self._registry.validate_call(call)

    def request(
        self,
        plan: Plan,
        result: CheckResult,
        *,
        justification: str,
        requested_by: str,
        waive: list[str] | None = None,
    ) -> ApprovalRecord:
        """Ask to run ``plan`` although ``result`` found violations."""
        if result.ok:
            raise OverrideError("the plan passed the check; there is nothing to override")
        if result.plan_sha256 != plan_sha256(plan):
            raise OverrideError("the check result belongs to a different plan")
        violated = set(result.violated_instances())
        waived = sorted(waive if waive is not None else violated)
        if not waived or not set(waived) <= violated:
            raise OverrideError("an override may only waive instances the check found violated")
        if len(justification.strip()) < MIN_JUSTIFICATION:
            raise OverrideError(
                f"a justification of at least {MIN_JUSTIFICATION} characters is required"
            )
        call = self._validated(
            {
                "plan_sha256": result.plan_sha256,
                "waived": list(waived),
                "justification": justification.strip(),
            }
        )
        record = self._store.request(call, requested_by=requested_by)
        self._log(
            requested_by,
            "override.requested",
            {
                "request_id": record.id,
                "plan_sha256": result.plan_sha256,
                "waived": list(waived),
                "justification": justification.strip(),
            },
        )
        return record

    def decide(
        self, request_id: str, *, approver: str, approve: bool, reason: str | None = None
    ) -> ApprovalRecord:
        try:
            record = self._store.decide(
                request_id, approver=approver, approve=approve, reason=reason
            )
        except ApprovalError as exc:
            raise OverrideError(str(exc)) from exc
        self._log(
            approver,
            "override.decided",
            {"request_id": request_id, "approved": approve, "reason": reason},
        )
        return record

    def consume(self, plan: Plan, request_id: str, *, principal: str) -> OverrideGrant:
        """Use up an approved override for exactly this plan; return what it waives."""
        try:
            record = self._store.get(request_id)
        except ApprovalError as exc:
            raise OverrideError(str(exc)) from exc
        if record.tool != OVERRIDE_TOOL:
            raise OverrideError("approval record is not an override")
        args = OverrideArgs.model_validate(record.args)
        if args.plan_sha256 != plan_sha256(plan):
            raise OverrideError("override was granted for a different plan")
        if record.status is not ApprovalStatus.APPROVED:
            raise OverrideError(f"override is {record.status.value}, not approved")
        call = self._validated(record.args)
        consumed = self._store.consume(call.call_hash, principal=principal)
        if consumed is None:
            raise OverrideError("override is expired, already used, or not yours to use")
        self._log(
            principal,
            "override.consumed",
            {
                "request_id": request_id,
                "plan_sha256": args.plan_sha256,
                "waived": [*args.waived],
            },
        )
        return OverrideGrant(
            request_id=request_id, plan_sha256=args.plan_sha256, waived=frozenset(args.waived)
        )
