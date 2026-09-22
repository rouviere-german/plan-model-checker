"""HTTP API: check, repair, monitor and override endpoints on the hardened app factory.

Every route needs an API key with the matching scope (``check``, ``repair``, ``monitor``,
``override``, ``approve``, ``audit``). Bodies are parsed from raw bytes with pydantic's JSON
parser (bounded nesting, closed schemas), size-limited and rate-limited by the core middleware,
and CPU-bound work runs in the thread pool so one expensive check cannot stall the event loop.
Checks are bounded by state, instance and size-assignment budgets, and monitor requests by an
instance budget and a work budget (monitor steps); exhausting any of them is a 422. Error
messages about a rejected plan are redacted and truncated.

Audit: checks, repairs, overrides and monitor decisions are appended to the hash-chained audit
log. ``GET /v1/audit/head`` (scope ``audit``) returns the verified head hash and entry count so an
operator can anchor it outside the service; a rewritten chain then no longer matches the anchor.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from plan_model_checker.agentic.audit import verify_file
from plan_model_checker.agentic.providers.base import LLMProvider
from plan_model_checker.checker import CheckBudgetExceededError, GroundCall
from plan_model_checker.core.app import create_app
from plan_model_checker.core.config import Settings, get_settings
from plan_model_checker.core.errors import AppError
from plan_model_checker.core.redaction import redact_text
from plan_model_checker.core.security import Principal, require_scope
from plan_model_checker.core.tracing import build_tracer_provider
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.engine import Engine
from plan_model_checker.ir import IDENT_PATTERN, Plan, PlanError, ResourceValue, ToolArg
from plan_model_checker.override import OverrideError
from plan_model_checker.props import GroundProp
from plan_model_checker.runtime import (
    EffectReportError,
    MonitorBudgetExceededError,
    MonitorViolation,
    RuntimeMonitor,
)
from plan_model_checker.spec.specs import InstanceBudgetExceededError, SpecError

__all__ = ["create_service_app"]


# Route dependencies: each resolves the caller's key and requires one scope. They live at module
# level so FastAPI can resolve the (postponed) annotations.
CheckScope = Annotated[Principal, Depends(require_scope("check"))]
RepairScope = Annotated[Principal, Depends(require_scope("repair"))]
MonitorScope = Annotated[Principal, Depends(require_scope("monitor"))]
OverrideScope = Annotated[Principal, Depends(require_scope("override"))]
ApproveScope = Annotated[Principal, Depends(require_scope("approve"))]
AuditScope = Annotated[Principal, Depends(require_scope("audit"))]

_EFFECT_ERRORS = {
    "unknown_proposition": "An effect names an undeclared proposition.",
    "invalid_effect": "Effects must set declared fluents.",
    "undeclared_effect": "An effect is not among the effects the tool declares.",
    "conflicting_effect": "An effect is reported with two different values.",
}


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckRequest(_Body):
    plan: Plan
    specs: list[Annotated[str, Field(pattern=IDENT_PATTERN)]] | None = Field(
        default=None, max_length=64
    )


class RepairRequest(_Body):
    task: str = Field(min_length=1, max_length=2_000)
    plan: Plan


class EffectReport(_Body):
    prop: str = Field(pattern=IDENT_PATTERN)
    args: list[ResourceValue] = Field(max_length=3)
    value: bool


class ObservedStep(_Body):
    tool: str = Field(pattern=IDENT_PATTERN)
    args: list[ToolArg] = Field(max_length=8)
    effects: list[EffectReport] = Field(default_factory=list, max_length=32)


class ProposedCall(_Body):
    tool: str = Field(pattern=IDENT_PATTERN)
    args: list[ToolArg] = Field(max_length=8)


class MonitorRequest(_Body):
    observed: list[ObservedStep] = Field(default_factory=list, max_length=500)
    proposed: ProposedCall | None = None
    finished: bool = False


class OverrideRequest(_Body):
    plan: Plan
    justification: str = Field(min_length=20, max_length=2_000)
    waive: list[str] | None = Field(default=None, max_length=50)


class DecisionRequest(_Body):
    approve: bool
    reason: str | None = Field(default=None, max_length=500)


async def _parse[ModelT: BaseModel](request: Request, model: type[ModelT]) -> ModelT:
    body = await request.body()
    try:
        return model.model_validate_json(body)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_input=False)) from exc


def _plan_error(exc: Exception, code: str = "invalid_plan") -> AppError:
    return AppError(422, code, redact_text(str(exc))[:300])


def _ground(registry: EffectRegistry, tool: str, args: list[ToolArg], path: str) -> GroundCall:
    if not registry.has_tool(tool):
        raise AppError(422, "unknown_tool", "The tool is not in the signed registry.")
    params = [param.name for param in registry.tool(tool).params]
    given = {arg.name: arg.value for arg in args}
    if set(given) != set(params) or any(value.startswith("$") for value in given.values()):
        raise AppError(422, "invalid_arguments", "Arguments do not match the tool declaration.")
    return GroundCall(tool=tool, args=tuple((name, given[name]) for name in params), path=path)


def _effects(step: ObservedStep) -> tuple[tuple[GroundProp, bool], ...]:
    """The reported effects; the monitor rejects any the tool's declaration does not allow."""
    return tuple(
        (GroundProp(report.prop, tuple(report.args)), report.value) for report in step.effects
    )


def _monitor(engine: Engine, body: MonitorRequest, actor: str) -> dict[str, Any]:
    """Replay the observed steps and judge the proposed call, within the monitor budgets.

    There is no sensor behind this endpoint: the monitor trusts the effects the client reports,
    within each tool's declared effects, and assumes the registry's initial values.
    """
    monitor = RuntimeMonitor(
        engine.registry,
        engine.specs,
        max_instances=engine.settings.monitor_max_instances,
        max_work=engine.settings.monitor_max_work,
    )
    allowed: bool | None = None
    proposed: str | None = None
    try:
        violations: list[MonitorViolation] = list(monitor.start())
        for index, step in enumerate(body.observed):
            call = _ground(engine.registry, step.tool, step.args, f"observed/{index}")
            violations.extend(monitor.commit(call, _effects(step)))
        if body.proposed is not None:
            call = _ground(engine.registry, body.proposed.tool, body.proposed.args, "proposed")
            proposed = call.label()
            blocking = monitor.precheck(call)
            allowed = not blocking
            violations.extend(blocking)
            if blocking:
                engine.metrics.monitor_blocks_total.inc()
        if body.finished:
            violations.extend(monitor.finish())
    except EffectReportError as exc:
        raise AppError(422, exc.code, _EFFECT_ERRORS[exc.code]) from exc
    except (InstanceBudgetExceededError, MonitorBudgetExceededError) as exc:
        raise AppError(
            422,
            "monitor_budget_exceeded",
            "The request needs more spec instances or monitor steps than the service allows.",
        ) from exc
    engine.audit.append(
        actor,
        "monitor.checked",
        {
            "observed_steps": len(body.observed),
            "proposed": proposed,
            "allowed": allowed,
            "finished": body.finished,
            "violations": [violation.instance for violation in violations],
            "work": monitor.work,
        },
    )
    return {
        "allowed": allowed,
        "position": monitor.position,
        "violations": [violation.to_json() for violation in violations],
    }


def _run_check(engine: Engine, body: CheckRequest, actor: str) -> dict[str, Any]:
    try:
        result = engine.check(body.plan, body.specs, actor=actor)
    except SpecError as exc:
        raise _plan_error(exc, "unknown_spec") from exc
    except (CheckBudgetExceededError, InstanceBudgetExceededError) as exc:
        raise _plan_error(exc, "check_budget_exceeded") from exc
    except PlanError as exc:
        raise _plan_error(exc) from exc
    payload = result.to_json()
    for violation, hint in zip(payload["violations"], engine.hints(body.plan, result), strict=True):
        violation["hint"] = hint
    payload["seconds"] = round(result.seconds, 6)
    return payload


def _run_repair(engine: Engine, body: RepairRequest, actor: str) -> dict[str, Any]:
    try:
        outcome = engine.repair(body.task, body.plan, actor=actor)
    except (CheckBudgetExceededError, InstanceBudgetExceededError) as exc:
        raise _plan_error(exc, "check_budget_exceeded") from exc
    except PlanError as exc:
        raise _plan_error(exc) from exc
    payload = outcome.to_json()
    payload["hints"] = engine.hints(outcome.plan, outcome.result)
    return payload


def _run_override(engine: Engine, body: OverrideRequest, actor: str) -> dict[str, Any]:
    try:
        result = engine.check(body.plan, actor=actor)
        record = engine.overrides.request(
            body.plan,
            result,
            justification=body.justification,
            requested_by=actor,
            waive=body.waive,
        )
    except (CheckBudgetExceededError, InstanceBudgetExceededError) as exc:
        raise _plan_error(exc, "check_budget_exceeded") from exc
    except PlanError as exc:
        raise _plan_error(exc) from exc
    except OverrideError as exc:
        raise AppError(409, "override_rejected", redact_text(str(exc))[:300]) from exc
    return {
        "request_id": record.id,
        "status": record.status.value,
        "plan_sha256": result.plan_sha256,
        "waived": record.args.get("waived"),
    }


def build_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/check")
    async def check(request: Request, principal: CheckScope) -> dict[str, Any]:
        body = await _parse(request, CheckRequest)
        return await run_in_threadpool(_run_check, engine, body, principal.name)

    @router.post("/repair")
    async def repair(request: Request, principal: RepairScope) -> dict[str, Any]:
        body = await _parse(request, RepairRequest)
        return await run_in_threadpool(_run_repair, engine, body, principal.name)

    @router.post("/monitor")
    async def monitor(request: Request, principal: MonitorScope) -> dict[str, Any]:
        body = await _parse(request, MonitorRequest)
        return await run_in_threadpool(_monitor, engine, body, principal.name)

    @router.post("/overrides")
    async def request_override(request: Request, principal: OverrideScope) -> dict[str, Any]:
        body = await _parse(request, OverrideRequest)
        return await run_in_threadpool(_run_override, engine, body, principal.name)

    @router.post("/overrides/{request_id}/decision")
    async def decide_override(
        request_id: Annotated[str, PathParam(pattern=r"^[0-9a-f]{32}$")],
        request: Request,
        principal: ApproveScope,
    ) -> dict[str, Any]:
        body = await _parse(request, DecisionRequest)
        try:
            record = engine.overrides.decide(
                request_id, approver=principal.name, approve=body.approve, reason=body.reason
            )
        except OverrideError as exc:
            raise AppError(409, "decision_rejected", redact_text(str(exc))[:300]) from exc
        return {"request_id": record.id, "status": record.status.value}

    @router.get("/audit/head")
    async def audit_head(principal: AuditScope) -> dict[str, Any]:
        """The verified head of the audit chain, for anchoring outside the service."""
        del principal
        verified = await run_in_threadpool(verify_file, engine.audit.path)
        if not verified.ok:
            raise AppError(500, "audit_integrity", "The audit log failed verification.")
        return {"head": verified.head, "entries": verified.entries}

    return router


def create_service_app(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    metrics_registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    cfg = settings if settings is not None else get_settings()
    registry = metrics_registry if metrics_registry is not None else CollectorRegistry()
    tracing = tracer_provider or build_tracer_provider(cfg.service_name, cfg.tracing_exporter)
    engine = Engine.from_settings(
        cfg, metrics_registry=registry, tracer_provider=tracing, provider=provider
    )
    app = create_app(
        cfg,
        readiness_checks={
            "specs_compiled": engine.specs_compiled,
            "registry_signature": engine.registry_verified,
        },
        registry=registry,
        tracer_provider=tracing,
        routers=[build_router(engine)],
    )
    app.state.engine = engine
    return app
