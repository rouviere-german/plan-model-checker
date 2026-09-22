"""LLM planning and counterexample-guided repair, behind the provider protocol.

Two roles share one provider: ``planner`` drafts a plan from the task (claude-sonnet-5 when live)
and ``repairer`` revises a rejected plan given the checker's counterexamples (claude-opus-5 when
live). Both answer in the flat wire format through structured output, and both are charged to one
:class:`~plan_model_checker.agentic.budget.BudgetLedger` (steps, tokens and USD).

Prompt hygiene: the task is operator text; specs and tool declarations come from the server's own
compiled copies; everything a model or a tool produced (the rejected plan, counterexamples that
quote it, runtime tool outputs) goes into untrusted-data envelopes. Nothing in a prompt or a
response can change the specs or the registry: the only thing read back is a plan, which is
validated against a closed schema and then checked again.

Offline, :class:`ScriptedPlannerPolicy` replays plans from the corpus and
:class:`ScriptedRepairPolicy` applies the first repair hint of the first violation, i.e. inserts
the step the counterexample names. That exercises the loop, not LLM repair quality.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import JsonValue

from plan_model_checker.agentic.audit import AuditLog
from plan_model_checker.agentic.budget import BudgetLedger
from plan_model_checker.agentic.envelope import (
    UNTRUSTED_DATA_POLICY,
    UntrustedData,
    parse_envelopes,
)
from plan_model_checker.agentic.errors import BudgetExceededError, ProviderError
from plan_model_checker.agentic.providers.base import (
    ChatMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TextBlock,
)
from plan_model_checker.checker import CheckBudgetExceededError, CheckResult, ModelChecker
from plan_model_checker.core.canonical import canonical_json
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.explain import RepairHint, explain
from plan_model_checker.ir import Plan, PlanError, ToolCall, insert_node, plan_sha256
from plan_model_checker.spec.specs import SpecSet
from plan_model_checker.wire import plan_to_wire, wire_schema, wire_to_plan

__all__ = [
    "PLANNER_ROLE",
    "REPAIRER_ROLE",
    "REPAIR_CONTEXT_SOURCE",
    "Planner",
    "RepairAttempt",
    "RepairLoop",
    "RepairOutcome",
    "ScriptedPlannerPolicy",
    "ScriptedRepairPolicy",
    "catalogue_text",
    "hints_for",
]

PLANNER_ROLE = "planner"
REPAIRER_ROLE = "repairer"
REPAIR_CONTEXT_SOURCE = "checker:repair_context"

_PLAN_FORMAT = (
    "Answer with a plan in the flat JSON format of the output schema: every step is a node with "
    "an integer id; 'root' lists the top-level step ids in order; 'if' nodes list their 'then' and "
    "'otherwise' step ids; 'foreach' nodes list their 'body' ids and iterate either over 'items' "
    "or over a data-dependent 'collection' of at most 'max_iterations' items, binding 'var' "
    "(refer to it as '$var'). Conditions test a declared fluent ('prop' with 'args') or an opaque "
    "'label' when the outcome depends on data. Use only the tools listed, with exactly their "
    "parameters. Unused fields are null or empty lists."
)


def catalogue_text(registry: EffectRegistry, specs: SpecSet) -> str:
    """Server-side, trusted description of tools (with effects) and rules."""
    lines = ["Tools (name(params): event; outcomes -> effects):"]
    for name in registry.tool_names:
        tool = registry.tool(name)
        params = ", ".join(f"{p.name}: {p.sort}" for p in tool.params)
        outcomes = "; ".join(
            f"{o.name} -> "
            + (
                ", ".join(
                    f"{e.prop}[{','.join(e.args)}]={'true' if e.value else 'false'}"
                    for e in o.effects
                )
                or "no change"
            )
            for o in tool.outcomes
        )
        flag = " [destructive, needs " + tool.approval.prop + "]" if tool.approval else ""
        lines.append(f"- {name}({params}): {tool.event.prop}; {outcomes}{flag}")
    lines.append("Rules every plan must satisfy on every path (past-time temporal logic):")
    lines.extend(f"- {spec.name}: {spec.text}" for spec in specs.specs)
    return "\n".join(lines)


def _charge_failure(budget: BudgetLedger, exc: ProviderError) -> None:
    if exc.usage is not None:
        budget.record_usage(model=exc.model or "unknown", usage=exc.usage)


def _response_plan(response: LLMResponse) -> Plan:
    payload: object = response.parsed
    if payload is None:
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise PlanError("model answer is not JSON") from exc
    if not isinstance(payload, dict):
        raise PlanError("model answer is not a JSON object")
    return wire_to_plan(payload)


@dataclass
class Planner:
    provider: LLMProvider
    registry: EffectRegistry
    specs: SpecSet
    budget: BudgetLedger
    audit: AuditLog | None = None
    max_tokens: int = 8_000

    def request(self, task: str) -> LLMRequest:
        system = (
            "You plan file-maintenance work for an agent whose plans are model-checked before "
            f"they run.\n\n{catalogue_text(self.registry, self.specs)}\n\n{_PLAN_FORMAT}\n\n"
            f"{UNTRUSTED_DATA_POLICY}"
        )
        return LLMRequest(
            role=PLANNER_ROLE,
            system=system,
            messages=(ChatMessage.user(f"Task: {task}"),),
            output_schema=wire_schema(),
            max_tokens=self.max_tokens,
        )

    def plan(self, task: str) -> Plan:
        request = self.request(task)
        self.budget.reserve_step()
        try:
            response = self.provider.complete(request)
        except ProviderError as exc:
            _charge_failure(self.budget, exc)
            raise
        self.budget.record_usage(model=response.model, usage=response.usage)
        plan = _response_plan(response)
        if self.audit is not None:
            self.audit.append(
                "planner",
                "plan.proposed",
                {"model": response.model, "plan_sha256": plan_sha256(plan)},
            )
        return plan


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    attempt: int
    plan_sha256: str | None
    verdict: str
    violated: tuple[str, ...]
    error: str | None = None

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "attempt": self.attempt,
            "plan_sha256": self.plan_sha256,
            "verdict": self.verdict,
            "violated": list(self.violated),
            "error": self.error,
        }


@dataclass
class RepairOutcome:
    status: Literal["verified", "needs_override", "budget_exhausted"]
    plan: Plan
    result: CheckResult
    initial: CheckResult
    attempts: list[RepairAttempt] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return self.status == "verified" and not self.initial.ok

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan": self.plan.model_dump(mode="json"),
            "initial_violations": list(self.initial.violated_specs()),
            "result": self.result.to_json(),
            "attempts": [attempt.to_json() for attempt in self.attempts],
        }


def hints_for(plan: Plan, registry: EffectRegistry, result: CheckResult) -> list[RepairHint | None]:
    return [explain(plan, registry, violation.counterexample) for violation in result.violations]


def repair_context(plan: Plan, registry: EffectRegistry, result: CheckResult) -> UntrustedData:
    """The plan and its counterexamples, as data (they quote model-chosen names)."""
    hints = hints_for(plan, registry, result)
    payload = {
        "plan": plan.model_dump(mode="json"),
        "violations": [
            {
                "spec": violation.spec,
                "counterexample": violation.counterexample.to_json(),
                "hint": hint.to_json() if hint is not None else None,
                "hint_text": hint.describe() if hint is not None else None,
            }
            for violation, hint in zip(result.violations, hints, strict=True)
        ],
    }
    return UntrustedData(source=REPAIR_CONTEXT_SOURCE, content=canonical_json(payload))


@dataclass
class RepairLoop:
    """Check, and while the plan violates a spec, ask the repairer for a revision (bounded)."""

    checker: ModelChecker
    provider: LLMProvider
    budget: BudgetLedger
    audit: AuditLog | None = None
    max_attempts: int = 3
    max_tokens: int = 8_000
    actor: str = "repair-loop"
    on_check: Callable[[CheckResult], None] | None = None

    def _log(self, action: str, data: Mapping[str, JsonValue]) -> None:
        if self.audit is not None:
            self.audit.append(self.actor, action, dict(data))

    def request(
        self,
        task: str,
        plan: Plan,
        result: CheckResult,
        observations: Sequence[UntrustedData] = (),
    ) -> LLMRequest:
        registry = self.checker.registry
        system = (
            "You repair plans that a model checker rejected. Each counterexample is a shortest "
            "execution of the plan that breaks a rule; the hint names the step that is missing "
            "and where it goes. Return the whole corrected plan. Do not drop steps the task "
            f"needs.\n\n{catalogue_text(registry, self.checker.specs)}\n\n{_PLAN_FORMAT}\n\n"
            f"{UNTRUSTED_DATA_POLICY}"
        )
        blocks = [
            TextBlock(text=f"Task: {task}"),
            TextBlock(text=repair_context(plan, registry, result).render()),
        ]
        blocks.extend(TextBlock(text=item.render()) for item in observations)
        return LLMRequest(
            role=REPAIRER_ROLE,
            system=system,
            messages=(ChatMessage(role="user", content=tuple(blocks)),),
            output_schema=wire_schema(),
            max_tokens=self.max_tokens,
        )

    def _check(self, plan: Plan) -> CheckResult:
        result = self.checker.check(plan)
        if self.on_check is not None:
            self.on_check(result)
        self._log(
            "plan.checked",
            {
                "plan_sha256": result.plan_sha256,
                "verdict": result.verdict,
                "violated": list(result.violated_specs()),
                "states_explored": result.states_explored,
                "counterexamples": [v.counterexample.to_json() for v in result.violations],
            },
        )
        return result

    def run(
        self, task: str, plan: Plan, *, observations: Sequence[UntrustedData] = ()
    ) -> RepairOutcome:
        initial = self._check(plan)
        outcome = RepairOutcome(status="verified", plan=plan, result=initial, initial=initial)
        attempt = 0
        while not outcome.result.ok and attempt < self.max_attempts:
            attempt += 1
            request = self.request(task, outcome.plan, outcome.result, observations)
            try:
                self.budget.reserve_step()
                response = self.provider.complete(request)
                self.budget.record_usage(model=response.model, usage=response.usage)
            except BudgetExceededError as exc:
                outcome.attempts.append(
                    RepairAttempt(attempt, None, "error", (), error=f"budget: {exc.dimension}")
                )
                outcome.status = "budget_exhausted"
                break
            except ProviderError as exc:
                _charge_failure(self.budget, exc)
                outcome.attempts.append(
                    RepairAttempt(attempt, None, "error", (), type(exc).__name__)
                )
                if not exc.retryable:
                    break
                continue
            try:
                candidate = _response_plan(response)
                result = self._check(candidate)
            except (PlanError, CheckBudgetExceededError) as exc:
                outcome.attempts.append(
                    RepairAttempt(attempt, None, "rejected", (), str(exc)[:200])
                )
                continue
            outcome.plan, outcome.result = candidate, result
            outcome.attempts.append(
                RepairAttempt(attempt, result.plan_sha256, result.verdict, result.violated_specs())
            )
        if outcome.status != "budget_exhausted":
            outcome.status = "verified" if outcome.result.ok else "needs_override"
        self._log(
            "repair.finished",
            {
                "status": outcome.status,
                "attempts": len(outcome.attempts),
                "plan_sha256": outcome.result.plan_sha256,
            },
        )
        return outcome


class ScriptedPlannerPolicy:
    """Offline planner: returns prepared plans in order (as structured output)."""

    def __init__(self, plans: Sequence[Plan]) -> None:
        self._plans = tuple(plans)

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> LLMResponse:
        del request, rng
        plan = self._plans[turn % len(self._plans)]
        wire = plan_to_wire(plan).model_dump(mode="json")
        return LLMResponse(
            model="fake-scripted-planner",
            stop_reason="end_turn",
            text=canonical_json(wire),
            parsed=wire,
        )


class ScriptedRepairPolicy:
    """Offline repairer: insert the first suggested step of the first hint that has one."""

    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> LLMResponse:
        del rng, turn
        context = _find_context(request)
        plan = Plan.model_validate(context["plan"])
        for violation in context.get("violations", []):
            hint = violation.get("hint") if isinstance(violation, dict) else None
            if not isinstance(hint, dict) or not hint.get("suggestions"):
                continue
            suggestion = hint["suggestions"][0]
            node = {"kind": "call", "tool": suggestion["tool"], "args": suggestion["args"]}
            anchor = hint.get("anchor_path")
            plan = insert_node(
                plan,
                anchor if isinstance(anchor, str) else None,
                ToolCall.model_validate(node),
                after=hint.get("placement") == "after",
            )
            break
        wire = plan_to_wire(plan).model_dump(mode="json")
        return LLMResponse(
            model="fake-scripted-repairer",
            stop_reason="end_turn",
            text=canonical_json(wire),
            parsed=wire,
        )


def _find_context(request: LLMRequest) -> dict[str, Any]:
    for block in request.messages[-1].content:
        if not isinstance(block, TextBlock):
            continue
        for envelope in parse_envelopes(block.text):
            if envelope.source == REPAIR_CONTEXT_SOURCE:
                data = json.loads(envelope.content)
                if isinstance(data, dict):
                    return data
    raise ValueError("repair request has no repair context")
