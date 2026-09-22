"""The checking engine behind the CLI, the HTTP API, the demo and the benchmarks.

It owns the verified registry, the compiled spec set, the model checker, the audit log and the
approval store, and wraps each operation in the observability the service promises:

- metrics: ``check_latency_seconds``, ``states_explored``, ``violations_total{spec}``,
  ``repairs_succeeded_total``, ``monitor_blocks_total``;
- spans: ``parse``, ``check`` and ``repair``;
- hash-chained audit entries for plans, verdicts, counterexamples, repairs and overrides.

Registry provenance: with ``APP_REGISTRY_PATH`` set, the signed bundle is loaded and verified
against ``APP_REGISTRY_PUBLIC_KEYS`` and the optional ``APP_REGISTRY_SHA256`` pin. Without it, the
built-in catalog is signed with an ephemeral key at start-up and verified through the same code
path; that is fine for development, and readiness reports it as not ready in ``prod``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from prometheus_client import CollectorRegistry, Counter, Histogram
from pydantic import JsonValue

from plan_model_checker.agentic.approval import ApprovalStore
from plan_model_checker.agentic.audit import AuditLog
from plan_model_checker.agentic.budget import BudgetLedger, BudgetLimits
from plan_model_checker.agentic.providers.anthropic import AnthropicProvider
from plan_model_checker.agentic.providers.base import LLMProvider
from plan_model_checker.agentic.providers.fake import FakeProvider
from plan_model_checker.catalog import PROTOCOL_SPECS, file_maintenance_registry_model
from plan_model_checker.checker import CheckLimits, CheckResult, ModelChecker
from plan_model_checker.core.config import Settings
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.explain import explain
from plan_model_checker.ir import Plan, PlanLimits
from plan_model_checker.override import OverrideGate
from plan_model_checker.planner import (
    PLANNER_ROLE,
    REPAIRER_ROLE,
    RepairLoop,
    RepairOutcome,
    ScriptedRepairPolicy,
)
from plan_model_checker.signing import (
    generate_signing_key,
    key_id_of,
    load_signed_registry,
    public_key_from_hex,
    sign_registry,
    verify_signed_registry,
)
from plan_model_checker.spec.specs import SpecSet

__all__ = [
    "CheckerMetrics",
    "Engine",
    "create_checker_metrics",
    "live_provider",
    "load_registry",
]


@dataclass(frozen=True, slots=True)
class CheckerMetrics:
    check_latency_seconds: Histogram
    states_explored: Histogram
    violations_total: Counter
    repairs_succeeded_total: Counter
    monitor_blocks_total: Counter


def create_checker_metrics(registry: CollectorRegistry) -> CheckerMetrics:
    return CheckerMetrics(
        check_latency_seconds=Histogram(
            "check_latency_seconds",
            "Wall time of one plan check.",
            registry=registry,
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
        ),
        states_explored=Histogram(
            "states_explored",
            "States popped by the search during one plan check.",
            registry=registry,
            buckets=(10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 250_000),
        ),
        violations_total=Counter(
            "violations_total",
            "Violated specs found by static checks.",
            ["spec"],
            registry=registry,
        ),
        repairs_succeeded_total=Counter(
            "repairs_succeeded_total",
            "Violating plans that the repair loop turned into verified plans.",
            registry=registry,
        ),
        monitor_blocks_total=Counter(
            "monitor_blocks_total",
            "Tool calls the runtime monitor refused to run.",
            registry=registry,
        ),
    )


def load_registry(settings: Settings) -> tuple[EffectRegistry, Literal["configured", "builtin"]]:
    """The verified effect registry and where it came from. Raises RegistryIntegrityError."""
    if settings.registry_path is not None:
        keys = {}
        for hex_key in settings.registry_public_keys:
            key = public_key_from_hex(hex_key)
            keys[key_id_of(key)] = key
        registry = load_signed_registry(
            settings.registry_path, keys, pinned_sha256=settings.registry_sha256
        )
        return registry, "configured"
    signing_key = generate_signing_key()
    bundle = sign_registry(file_maintenance_registry_model(), signing_key)
    public = signing_key.public_key()
    registry = verify_signed_registry(
        bundle, {key_id_of(public): public}, pinned_sha256=settings.registry_sha256
    )
    return registry, "builtin"


def live_provider(settings: Settings) -> AnthropicProvider:
    """Claude with the two roles the design calls for: Sonnet plans, Opus repairs."""
    return AnthropicProvider.from_settings(
        settings,
        role_models={
            PLANNER_ROLE: settings.llm_planner_model,
            REPAIRER_ROLE: settings.llm_repair_model,
        },
    )


def _load_specs(settings: Settings, registry: EffectRegistry) -> SpecSet:
    text = (
        settings.specs_path.read_text(encoding="utf-8")
        if settings.specs_path is not None
        else PROTOCOL_SPECS
    )
    return SpecSet.parse(text, registry)


@dataclass
class Engine:
    settings: Settings
    registry: EffectRegistry
    registry_source: Literal["configured", "builtin"]
    specs: SpecSet
    checker: ModelChecker
    metrics: CheckerMetrics
    tracer: Tracer
    audit: AuditLog
    approvals: ApprovalStore
    overrides: OverrideGate
    budget: BudgetLedger
    provider: LLMProvider | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        metrics_registry: CollectorRegistry,
        tracer_provider: TracerProvider,
        provider: LLMProvider | None = None,
    ) -> Engine:
        tracer = tracer_provider.get_tracer("plan_model_checker.engine")
        with tracer.start_as_current_span("parse"):
            registry, source = load_registry(settings)
            specs = _load_specs(settings, registry)
        limits = CheckLimits(
            plan=PlanLimits(
                max_nodes=settings.plan_max_nodes,
                max_depth=settings.plan_max_depth,
                max_loop_bound=settings.plan_max_loop_bound,
            ),
            max_states=settings.check_max_states,
        )
        data_dir = Path(settings.data_dir)
        audit = AuditLog(data_dir / "audit.jsonl")
        approvals = ApprovalStore(data_dir / "approvals.json")
        budget = BudgetLedger(
            BudgetLimits.of(
                steps=settings.budget_max_steps,
                tokens=settings.budget_max_tokens,
                usd=settings.budget_max_usd,
            ),
            data_dir / "budget.json",
        )
        return cls(
            settings=settings,
            registry=registry,
            registry_source=source,
            specs=specs,
            checker=ModelChecker(registry, specs, limits),
            metrics=create_checker_metrics(metrics_registry),
            tracer=tracer,
            audit=audit,
            approvals=approvals,
            overrides=OverrideGate(approvals, audit=audit),
            budget=budget,
            provider=provider,
        )

    # --- readiness ------------------------------------------------------------------------------

    def specs_compiled(self) -> bool:
        return bool(self.specs.specs)

    def registry_verified(self) -> bool:
        """The registry passed signature (and pin) verification when the engine was built.

        In ``prod`` only an operator-configured bundle counts; the built-in catalog signed with
        an ephemeral key is a development convenience.
        """
        return self.settings.environment != "prod" or self.registry_source == "configured"

    # --- operations -----------------------------------------------------------------------------

    def check(self, plan: Plan, spec_names: list[str] | None = None, *, actor: str) -> CheckResult:
        with self.tracer.start_as_current_span("check") as span:
            result = self.checker.check(plan, spec_names)
            span.set_attribute("check.verdict", result.verdict)
            span.set_attribute("check.states_explored", result.states_explored)
        self.observe_check(result)
        self.audit.append(
            actor,
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

    def observe_check(self, result: CheckResult) -> None:
        self.metrics.check_latency_seconds.observe(result.seconds)
        self.metrics.states_explored.observe(result.states_explored)
        for spec in result.violated_specs():
            self.metrics.violations_total.labels(spec).inc()

    def hints(self, plan: Plan, result: CheckResult) -> list[JsonValue]:
        found: list[JsonValue] = []
        for violation in result.violations:
            hint = explain(plan, self.registry, violation.counterexample)
            found.append(hint.to_json() if hint is not None else None)
        return found

    def repair_provider(self) -> LLMProvider:
        if self.provider is not None:
            return self.provider
        if self.settings.repairer == "anthropic":
            return live_provider(self.settings)
        return FakeProvider(ScriptedRepairPolicy())

    def repair(self, task: str, plan: Plan, *, actor: str) -> RepairOutcome:
        loop = RepairLoop(
            checker=self.checker,
            provider=self.repair_provider(),
            budget=self.budget,
            audit=self.audit,
            max_attempts=self.settings.repair_max_attempts,
            actor=actor,
            on_check=self.observe_check,
        )
        with self.tracer.start_as_current_span("repair") as span:
            outcome = loop.run(task, plan)
            span.set_attribute("repair.status", outcome.status)
            span.set_attribute("repair.attempts", len(outcome.attempts))
        if outcome.repaired:
            self.metrics.repairs_succeeded_total.inc()
        return outcome
