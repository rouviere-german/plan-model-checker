"""The labeled corpus, the scripted repair loop and the runtime divergence scenarios."""

from __future__ import annotations

import pytest

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStore
from plan_model_checker.agentic.budget import BudgetLedger, BudgetLimits
from plan_model_checker.agentic.providers.fake import FakeProvider
from plan_model_checker.checker import ModelChecker
from plan_model_checker.corpus import CorpusCase, build_corpus, build_divergence_scenarios
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.oracle import oracle_check, replay_counterexample, violates
from plan_model_checker.planner import RepairLoop, ScriptedRepairPolicy
from plan_model_checker.runtime import PlanExecutor, ToolGateway
from plan_model_checker.spec.specs import SpecSet
from plan_model_checker.world import FileMaintenanceWorld, build_tools

pytestmark = pytest.mark.integration

CORPUS = build_corpus()
SCENARIOS = build_divergence_scenarios()


def test_corpus_shape() -> None:
    assert len(CORPUS) == 60
    assert sum(case.violating for case in CORPUS) == 30
    assert len({case.spec for case in CORPUS}) == 8
    assert len({case.name for case in CORPUS}) == 60
    assert len(SCENARIOS) == 30


@pytest.mark.parametrize("case", CORPUS, ids=[case.name for case in CORPUS])
def test_every_corpus_verdict_matches_its_label_and_the_oracle(
    case: CorpusCase, checker: ModelChecker, registry: EffectRegistry, specs: SpecSet
) -> None:
    result = checker.check(case.plan)
    assert result.verdict == case.expected
    if case.violating:
        assert case.spec in result.violated_specs()
    oracle = oracle_check(case.plan, registry, specs)
    assert set(result.violated_specs()) == {name for name, bad in oracle.violated.items() if bad}
    for violation in result.violations:
        cex = violation.counterexample
        assert violates(replay_counterexample(case.plan, registry, cex), cex.instance.source)
        assert cex.length == oracle.shortest[violation.spec]


def test_the_scripted_repairer_fixes_every_single_omission(checker: ModelChecker) -> None:
    fixed = []
    for case in (c for c in CORPUS if c.violating):
        loop = RepairLoop(
            checker,
            FakeProvider(ScriptedRepairPolicy()),
            BudgetLedger(BudgetLimits.of(steps=10, tokens=10_000_000, usd=1.0)),
        )
        outcome = loop.run(case.plan.goal, case.plan)
        if outcome.status == "verified" and len(outcome.attempts) <= 3:
            fixed.append(case.name)
    assert len(fixed) == 30


def _approve(record: ApprovalRecord, store: ApprovalStore) -> None:
    store.decide(record.id, approver="human:ops", approve=True)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_divergence_is_blocked_before_the_violating_step_runs(
    scenario: object, checker: ModelChecker, registry: EffectRegistry, specs: SpecSet
) -> None:
    from plan_model_checker.corpus import DivergenceScenario

    assert isinstance(scenario, DivergenceScenario)
    assert checker.check(scenario.plan).ok, "scenarios start from verified plans"
    world = FileMaintenanceWorld(
        dirs=dict(scenario.dirs),
        faults=dict(scenario.faults),
        collections={k: list(v) for k, v in scenario.collections.items()},
        opaque=dict(scenario.opaque),
        reviewer=_approve,
    )
    result = PlanExecutor(
        registry,
        specs,
        ToolGateway(registry, build_tools(world, registry)),
        opaque=world.decide,
        collections=world.collection,
        sensor=world.sense,
    ).run(scenario.plan)
    assert result.status == "blocked"
    assert result.blocked is not None
    assert result.blocked.spec == scenario.blocked_spec
    assert result.blocked.step == scenario.blocked_step
    assert result.divergences, "the fault shows up as a divergence from the declaration"
    # The blocked call would have been the next position: it never ran.
    assert result.blocked.position == len(result.executed) + 1
