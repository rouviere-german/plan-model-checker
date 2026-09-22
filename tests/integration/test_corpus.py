"""The labeled corpus, the scripted repair loop and the runtime divergence scenarios."""

from __future__ import annotations

import pytest

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStore
from plan_model_checker.agentic.budget import BudgetLedger, BudgetLimits
from plan_model_checker.agentic.providers.fake import FakeProvider
from plan_model_checker.checker import ModelChecker
from plan_model_checker.corpus import (
    CorpusCase,
    HardCase,
    build_corpus,
    build_divergence_scenarios,
    build_hard_corpus,
)
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.oracle import oracle_check, replay_counterexample, violates
from plan_model_checker.planner import RepairLoop, ScriptedRepairPolicy, missing_calls
from plan_model_checker.runtime import PlanExecutor, RunResult, ToolGateway
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


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_a_tool_that_lies_about_the_fault_is_stopped_no_later(
    scenario: object, registry: EffectRegistry, specs: SpecSet
) -> None:
    """The same faults, but the faulty tool reports its declared (successful) effects."""
    from plan_model_checker.corpus import DivergenceScenario

    assert isinstance(scenario, DivergenceScenario)

    def run(*, lying: bool) -> tuple[RunResult, FileMaintenanceWorld]:
        world = FileMaintenanceWorld(
            dirs=dict(scenario.dirs),
            faults=dict(scenario.faults),
            collections={k: list(v) for k, v in scenario.collections.items()},
            opaque=dict(scenario.opaque),
            reviewer=_approve,
        )
        misreporting = scenario.faulted_tools if lying else frozenset()
        tools = build_tools(world, registry, misreporting=misreporting)
        result = PlanExecutor(
            registry,
            specs,
            ToolGateway(registry, tools),
            opaque=world.decide,
            collections=world.collection,
            sensor=world.sense,
        ).run(scenario.plan)
        return result, world

    honest, honest_world = run(lying=False)
    lying, world = run(lying=True)
    assert len(lying.executed) <= len(honest.executed), "the violating step never runs"
    assert world.deleted <= honest_world.deleted, "nothing is deleted that the honest run kept"
    if scenario.name.startswith("lock_lease_expires"):
        # compact_dir does not declare lock_held, so it has nothing to lie about; the sensor
        # still sees the lost lease and the next compaction is blocked.
        assert lying.status == "blocked"
        assert lying.misreports == []
    else:
        assert lying.status == "aborted"
        assert lying.misreports, "the sensor contradicts the tool's report"


HARD = build_hard_corpus()


def test_hard_corpus_shape() -> None:
    assert len(HARD) == 24
    assert len({case.name for case in HARD}) == 24
    assert {case.category for case in HARD} == {
        "multi_omission",
        "wrong_order",
        "wrong_args",
        "in_loop",
        "in_branch",
        "needs_restructure",
    }


@pytest.mark.parametrize("case", HARD, ids=[case.name for case in HARD])
def test_hard_cases_violate_what_they_were_built_to_violate(
    case: HardCase, checker: ModelChecker, registry: EffectRegistry, specs: SpecSet
) -> None:
    result = checker.check(case.plan)
    assert not result.ok
    assert case.violated <= set(result.violated_specs())
    oracle = oracle_check(case.plan, registry, specs)
    assert set(result.violated_specs()) == {name for name, bad in oracle.violated.items() if bad}


def test_scripted_repairs_of_hard_cases_verify_keep_every_step_or_need_an_override(
    checker: ModelChecker,
) -> None:
    statuses: dict[str, list[str]] = {}
    for case in HARD:
        loop = RepairLoop(
            checker,
            FakeProvider(ScriptedRepairPolicy()),
            BudgetLedger(BudgetLimits.of(steps=10, tokens=10_000_000, usd=1.0)),
        )
        outcome = loop.run(case.plan.goal, case.plan)
        statuses.setdefault(case.category, []).append(outcome.status)
        assert outcome.status in {"verified", "needs_override"}
        assert missing_calls(case.plan, outcome.plan) == []
        if outcome.status == "verified":
            assert checker.check(outcome.plan).ok
    # Moving a delete under an approval guard is not an insertion: the scripted repairer
    # cannot do it, and the loop hands these plans to a human.
    assert statuses["needs_restructure"] == ["needs_override"] * 3
    assert "verified" in statuses["multi_omission"]
