"""Differential tests: the checker against the brute-force oracle on generated plans and specs.

``DIFF_EXAMPLES`` sets the number of hypothesis examples (the CI differential job uses 5,000).
"""

from __future__ import annotations

import dataclasses
import os
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from plan_model_checker.checker import (
    Counterexample,
    Decision,
    GroundCall,
    ModelChecker,
    TraceStep,
)
from plan_model_checker.corpus import call, if_opaque, plan
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.gen import diff_registry, random_plan, random_specs
from plan_model_checker.oracle import (
    OracleBudgetExceededError,
    ReplayError,
    enumerate_traces,
    oracle_check,
    plan_universe,
    replay_counterexample,
    violates,
)
from plan_model_checker.spec.ast import (
    And,
    Atom,
    Bool,
    Formula,
    Iff,
    Implies,
    Or,
    Since,
    WeakYesterday,
    Yesterday,
)
from plan_model_checker.spec.specs import SpecSet

pytestmark = [pytest.mark.unit, pytest.mark.differential]

REGISTRY = diff_registry()
EXAMPLES = int(os.environ.get("DIFF_EXAMPLES", "150"))


@settings(max_examples=EXAMPLES)
@given(st.randoms(use_true_random=False))
def test_checker_verdicts_equal_the_brute_force_oracle(rng: random.Random) -> None:
    subject = random_plan(rng)
    spec_set = random_specs(rng, REGISTRY, count=2)
    result = ModelChecker(REGISTRY, spec_set).check(subject)
    expected = oracle_check(subject, REGISTRY, spec_set)
    assert set(result.violated_specs()) == {k for k, v in expected.violated.items() if v}
    for violation in result.violations:
        cex = violation.counterexample
        trace = replay_counterexample(subject, REGISTRY, cex)
        assert violates(trace, cex.instance.source), "counterexample must be a real violation"
        assert cex.length == expected.shortest[violation.spec], "counterexample must be shortest"


def test_the_oracle_enumerates_every_trace(registry: EffectRegistry) -> None:
    subject = plan(
        "x",
        call("request_approval", path="/a"),
        if_opaque("o", [call("archive_dir", path="/a")], [call("snapshot_dir", path="/a")]),
    )
    traces = enumerate_traces(subject, registry)
    assert len(traces) == 4  # 2 approval outcomes x 2 branches
    assert plan_universe(subject, registry) == {"path": ("/a",)}
    with pytest.raises(OracleBudgetExceededError):
        enumerate_traces(subject, registry, max_traces=1)


def test_replay_rejects_counterexamples_that_the_plan_cannot_produce(
    registry: EffectRegistry, specs: SpecSet
) -> None:
    subject = plan(
        "x",
        call("acquire_lock", path="/a"),
        if_opaque("o", [call("compact_dir", path="/a")]),
    )
    cex = ModelChecker(registry, specs).check(subject).violations[0].counterexample
    assert violates(replay_counterexample(subject, registry, cex), cex.instance.source)

    def altered(**changes: object) -> Counterexample:
        return dataclasses.replace(cex, **changes)  # type: ignore[arg-type]

    wrong_tool = TraceStep(
        call=GroundCall("release_lock", (("path", "/a"),), "steps/0"), outcome="ok"
    )
    wrong_outcome = TraceStep(call=cex.steps[0].call, outcome="exploded")
    for bad in (
        altered(steps=(wrong_tool, *cex.steps[1:])),
        altered(steps=(wrong_outcome, *cex.steps[1:])),
        altered(decisions=()),
        altered(decisions=(Decision("steps/9", "then"),)),
        altered(steps=(*cex.steps, cex.steps[0])),
    ):
        with pytest.raises(ReplayError):
            replay_counterexample(subject, registry, bad)


def _weaken(formula: Formula) -> Formula:
    """A seeded bug: every Y (false at position 0) becomes Z (true at position 0)."""
    if isinstance(formula, Yesterday):
        return WeakYesterday(_weaken(formula.arg))
    if isinstance(formula, Atom | Bool):
        return formula
    if isinstance(formula, And | Or | Implies | Iff | Since):
        return type(formula)(_weaken(formula.left), _weaken(formula.right))
    return type(formula)(_weaken(formula.arg))


def test_the_oracle_catches_a_seeded_monitor_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    from plan_model_checker.spec.monitor import compile_monitor

    monkeypatch.setattr(
        "plan_model_checker.checker.compile_monitor", lambda f: compile_monitor(_weaken(f))
    )
    mismatches = 0
    for seed in range(300):
        rng = random.Random(seed)
        subject = random_plan(rng)
        spec_set = random_specs(rng, REGISTRY, count=2)
        got = set(ModelChecker(REGISTRY, spec_set).check(subject).violated_specs())
        expected = oracle_check(subject, REGISTRY, spec_set)
        mismatches += got != {k for k, v in expected.violated.items() if v}
    assert mismatches > 0
