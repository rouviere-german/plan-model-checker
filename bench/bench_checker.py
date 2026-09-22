"""Checker, repair and runtime benchmarks on the labeled corpus, with CI thresholds.

Usage:
  uv run python bench/bench_checker.py          # run, write bench/results/checker.json
  uv run python bench/bench_checker.py --check  # also exit 1 if bench/thresholds_checker.json fails

Measures, on CPU and offline:
1. corpus verdicts: 60 labeled plans (30 safe, 30 single-omission violations, 8 specs);
2. check latency on corpus plans (p50/p95 over repeated runs; the design budget is p95 < 50 ms);
3. state count and latency against plan size, for two plan families, with and without skipping
   spec instances that cannot fail on the plan (vacuous instances);
4. repair: violating corpus plans fixed by the scripted repairer within 3 attempts, and the same
   loop on 24 harder plans (several omissions, wrong order or arguments, omissions in loops and
   branches, and plans that need restructuring), where it is expected to fail sometimes;
5. runtime: divergence scenarios blocked before the violating step runs, with honest tools, with
   tools that lie about the fault (they report their declared effects), and with lying tools but
   no sensor.

The repairer is scripted (it inserts the step the counterexample names), so the repair rates
test the loop, not an LLM's repair quality. No live-model repair runs are recorded.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStore
from plan_model_checker.agentic.budget import BudgetLedger, BudgetLimits
from plan_model_checker.agentic.providers.fake import FakeProvider
from plan_model_checker.catalog import file_maintenance_registry, protocol_specs
from plan_model_checker.checker import ModelChecker
from plan_model_checker.corpus import (
    DivergenceScenario,
    build_corpus,
    build_divergence_scenarios,
    build_hard_corpus,
    call,
    if_opaque,
    if_prop,
    plan,
)
from plan_model_checker.ir import Plan, unrolled_call_count
from plan_model_checker.planner import (
    RepairLoop,
    RepairOutcome,
    ScriptedRepairPolicy,
    missing_calls,
)
from plan_model_checker.runtime import PlanExecutor, RunResult, ToolGateway
from plan_model_checker.world import FileMaintenanceWorld, build_tools

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "checker.json"
THRESHOLDS = ROOT / "bench" / "thresholds_checker.json"
REPEATS = 20


def percentile(values: list[float], q: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def timed(function: Callable[[], object], repeats: int) -> list[float]:
    function()  # warm-up (compiles monitors, fills caches)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def corpus_section(checker: ModelChecker) -> dict[str, Any]:
    cases = build_corpus()
    correct = 0
    latencies: list[float] = []
    per_spec: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        result = checker.check(case.plan)
        ok = result.verdict == case.expected and (
            not case.violating or case.spec in result.violated_specs()
        )
        correct += ok
        bucket = per_spec.setdefault(case.spec, {"cases": 0, "correct": 0})
        bucket["cases"] += 1
        bucket["correct"] += ok
        samples = timed(functools.partial(checker.check, case.plan), REPEATS)
        latencies.extend(samples)
        rows.append(
            {
                "case": case.name,
                "expected": case.expected,
                "verdict": result.verdict,
                "states": result.states_explored,
                "p95_ms": round(percentile(samples, 95), 3),
            }
        )
    return {
        "cases": len(cases),
        "violating": sum(case.violating for case in cases),
        "correct": correct,
        "accuracy": correct / len(cases),
        "per_spec": per_spec,
        "check_p50_ms": round(percentile(latencies, 50), 3),
        "check_p95_ms": round(percentile(latencies, 95), 3),
        "check_max_ms": round(max(latencies), 3),
        "samples": len(latencies),
        "max_states": max(row["states"] for row in rows),
        "rows": rows,
    }


def delete_chain(n: int) -> Plan:
    steps: list[dict[str, Any]] = []
    for i in range(n):
        path = f"/srv/d{i}"
        steps += [
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
        ]
    return plan(f"delete {n} directories", *steps)


def diamond_chain(n: int) -> Plan:
    diamonds = [
        if_opaque(f"b{i}", [call("check_status", job="job")], [call("check_status", job="job")])
        for i in range(n)
    ]
    return plan(
        f"{n} diamonds", call("submit_job", job="job"), *diamonds, call("retry_job", job="job")
    )


def scaling_section(checker: ModelChecker) -> list[dict[str, Any]]:
    everything = ModelChecker(checker.registry, checker.specs, skip_vacuous=False)
    rows = []
    for family, build in (("delete_chain", delete_chain), ("diamond_chain", diamond_chain)):
        for n in (1, 2, 4, 8, 16, 32):
            subject = build(n)
            result = checker.check(subject)
            samples = timed(functools.partial(checker.check, subject), 5)
            baseline = everything.check(subject)
            baseline_samples = timed(functools.partial(everything.check, subject), 5)
            if baseline.verdict != result.verdict:
                raise SystemExit(f"skipping vacuous instances changed the verdict of {family}")
            rows.append(
                {
                    "family": family,
                    "n": n,
                    "calls": unrolled_call_count(subject.steps),
                    "paths": 2**n,
                    "instances": result.instances_checked,
                    "searched": result.instances_checked - result.instances_skipped,
                    "states": result.states_explored,
                    "verdict": result.verdict,
                    "p95_ms": round(percentile(samples, 95), 3),
                    "states_without_skip": baseline.states_explored,
                    "p95_ms_without_skip": round(percentile(baseline_samples, 95), 3),
                }
            )
    return rows


def _repair(checker: ModelChecker, subject: Plan) -> RepairOutcome:
    loop = RepairLoop(
        checker,
        FakeProvider(ScriptedRepairPolicy()),
        BudgetLedger(BudgetLimits.of(steps=10, tokens=10_000_000, usd=1.0)),
        max_attempts=3,
    )
    return loop.run(subject.goal, subject)


def repair_section(checker: ModelChecker) -> dict[str, Any]:
    violating = [case for case in build_corpus() if case.violating]
    fixed = 0
    dropped = 0
    attempts: dict[int, int] = {}
    for case in violating:
        outcome = _repair(checker, case.plan)
        dropped += outcome.dropped_step_rejections
        if outcome.status == "verified":
            fixed += 1
            attempts[len(outcome.attempts)] = attempts.get(len(outcome.attempts), 0) + 1
    return {
        "violating_plans": len(violating),
        "fixed_within_3_attempts": fixed,
        "success_rate": fixed / len(violating),
        "attempts_histogram": {str(k): v for k, v in sorted(attempts.items())},
        "dropped_step_rejections": dropped,
        "repairer": "scripted (inserts the step named by the counterexample)",
    }


def hard_repair_section(checker: ModelChecker) -> dict[str, Any]:
    """The same loop on plans whose fix is more than one inserted step."""
    cases = build_hard_corpus()
    flagged = 0
    verified = 0
    dropped = 0
    attempts: dict[int, int] = {}
    per_category: dict[str, dict[str, int]] = {}
    rows = []
    for case in cases:
        result = checker.check(case.plan)
        flagged += not result.ok and case.violated <= set(result.violated_specs())
        outcome = _repair(checker, case.plan)
        dropped += outcome.dropped_step_rejections
        ok = outcome.status == "verified" and not missing_calls(case.plan, outcome.plan)
        verified += ok
        if ok:
            attempts[len(outcome.attempts)] = attempts.get(len(outcome.attempts), 0) + 1
        bucket = per_category.setdefault(case.category, {"cases": 0, "verified": 0})
        bucket["cases"] += 1
        bucket["verified"] += ok
        rows.append(
            {
                "case": case.name,
                "category": case.category,
                "violated": sorted(result.violated_specs()),
                "repair": outcome.status,
                "attempts": len(outcome.attempts),
            }
        )
    return {
        "cases": len(cases),
        "flagged_by_checker": flagged,
        "flagged_rate": flagged / len(cases),
        "verified_within_3_attempts": verified,
        "needs_override": len(cases) - verified,
        "failure_rate": round((len(cases) - verified) / len(cases), 4),
        "per_category": per_category,
        "attempts_histogram": {str(k): v for k, v in sorted(attempts.items())},
        "dropped_step_rejections": dropped,
        "rows": rows,
    }


def _approve(record: ApprovalRecord, store: ApprovalStore) -> None:
    store.decide(record.id, approver="human:bench", approve=True)


def _run_scenario(
    checker: ModelChecker, scenario: DivergenceScenario, *, lying: bool, sensor: bool
) -> RunResult:
    registry = checker.registry
    world = FileMaintenanceWorld(
        dirs=dict(scenario.dirs), faults=dict(scenario.faults), reviewer=_approve
    )
    tools = build_tools(
        world, registry, misreporting=scenario.faulted_tools if lying else frozenset()
    )
    return PlanExecutor(
        registry,
        checker.specs,
        ToolGateway(registry, tools),
        opaque=world.decide,
        collections=world.collection,
        sensor=world.sense if sensor else None,
    ).run(scenario.plan)


def runtime_section(checker: ModelChecker) -> dict[str, Any]:
    scenarios = build_divergence_scenarios()
    blocked = 0
    statically_verified = 0
    kinds: dict[str, int] = {}
    lying_stopped = 0
    lying_by: dict[str, int] = {}
    blind_ran_past = 0
    for scenario in scenarios:
        statically_verified += checker.check(scenario.plan).ok
        honest = _run_scenario(checker, scenario, lying=False, sensor=True)
        ok = (
            honest.status == "blocked"
            and honest.blocked is not None
            and honest.blocked.spec == scenario.blocked_spec
            and honest.blocked.step == scenario.blocked_step
        )
        blocked += ok
        kind = scenario.name.rsplit("_", 1)[0]
        kinds[kind] = kinds.get(kind, 0) + ok
        # The faulty tool reports its declared (successful) effects instead of the truth.
        lying = _run_scenario(checker, scenario, lying=True, sensor=True)
        stopped = lying.status in {"blocked", "aborted"} and len(lying.executed) <= len(
            honest.executed
        )
        lying_stopped += stopped
        how = "misreport" if lying.misreports else lying.status
        lying_by[how] = lying_by.get(how, 0) + stopped
        blind = _run_scenario(checker, scenario, lying=True, sensor=False)
        blind_ran_past += len(blind.executed) > len(honest.executed)
    return {
        "scenarios": len(scenarios),
        "statically_verified": statically_verified,
        "blocked_before_violating_step": blocked,
        "block_rate": blocked / len(scenarios),
        "blocked_by_fault_kind": kinds,
        "lying_tools_stopped_with_sensor": lying_stopped,
        "lying_stop_rate": lying_stopped / len(scenarios),
        "lying_tools_stopped_by": lying_by,
        "lying_tools_without_sensor_ran_violating_step": blind_ran_past,
    }


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def check_thresholds(summary: dict[str, float]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        value = summary[name]
        if "min" in limits and value < limits["min"]:
            problems.append(f"{name} = {value} is below {limits['min']}")
        if "max" in limits and value > limits["max"]:
            problems.append(f"{name} = {value} exceeds {limits['max']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="checker, repair and runtime benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce thresholds_checker.json")
    args = parser.parse_args(argv)
    registry = file_maintenance_registry()
    checker = ModelChecker(registry, protocol_specs(registry))
    started = time.perf_counter()
    corpus = corpus_section(checker)
    scaling = scaling_section(checker)
    repair = repair_section(checker)
    hard = hard_repair_section(checker)
    runtime = runtime_section(checker)
    elapsed = time.perf_counter() - started
    summary = {
        "corpus_accuracy": corpus["accuracy"],
        "check_p95_ms": corpus["check_p95_ms"],
        "repair_success_rate": repair["success_rate"],
        "hard_corpus_flagged_rate": hard["flagged_rate"],
        "divergence_block_rate": runtime["block_rate"],
        "lying_stop_rate": runtime["lying_stop_rate"],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "machine": machine(),
                "summary": summary,
                "corpus": corpus,
                "scaling": scaling,
                "repair": repair,
                "hard_repair": hard,
                "runtime": runtime,
                "seconds": round(elapsed, 2),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"corpus: {corpus['correct']}/{corpus['cases']} verdicts correct "
        f"({corpus['violating']} violating, {corpus['cases'] - corpus['violating']} safe)"
    )
    print(
        f"check latency over {corpus['samples']} corpus checks: p50 {corpus['check_p50_ms']} ms, "
        f"p95 {corpus['check_p95_ms']} ms, max {corpus['check_max_ms']} ms; "
        f"max states {corpus['max_states']}"
    )
    print(
        "| family | n | calls | paths | instances | searched | states | p95 (ms) "
        "| states, no skip | p95 (ms), no skip |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in scaling:
        print(
            f"| {row['family']} | {row['n']} | {row['calls']} | {row['paths']} | "
            f"{row['instances']} | {row['searched']} | {row['states']} | {row['p95_ms']} | "
            f"{row['states_without_skip']} | {row['p95_ms_without_skip']} |"
        )
    print(
        f"repair: {repair['fixed_within_3_attempts']}/{repair['violating_plans']} fixed within 3 "
        f"attempts, attempts histogram {repair['attempts_histogram']}, dropped-step rejections "
        f"{repair['dropped_step_rejections']} ({repair['repairer']})"
    )
    print(
        f"hard repair corpus: {hard['cases']} plans, {hard['flagged_by_checker']} flagged by the "
        f"checker; {hard['verified_within_3_attempts']} verified within 3 attempts, "
        f"{hard['needs_override']} need a human override (failure rate "
        f"{hard['failure_rate']:.1%}); attempts histogram {hard['attempts_histogram']}; "
        f"dropped-step rejections {hard['dropped_step_rejections']}"
    )
    print(f"hard repair corpus by category: {hard['per_category']}")
    print(
        f"runtime: {runtime['blocked_before_violating_step']}/{runtime['scenarios']} divergences "
        f"blocked ({runtime['statically_verified']} plans statically verified); "
        f"by fault: {runtime['blocked_by_fault_kind']}"
    )
    print(
        f"lying tools: with the sensor {runtime['lying_tools_stopped_with_sensor']}/"
        f"{runtime['scenarios']} stopped before the violating step "
        f"(by {runtime['lying_tools_stopped_by']}); without a sensor the violating step ran in "
        f"{runtime['lying_tools_without_sensor_ran_violating_step']}/{runtime['scenarios']}"
    )
    print(f"total bench time {elapsed:.1f} s")
    if args.check:
        problems = check_thresholds(summary)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all checker benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
