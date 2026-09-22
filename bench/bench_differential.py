"""Differential correctness: model checker versus brute-force oracle on generated cases.

Usage:
  uv run python bench/bench_differential.py                 # 5,000 cases
  uv run python bench/bench_differential.py --cases 500     # quicker
  uv run python bench/bench_differential.py --check         # exit 1 on any disagreement

Each case is a generated plan (nesting depth <= 6, at most 8 unrolled call sites, opaque and
guarded branches, fixed and data-dependent loops, nondeterministic tool outcomes) and two
generated specs (every past-time operator under a top-level G). For every case and spec the
checker's verdict must equal the oracle's, which enumerates every trace and evaluates the textbook
semantics directly. For every counterexample, replaying it through the plan must give a trace the
oracle says violates the spec, and its length must equal the oracle's shortest violation.
The run also counts the spec instances the checker skipped as vacuous (they fold to true once
events no call emits and fluents no call changes are replaced by constants), so the oracle
comparison covers that shortcut too.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

from plan_model_checker.checker import ModelChecker
from plan_model_checker.gen import diff_registry, random_plan, random_specs
from plan_model_checker.ir import unrolled_call_count
from plan_model_checker.oracle import oracle_check, replay_counterexample, violates

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "differential.json"


def run(cases: int, seed: int) -> dict[str, object]:
    registry = diff_registry()
    verdict_mismatches = 0
    replay_failures = 0
    length_mismatches = 0
    spec_checks = 0
    violated_checks = 0
    counterexamples = 0
    instances = 0
    skipped = 0
    traces = 0
    checker_seconds = 0.0
    oracle_seconds = 0.0
    max_calls = 0
    examples: list[dict[str, object]] = []
    for index in range(cases):
        rng = random.Random(seed + index)
        subject = random_plan(rng)
        specs = random_specs(rng, registry, count=2)
        max_calls = max(max_calls, unrolled_call_count(subject.steps))
        started = time.perf_counter()
        result = ModelChecker(registry, specs).check(subject)
        checker_seconds += time.perf_counter() - started
        instances += result.instances_checked
        skipped += result.instances_skipped
        started = time.perf_counter()
        expected = oracle_check(subject, registry, specs)
        oracle_seconds += time.perf_counter() - started
        traces += expected.traces
        spec_checks += len(specs.names)
        violated_checks += sum(expected.violated.values())
        got = set(result.violated_specs())
        want = {name for name, bad in expected.violated.items() if bad}
        if got != want:
            verdict_mismatches += 1
            if len(examples) < 5:
                examples.append(
                    {"seed": seed + index, "checker": sorted(got), "oracle": sorted(want)}
                )
        for violation in result.violations:
            counterexamples += 1
            cex = violation.counterexample
            try:
                ok = violates(replay_counterexample(subject, registry, cex), cex.instance.source)
            except Exception:  # noqa: BLE001 - any replay failure counts as a failure
                ok = False
            replay_failures += not ok
            length_mismatches += cex.length != expected.shortest.get(violation.spec)
    return {
        "cases": cases,
        "seed": seed,
        "spec_checks": spec_checks,
        "violated_spec_checks": violated_checks,
        "safe_spec_checks": spec_checks - violated_checks,
        "verdict_mismatches": verdict_mismatches,
        "counterexamples": counterexamples,
        "replay_failures": replay_failures,
        "shortest_length_mismatches": length_mismatches,
        "instances": instances,
        "instances_skipped_as_vacuous": skipped,
        "oracle_traces_enumerated": traces,
        "max_unrolled_calls": max_calls,
        "checker_seconds": round(checker_seconds, 2),
        "oracle_seconds": round(oracle_seconds, 2),
        "mismatch_examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="checker vs brute-force oracle")
    parser.add_argument("--cases", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="exit 1 on any disagreement")
    args = parser.parse_args(argv)
    summary = run(args.cases, args.seed)
    machine = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"cases: {summary['cases']} generated plans x 2 specs = {summary['spec_checks']} verdicts "
        f"({summary['violated_spec_checks']} violated, {summary['safe_spec_checks']} satisfied)"
    )
    print(f"verdict mismatches: {summary['verdict_mismatches']}")
    print(
        f"counterexamples: {summary['counterexamples']}; replay failures: "
        f"{summary['replay_failures']}; not shortest: {summary['shortest_length_mismatches']}"
    )
    print(
        f"spec instances: {summary['instances']}, skipped as vacuous: "
        f"{summary['instances_skipped_as_vacuous']} (verdicts above include them)"
    )
    print(
        f"oracle enumerated {summary['oracle_traces_enumerated']} traces "
        f"(max {summary['max_unrolled_calls']} unrolled calls per plan); "
        f"checker {summary['checker_seconds']} s, oracle {summary['oracle_seconds']} s"
    )
    failures = (
        int(str(summary["verdict_mismatches"]))
        + int(str(summary["replay_failures"]))
        + int(str(summary["shortest_length_mismatches"]))
    )
    if args.check:
        if failures:
            print(f"REGRESSION: {failures} disagreements with the oracle")
            return 1
        print("checker agrees with the oracle on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
