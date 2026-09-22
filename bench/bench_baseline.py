"""Micro-benchmarks for the shared baseline components, with CI regression thresholds.

The checker, repair loop, runtime monitor and differential oracle have their own benchmarks:
bench/bench_checker.py and bench/bench_differential.py.

Usage:
  uv run python bench/bench_baseline.py          # run, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if bench/thresholds.json is exceeded

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from plan_model_checker.agentic.audit import AuditLog, verify_file
from plan_model_checker.agentic.providers.base import ChatMessage, LLMRequest
from plan_model_checker.core.ratelimit import RateLimiter
from plan_model_checker.core.redaction import redact_text

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = [sample / 1_000.0 for sample in samples]
    cuts = statistics.quantiles(micros, n=100, method="inclusive")
    return {
        "runs": runs,
        "p50_us": round(cuts[49], 1),
        "p95_us": round(cuts[94], 1),
    }


def bench_redaction() -> dict[str, float]:
    line = (
        "user jane.doe@example.com paid with 4111 1111 1111 1111 from +49 30 1234 5678; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return _timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("client"), runs=20_000)


def bench_request_hash() -> dict[str, float]:
    messages = tuple(ChatMessage.user(f"message {i} " + "x" * 400) for i in range(10))
    request = LLMRequest(system="bench", messages=messages)
    return _timed(request.sha256, runs=2_000)


def bench_audit_append(workdir: Path) -> dict[str, float]:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return _timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=200
    )


def bench_audit_verify(workdir: Path) -> dict[str, float]:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    for index in range(1_000):
        log.append("bench", "step", {"index": index, "note": "x" * 100})
    return _timed(lambda: verify_file(path), runs=10, warmup=1)


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def run() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        return {
            "redact_text_1kb": bench_redaction(),
            "rate_limiter_check": bench_rate_limiter(),
            "llm_request_sha256_10_messages": bench_request_hash(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_verify_1000_entries": bench_audit_verify(workdir),
        }


def check_thresholds(results: dict[str, dict[str, float]]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            value = results[name][metric]
            if value > limit:
                problems.append(f"{name}.{metric} = {value} exceeds threshold {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="baseline micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results}, indent=2) + "\n", encoding="utf-8"
    )
    print("| benchmark | runs | p50 (us) | p95 (us) |")
    print("|---|---:|---:|---:|")
    for name, stats in results.items():
        print(f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} |")
    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
