"""Cost of expensive requests against the HTTP service, and that budgets answer 422, never 500.

Usage:
  uv run python bench/bench_service.py          # run, write bench/results/service.json
  uv run python bench/bench_service.py --check  # also exit 1 on a 5xx or an unexpected status

Runs the real app in-process (FastAPI TestClient, default settings apart from a raised rate limit)
and times requests built to be expensive: /v1/monitor histories over many distinct paths (each
path adds six spec instances) or many steps on a few paths, and /v1/check plans at the size limits
or past the state budget. A second app loads an operator spec whose instances all read one event,
the case the monitor's work budget (200,000 monitor steps) exists for. The timings are
single-machine numbers for the threat model's denial-of-service row, not performance claims.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi.testclient import TestClient

from plan_model_checker.api import create_service_app
from plan_model_checker.core.config import Settings
from plan_model_checker.core.security import hash_api_key
from plan_model_checker.corpus import Step, call, each, each_of, if_opaque, if_prop, plan

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "service.json"
KEY = "bench-service-" + "k" * 24


# Every instance of this spec (one per choice of p, q and r) reads archive["/hub"], and archive
# steps alone never violate it, so all of them stay live and are stepped at every position.
HUB_SPEC = 'spec hub: G(archive["/hub"] -> !(O backup[p] & O mutate[q] & O acquire[r]));\n'


def _settings(data_dir: Path, specs_path: Path | None = None) -> Settings:
    return Settings(
        _env_file=None,
        api_key_sha256=[f"{hash_api_key(KEY)}:check+monitor"],
        environment="test",
        data_dir=data_dir,
        specs_path=specs_path,
        rate_limit_per_second=1_000.0,
        rate_limit_burst=1_000,
    )


def _step(tool: str, param: str, value: str, prop: str, flag: bool) -> dict[str, Any]:
    return {
        "tool": tool,
        "args": [{"name": param, "value": value}],
        "effects": [{"prop": prop, "args": [value], "value": flag}],
    }


def monitor_requests() -> list[tuple[str, dict[str, Any], int]]:
    """(name, body, expected status)."""

    def distinct(count: int) -> list[dict[str, Any]]:
        return [_step("archive_dir", "path", f"/p/{i}", "archived", True) for i in range(count)]

    churn: list[dict[str, Any]] = []
    for i in range(250):
        path = f"/q/{i % 5}"
        churn.append(_step("acquire_lock", "path", path, "lock_held", True))
        churn.append(_step("release_lock", "path", path, "lock_held", False))
    mixed: list[dict[str, Any]] = []
    for i in range(166):
        path = f"/m/{i}"
        mixed.append(_step("acquire_lock", "path", path, "lock_held", True))
        mixed.append(_step("compact_dir", "path", path, "backup_ok", False))
        mixed.append(_step("release_lock", "path", path, "lock_held", False))
    proposed = {"tool": "compact_dir", "args": [{"name": "path", "value": "/m/0"}]}
    return [
        ("333 steps on distinct paths (1,998 instances)", {"observed": distinct(333)}, 200),
        ("500 steps on distinct paths (3,000 instances)", {"observed": distinct(500)}, 422),
        ("500 lock steps on 5 paths", {"observed": churn, "finished": True}, 200),
        (
            "498 steps on 166 paths plus a proposed call",
            {"observed": mixed, "proposed": proposed, "finished": True},
            200,
        ),
    ]


def _guarded_delete(path: str) -> list[Step]:
    return [
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


def check_requests() -> list[tuple[str, dict[str, Any], int]]:
    chain = [step for i in range(40) for step in _guarded_delete(f"/c/{i}")]
    collections = [each_of(f"v{k}", f"coll{k}", 5, _guarded_delete(f"$v{k}")) for k in range(3)]
    nested = each(
        "a",
        [f"/a{i}" for i in range(8)],
        [
            each(
                "b",
                [f"/b{i}" for i in range(8)],
                [
                    call("acquire_lock", path="$a"),
                    if_opaque("x", [call("compact_dir", path="$a")]),
                    call("release_lock", path="$a"),
                    call("archive_dir", path="$b"),
                ],
            )
        ],
    )
    diamonds = [
        if_opaque(f"o{i}", [call("check_status", job="j")], [call("check_status", job="j")])
        for i in range(60)
    ]
    plans = [
        ("40 guarded deletes (200 nodes)", plan("chain", *chain), 200),
        ("3 collections of up to 5 items (216 size assignments)", plan("c", *collections), 422),
        ("8 x 8 nested loops", plan("nested", nested), 200),
        (
            "60 opaque diamonds",
            plan("d", call("submit_job", job="j"), *diamonds, call("retry_job", job="j")),
            200,
        ),
    ]
    return [(name, {"plan": p.model_dump(mode="json")}, status) for name, p, status in plans]


def hub_request() -> dict[str, Any]:
    """11 paths plus /hub make 12**3 = 1,728 hub instances; then alternate archive(/hub)."""
    observed = [_step("archive_dir", "path", f"/h/{i}", "archived", True) for i in range(11)]
    for i in range(489):
        path = "/hub" if i % 2 == 0 else "/h/0"
        observed.append(_step("archive_dir", "path", path, "archived", True))
    return {"observed": observed}


def _post(
    client: TestClient, route: str, name: str, body: dict[str, Any], expected: int
) -> dict[str, Any]:
    raw = json.dumps(body)
    started = time.perf_counter()
    response = client.post(
        route, content=raw, headers={"X-API-Key": KEY, "Content-Type": "application/json"}
    )
    seconds = time.perf_counter() - started
    payload = response.json()
    row: dict[str, Any] = {
        "route": route,
        "request": name,
        "bytes": len(raw),
        "status": response.status_code,
        "expected_status": expected,
        "seconds": round(seconds, 3),
        "error": payload.get("error", {}).get("code") if response.status_code != 200 else None,
    }
    if route == "/v1/check" and response.status_code == 200:
        row["states"] = payload["states_explored"]
        row["instances"] = payload["instances_checked"]
        row["skipped"] = payload["instances_skipped"]
    return row


def _monitor_steps(audit_path: Path, rows: list[dict[str, Any]]) -> None:
    """Copy each answered monitor request's work (audited as monitor.checked) into its row."""
    lines = audit_path.read_text(encoding="utf-8").splitlines() if audit_path.exists() else []
    work = [
        entry["data"]["work"]
        for entry in map(json.loads, lines)
        if entry["action"] == "monitor.checked"
    ]
    answered = [r for r in rows if r["status"] == 200 and r["route"] == "/v1/monitor"]
    for row, steps in zip(answered, work, strict=True):
        row["monitor_steps"] = steps


def run() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory() as tmp:
        app = create_service_app(_settings(Path(tmp)))
        with TestClient(app, raise_server_exceptions=False) as client:
            for route, requests in (
                ("/v1/monitor", monitor_requests()),
                ("/v1/check", check_requests()),
            ):
                rows.extend(_post(client, route, *request) for request in requests)
        _monitor_steps(Path(tmp) / "audit.jsonl", rows)
    with TemporaryDirectory() as tmp:
        specs = Path(tmp) / "hub.spec"
        specs.write_text(HUB_SPEC, encoding="utf-8")
        app = create_service_app(_settings(Path(tmp), specs_path=specs))
        with TestClient(app, raise_server_exceptions=False) as client:
            rows.append(
                _post(
                    client,
                    "/v1/monitor",
                    "operator spec read by all 1,728 instances, 500 steps",
                    hub_request(),
                    422,
                )
            )
        _monitor_steps(Path(tmp) / "audit.jsonl", rows[-1:])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="service request costs")
    parser.add_argument("--check", action="store_true", help="exit 1 on a 5xx or wrong status")
    args = parser.parse_args(argv)
    rows = run()
    machine = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"machine": machine, "rows": rows}, indent=2) + "\n", "utf-8")
    print("| route | request | bytes | status | seconds | details |")
    print("|---|---|---:|---:|---:|---|")
    for row in rows:
        details = row["error"] or ", ".join(
            f"{key} {row[key]}"
            for key in ("monitor_steps", "states", "instances", "skipped")
            if key in row
        )
        print(
            f"| {row['route']} | {row['request']} | {row['bytes']:,} | {row['status']} | "
            f"{row['seconds']} | {details} |"
        )
    if args.check:
        problems = [
            f"{row['route']} {row['request']}: HTTP {row['status']}, expected "
            f"{row['expected_status']}"
            for row in rows
            if row["status"] != row["expected_status"]
        ]
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("every expensive request answered as expected (no 5xx)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
