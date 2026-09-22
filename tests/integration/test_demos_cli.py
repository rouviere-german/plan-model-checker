from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from plan_model_checker import cli
from plan_model_checker.agentic.audit import AuditLog
from plan_model_checker.core import tracing
from plan_model_checker.core.security import hash_api_key, parse_key_entry
from plan_model_checker.corpus import call, if_prop, plan
from plan_model_checker.demo import DEMO_TARGET, run_demo
from plan_model_checker.signing import SignedRegistry

pytestmark = pytest.mark.integration


def test_demo_tells_the_whole_story_offline(tmp_path: Path) -> None:
    summary = run_demo(tmp_path)
    assert summary["registry_tamper_refused"] is True
    assert summary["initial_verdict"] == "violation"
    assert summary["initial_violations"] == ["backup_before_delete"]
    assert summary["repair_status"] == "verified"
    assert summary["repair_attempts"] == 1
    assert summary["repaired_verdict"] == "safe"
    healthy = summary["healthy_run"]
    divergent = summary["divergent_run"]
    assert isinstance(healthy, dict)
    assert isinstance(divergent, dict)
    assert healthy["status"] == "completed"
    assert divergent["status"] == "blocked"
    assert divergent["divergences"] == [f"snapshot_dir(path={DEMO_TARGET})"]
    assert summary["deleted_in_divergent_run"] == []
    lying = summary["lying_run"]
    assert isinstance(lying, dict)
    assert lying["status"] == "aborted"
    assert lying["misreports"] == [
        f"snapshot_dir(path={DEMO_TARGET}): backup_ok[{DEMO_TARGET}] reported true, sensed false"
    ]
    assert summary["deleted_in_lying_run"] == []
    assert summary["audit_ok"] is True
    assert summary["tampered_audit_detected"] is True
    # Running twice starts from a clean state.
    assert run_demo(tmp_path)["audit_entries"] == summary["audit_entries"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_one_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["demo"] == "plan-model-checker"
    assert "blocked delete_dir" in captured.err


def test_cli_check_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    unsafe = plan(
        "x",
        call("request_approval", path="/a"),
        if_prop("approved", "/a", [call("archive_dir", path="/a"), call("delete_dir", path="/a")]),
    )
    safe = plan("x", call("archive_dir", path="/a"))
    (tmp_path / "unsafe.json").write_text(unsafe.model_dump_json(), encoding="utf-8")
    (tmp_path / "safe.json").write_text(safe.model_dump_json(), encoding="utf-8")
    (tmp_path / "bad.json").write_text('{"steps": [{"kind": "call", "tool": "rm_rf"}]}')
    assert cli.main(["check", str(tmp_path / "unsafe.json")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["violations"][0]["hint"]["suggestions"][0]["tool"] == "snapshot_dir"
    assert cli.main(["check", str(tmp_path / "safe.json")]) == 0
    capsys.readouterr()
    assert cli.main(["check", str(tmp_path / "bad.json")]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "PlanError"
    assert cli.main(["check", str(tmp_path / "unsafe.json"), "--spec", "release_all_locks"]) == 0
    capsys.readouterr()


def test_cli_keygen_and_sign_registry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    key = tmp_path / "signing.pem"
    assert cli.main(["keygen", "--out", str(key)]) == 0
    public = json.loads(capsys.readouterr().out)
    assert len(public["public_key"]) == 64
    assert cli.main(["keygen", "--out", str(key)]) == 2  # never overwrites a key
    capsys.readouterr()
    bundle_path = tmp_path / "registry.signed.json"
    assert cli.main(["sign-registry", "--key", str(key), "--out", str(bundle_path)]) == 0
    signed = json.loads(capsys.readouterr().out)
    bundle = SignedRegistry.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    assert bundle.sha256 == signed["sha256"]
    assert bundle.key_id == public["key_id"]
    registry_json = tmp_path / "registry.json"
    registry_json.write_text(bundle.registry.model_dump_json(), encoding="utf-8")
    again = tmp_path / "again.json"
    assert (
        cli.main(
            [
                "sign-registry",
                "--key",
                str(key),
                "--out",
                str(again),
                "--registry",
                str(registry_json),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["sha256"] == bundle.sha256


def test_cli_generate_api_key_with_scopes(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key", "--scopes", "check+repair"]) == 0
    payload = json.loads(capsys.readouterr().out)
    digest, _, scopes = payload["api_key_sha256"].partition(":")
    assert digest == hash_api_key(payload["api_key"])
    assert scopes == "check+repair"
    assert parse_key_entry(payload["api_key_sha256"])[1] == frozenset({"check", "repair"})


def test_cli_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("agent", "step", {})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from plan_model_checker.core.config import get_settings

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        get_settings.cache_clear()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["plan_model_checker", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("plan_model_checker", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
