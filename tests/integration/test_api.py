from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from plan_model_checker.agentic.audit import verify_file
from plan_model_checker.api import create_service_app
from plan_model_checker.catalog import file_maintenance_registry_model
from plan_model_checker.core.config import Settings
from plan_model_checker.core.security import hash_api_key
from plan_model_checker.core.tracing import build_tracer_provider
from plan_model_checker.corpus import call, if_prop, plan
from plan_model_checker.signing import (
    RegistryIntegrityError,
    generate_signing_key,
    public_key_hex,
    sign_registry,
)

pytestmark = pytest.mark.integration

KEYS = {
    "checker": "key-checker-" + "c" * 20,
    "agent": "key-agent-" + "a" * 20,
    "lead": "key-lead-" + "l" * 20,
    "auditor": "key-auditor-" + "u" * 20,
}
SCOPES = {
    "checker": "check",
    "agent": "check+monitor+repair+override",
    "lead": "approve",
    "auditor": "audit",
}
PATH = "/srv/logs/2025"


def headers(who: str) -> dict[str, str]:
    return {"X-API-Key": KEYS[who], "Content-Type": "application/json"}


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "api_key_sha256": [f"{hash_api_key(KEYS[w])}:{SCOPES[w]}" for w in KEYS],
        "environment": "test",
        "data_dir": tmp_path,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_service_app(settings(tmp_path)), raise_server_exceptions=False) as c:
        yield c


def big_plan() -> dict[str, Any]:
    """Eight approved deletions: enough states to exhaust a 100-state budget.

    (Eight bare approval requests no longer do: every spec instance is vacuous on such a plan
    and the checker proves it safe without searching.)
    """
    steps: list[dict[str, Any]] = []
    for index in range(8):
        path = f"/d{index}"
        steps += [
            call("request_approval", path=path),
            if_prop(
                "approved", path, [call("archive_dir", path=path), call("delete_dir", path=path)]
            ),
        ]
    return plan("big", *steps).model_dump(mode="json")


def unsafe() -> dict[str, Any]:
    subject = plan(
        "delete logs",
        call("request_approval", path=PATH),
        if_prop("approved", PATH, [call("archive_dir", path=PATH), call("delete_dir", path=PATH)]),
    )
    return subject.model_dump(mode="json")


def test_check_returns_a_counterexample_with_a_repair_hint(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post("/v1/check", json={"plan": unsafe()}, headers=headers("checker"))
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "violation"
    (violation,) = body["violations"]
    assert violation["spec"] == "backup_before_delete"
    assert [s["tool"] for s in violation["counterexample"]["steps"]] == [
        "request_approval",
        "archive_dir",
        "delete_dir",
    ]
    assert violation["hint"]["suggestions"][0]["tool"] == "snapshot_dir"
    subset = client.post(
        "/v1/check",
        json={"plan": unsafe(), "specs": ["release_all_locks"]},
        headers=headers("checker"),
    )
    assert subset.json()["verdict"] == "safe"
    assert verify_file(tmp_path / "audit.jsonl").ok


def test_scopes_are_enforced_per_route(client: TestClient) -> None:
    body = {"plan": unsafe()}
    assert client.post("/v1/check", json=body).status_code == 401
    assert client.post("/v1/check", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    denied = client.post("/v1/repair", json={"task": "t", **body}, headers=headers("checker"))
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "forbidden"
    assert client.post("/v1/monitor", json={}, headers=headers("checker")).status_code == 403
    assert client.post("/v1/check", json=body, headers=headers("lead")).status_code == 403


def test_repair_endpoint_runs_the_bounded_loop(client: TestClient) -> None:
    response = client.post(
        "/v1/repair", json={"task": "delete old logs", "plan": unsafe()}, headers=headers("agent")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified"
    assert body["initial_violations"] == ["backup_before_delete"]
    tools = [s["tool"] for s in body["plan"]["steps"][1]["then"]]
    assert tools == ["archive_dir", "snapshot_dir", "delete_dir"]
    metrics = client.get("/metrics", headers=headers("agent")).text
    assert "repairs_succeeded_total 1.0" in metrics
    assert 'violations_total{spec="backup_before_delete"} 1.0' in metrics
    assert "check_latency_seconds_count" in metrics
    assert "states_explored_bucket" in metrics


def test_monitor_endpoint_blocks_a_delete_whose_backup_failed(client: TestClient) -> None:
    observed: list[dict[str, Any]] = [
        {
            "tool": "request_approval",
            "args": [{"name": "path", "value": PATH}],
            "effects": [{"prop": "approved", "args": [PATH], "value": True}],
        },
        {"tool": "archive_dir", "args": [{"name": "path", "value": PATH}], "effects": []},
        {
            "tool": "snapshot_dir",
            "args": [{"name": "path", "value": PATH}],
            "effects": [{"prop": "backup_ok", "args": [PATH], "value": False}],
        },
    ]
    proposed = {"tool": "delete_dir", "args": [{"name": "path", "value": PATH}]}
    response = client.post(
        "/v1/monitor", json={"observed": observed, "proposed": proposed}, headers=headers("agent")
    )
    body = response.json()
    assert body["allowed"] is False
    assert [v["spec"] for v in body["violations"]] == ["backup_before_delete"]
    observed[2]["effects"][0]["value"] = True
    allowed = client.post(
        "/v1/monitor",
        json={"observed": observed, "proposed": proposed, "finished": True},
        headers=headers("agent"),
    ).json()
    assert allowed["allowed"] is True
    assert allowed["violations"] == []
    metrics = client.get("/metrics", headers=headers("agent")).text
    assert "monitor_blocks_total 1.0" in metrics


@pytest.mark.parametrize(
    ("step", "code"),
    [
        ({"tool": "rm_rf", "args": [], "effects": []}, "unknown_tool"),
        ({"tool": "archive_dir", "args": [], "effects": []}, "invalid_arguments"),
        (
            {
                "tool": "archive_dir",
                "args": [{"name": "path", "value": "/a"}],
                "effects": [{"prop": "nope", "args": [], "value": True}],
            },
            "unknown_proposition",
        ),
        (
            {
                "tool": "archive_dir",
                "args": [{"name": "path", "value": "/a"}],
                "effects": [{"prop": "delete", "args": ["/a"], "value": True}],
            },
            "invalid_effect",
        ),
    ],
)
def test_monitor_rejects_calls_and_effects_outside_the_registry(
    client: TestClient, step: dict[str, Any], code: str
) -> None:
    response = client.post("/v1/monitor", json={"observed": [step]}, headers=headers("agent"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


def test_override_needs_a_justification_and_a_different_approver(client: TestClient) -> None:
    short = client.post(
        "/v1/overrides",
        json={"plan": unsafe(), "justification": "because"},
        headers=headers("agent"),
    )
    assert short.status_code == 422
    request = client.post(
        "/v1/overrides",
        json={"plan": unsafe(), "justification": "logs are duplicated in cold storage already"},
        headers=headers("agent"),
    )
    assert request.status_code == 200
    body = request.json()
    assert body["status"] == "pending"
    assert body["waived"] == [f"backup_before_delete[r={PATH}]"]
    decision_path = f"/v1/overrides/{body['request_id']}/decision"
    assert (
        client.post(decision_path, json={"approve": True}, headers=headers("agent")).status_code
        == 403
    )
    decided = client.post(
        decision_path, json={"approve": True, "reason": "ok"}, headers=headers("lead")
    )
    assert decided.json()["status"] == "approved"
    again = client.post(decision_path, json={"approve": False}, headers=headers("lead"))
    assert again.status_code == 409
    safe = plan("noop", call("archive_dir", path="/a")).model_dump(mode="json")
    nothing = client.post(
        "/v1/overrides",
        json={"plan": safe, "justification": "a long enough justification here"},
        headers=headers("agent"),
    )
    assert nothing.status_code == 409
    assert (
        client.post(
            "/v1/overrides/not-an-id/decision", json={"approve": True}, headers=headers("lead")
        ).status_code
        == 422
    )


def test_bad_payloads_get_generic_errors_without_echoing_input(client: TestClient) -> None:
    secret = "hunter2-" + "s" * 12
    extra = client.post(
        "/v1/check", json={"plan": unsafe(), "password": secret}, headers=headers("checker")
    )
    assert extra.status_code == 422
    assert secret not in extra.text
    unregistered = {"steps": [call("rm_rf", path="/")]}
    rejected = client.post("/v1/check", json={"plan": unregistered}, headers=headers("checker"))
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "invalid_plan"
    unknown_spec = client.post(
        "/v1/check", json={"plan": unsafe(), "specs": ["nope"]}, headers=headers("checker")
    )
    assert unknown_spec.json()["error"]["code"] == "unknown_spec"
    deep = (
        '{"plan": {"steps": ['
        + '{"kind": "if", "cond": {"kind": "opaque", "label": "x"}, "then": [' * 300
    )
    nested = client.post("/v1/check", content=deep, headers=headers("checker"))
    assert nested.status_code == 422
    steps: list[dict[str, Any]] = [call("archive_dir", path="/a")]
    for _ in range(10):
        steps = [{"kind": "if", "cond": {"kind": "opaque", "label": "x"}, "then": steps}]
    too_deep = {"steps": steps}
    limited = client.post("/v1/check", json={"plan": too_deep}, headers=headers("checker"))
    assert limited.status_code == 422
    assert "depth" in limited.json()["error"]["message"]


def test_state_budget_exhaustion_is_reported_not_verified(tmp_path: Path) -> None:
    app = create_service_app(settings(tmp_path, check_max_states=100))
    with TestClient(app) as client:
        response = client.post("/v1/check", json={"plan": unsafe()}, headers=headers("checker"))
        budget = client.post("/v1/check", json={"plan": big_plan()}, headers=headers("checker"))
    assert response.status_code == 200
    assert budget.status_code == 422
    assert budget.json()["error"]["code"] == "check_budget_exceeded"


def test_readiness_needs_compiled_specs_and_a_verified_registry(tmp_path: Path) -> None:
    with TestClient(create_service_app(settings(tmp_path))) as client:
        ready = client.get("/readyz").json()
    assert ready == {
        "status": "ready",
        "checks": {"specs_compiled": "ok", "registry_signature": "ok"},
    }
    prod = create_service_app(settings(tmp_path / "prod", environment="prod"))
    with TestClient(prod) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["registry_signature"] == "fail"


def test_a_configured_signed_and_pinned_registry_is_loaded(tmp_path: Path) -> None:
    key = generate_signing_key()
    bundle = sign_registry(file_maintenance_registry_model(), key)
    path = tmp_path / "registry.signed.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    good = settings(
        tmp_path,
        environment="prod",
        registry_path=path,
        registry_public_keys=[public_key_hex(key.public_key())],
        registry_sha256=bundle.sha256,
    )
    with TestClient(create_service_app(good)) as client:
        assert client.get("/readyz").status_code == 200
    wrong_pin = settings(
        tmp_path,
        registry_path=path,
        registry_public_keys=[public_key_hex(key.public_key())],
        registry_sha256="0" * 64,
    )
    with pytest.raises(RegistryIntegrityError):
        create_service_app(wrong_pin)
    untrusted = settings(
        tmp_path,
        registry_path=path,
        registry_public_keys=[public_key_hex(generate_signing_key().public_key())],
    )
    with pytest.raises(RegistryIntegrityError):
        create_service_app(untrusted)


def test_checks_emit_parse_check_and_repair_spans(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("test", extra_exporter=exporter)
    app = create_service_app(settings(tmp_path), tracer_provider=provider)
    with TestClient(app) as client:
        client.post("/v1/check", json={"plan": unsafe()}, headers=headers("checker"))
        client.post("/v1/repair", json={"task": "t", "plan": unsafe()}, headers=headers("agent"))
    names = {span.name for span in exporter.get_finished_spans()}
    assert {"parse", "check", "repair", "http.request"} <= names
    check = next(s for s in exporter.get_finished_spans() if s.name == "check")
    assert dict(check.attributes or {})["check.verdict"] == "violation"


def test_audit_entries_redact_pii_in_plans(tmp_path: Path) -> None:
    app = create_service_app(settings(tmp_path))
    pii_plan = plan("mail jane.doe@example.com when done", call("archive_dir", path="/a"))
    with TestClient(app) as client:
        client.post(
            "/v1/repair",
            json={"task": "notify jane.doe@example.com", "plan": pii_plan.model_dump(mode="json")},
            headers=headers("agent"),
        )
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "jane.doe@example.com" not in text
    for line in text.splitlines():
        json.loads(line)


def _archive_steps(count: int) -> list[dict[str, Any]]:
    return [
        {
            "tool": "archive_dir",
            "args": [{"name": "path", "value": f"/p/{index}"}],
            "effects": [{"prop": "archived", "args": [f"/p/{index}"], "value": True}],
        }
        for index in range(count)
    ]


def test_monitor_refuses_effects_the_tool_does_not_declare(client: TestClient) -> None:
    """A snapshot tool cannot also report a human approval for the path it backed up."""
    observed = [
        {
            "tool": "snapshot_dir",
            "args": [{"name": "path", "value": PATH}],
            "effects": [
                {"prop": "backup_ok", "args": [PATH], "value": True},
                {"prop": "approved", "args": [PATH], "value": True},
            ],
        },
        {"tool": "archive_dir", "args": [{"name": "path", "value": PATH}], "effects": []},
    ]
    proposed = {"tool": "delete_dir", "args": [{"name": "path", "value": PATH}]}
    response = client.post(
        "/v1/monitor", json={"observed": observed, "proposed": proposed}, headers=headers("agent")
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "undeclared_effect"


def test_monitor_requests_are_budgeted_and_never_a_500(tmp_path: Path) -> None:
    with TestClient(create_service_app(settings(tmp_path))) as client:
        within = client.post(
            "/v1/monitor", json={"observed": _archive_steps(300)}, headers=headers("agent")
        )
        # 500 distinct paths need 3,000 spec instances: over the default budget of 2,000.
        over = client.post(
            "/v1/monitor", json={"observed": _archive_steps(500)}, headers=headers("agent")
        )
    assert within.status_code == 200
    assert over.status_code == 422
    assert over.json()["error"]["code"] == "monitor_budget_exceeded"
    small = create_service_app(settings(tmp_path / "small", monitor_max_work=50))
    with TestClient(small) as client:
        busy = client.post(
            "/v1/monitor", json={"observed": _archive_steps(40)}, headers=headers("agent")
        )
    assert busy.status_code == 422
    assert busy.json()["error"]["code"] == "monitor_budget_exceeded"


def test_override_requests_report_check_budget_exhaustion_as_422(tmp_path: Path) -> None:
    app = create_service_app(settings(tmp_path, check_max_states=100))
    with TestClient(app) as client:
        response = client.post(
            "/v1/overrides",
            json={
                "plan": big_plan(),
                "justification": "a long enough justification for the test",
            },
            headers=headers("agent"),
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "check_budget_exceeded"


def test_monitor_decisions_are_audited_and_the_head_can_be_anchored(
    client: TestClient, tmp_path: Path
) -> None:
    observed = [
        {
            "tool": "request_approval",
            "args": [{"name": "path", "value": PATH}],
            "effects": [{"prop": "approved", "args": [PATH], "value": True}],
        },
    ]
    proposed = {"tool": "delete_dir", "args": [{"name": "path", "value": PATH}]}
    response = client.post(
        "/v1/monitor", json={"observed": observed, "proposed": proposed}, headers=headers("agent")
    )
    assert response.json()["allowed"] is False
    path = tmp_path / "audit.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    (decision,) = [entry for entry in entries if entry["action"] == "monitor.checked"]
    assert decision["data"]["allowed"] is False
    assert decision["data"]["proposed"] == f"delete_dir(path={PATH})"
    assert decision["data"]["observed_steps"] == 1
    assert f"backup_before_delete[r={PATH}]" in decision["data"]["violations"]
    head = client.get("/v1/audit/head", headers=headers("auditor"))
    assert head.status_code == 200
    verified = verify_file(path)
    assert head.json() == {"head": verified.head, "entries": verified.entries}
    assert client.get("/v1/audit/head", headers=headers("agent")).status_code == 403


def test_the_audit_head_is_only_published_for_a_verified_chain(
    client: TestClient, tmp_path: Path
) -> None:
    client.post("/v1/check", json={"plan": unsafe()}, headers=headers("checker"))
    path = tmp_path / "audit.jsonl"
    edited = path.read_bytes().replace(b"plan.checked", b"plan.checkex", 1)
    path.write_bytes(edited)
    response = client.get("/v1/audit/head", headers=headers("auditor"))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "audit_integrity"
