from __future__ import annotations

import hmac

import pytest

from plan_model_checker.core.security import (
    ApiKeyVerifier,
    extract_api_key,
    generate_api_key,
    hash_api_key,
)

pytestmark = pytest.mark.unit


def test_generated_keys_verify_against_their_digest() -> None:
    raw, digest = generate_api_key()
    assert digest == hash_api_key(raw)
    verifier = ApiKeyVerifier([hash_api_key("other-key"), digest])
    principal = verifier.verify(raw)
    assert principal is not None
    assert principal.key_id == digest[:12]
    assert verifier.verify("wrong-key") is None
    assert verifier.verify(None) is None
    assert verifier.verify("") is None
    assert verifier.verify("x" * 600) is None


def test_no_configured_keys_fails_closed() -> None:
    verifier = ApiKeyVerifier([])
    assert not verifier.configured
    assert verifier.verify("anything") is None


def test_every_configured_digest_is_compared_without_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = hmac.compare_digest

    def counting(a: bytes, b: bytes) -> bool:
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", counting)
    keys = [f"key-{i}" for i in range(5)]
    verifier = ApiKeyVerifier(hash_api_key(k) for k in keys)
    assert verifier.verify(keys[0]) is not None
    assert len(calls) == 5


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-api-key": "k1"}, "k1"),
        ({"authorization": "Bearer abc"}, "abc"),
        ({"authorization": "bearer  abc "}, "abc"),
        ({"authorization": "Basic abc"}, None),
        ({"authorization": "Bearer "}, None),
        ({}, None),
    ],
)
def test_extract_api_key(headers: dict[str, str], expected: str | None) -> None:
    assert extract_api_key(headers) == expected


def test_key_entries_carry_scopes_and_default_to_read_only() -> None:
    from plan_model_checker.core.security import DEFAULT_SCOPES, parse_key_entry

    digest = hash_api_key("scoped-key")
    verifier = ApiKeyVerifier([f"{digest}:check+repair", hash_api_key("plain-key")])
    scoped = verifier.verify("scoped-key")
    plain = verifier.verify("plain-key")
    assert scoped is not None
    assert plain is not None
    assert scoped.scopes == frozenset({"check", "repair"})
    assert plain.scopes == DEFAULT_SCOPES == frozenset({"check", "monitor"})
    assert scoped.name == f"key:{digest[:12]}"
    with pytest.raises(ValueError, match="unknown API key scopes"):
        parse_key_entry(f"{digest}:root")
    with pytest.raises(ValueError, match="SHA-256"):
        parse_key_entry("abcd")


def test_scope_dependencies_reject_unknown_scopes() -> None:
    from plan_model_checker.core.security import require_scope

    with pytest.raises(ValueError, match="unknown scope"):
        require_scope("superuser")
