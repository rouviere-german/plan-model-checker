"""API-key authentication.

Keys are high-entropy random tokens, so a plain SHA-256 digest is an adequate at-rest form (a
slow password hash buys nothing against a 256-bit secret). The presented key is hashed first and
the digest is compared against *every* configured digest with ``hmac.compare_digest`` and no
early exit, so response timing reveals neither key prefixes nor which configured key matched.

Keys are scoped. A configured entry is ``<sha256>`` or ``<sha256>:<scope>+<scope>``; an entry
without scopes gets :data:`DEFAULT_SCOPES` (read-only checking). Routes declare the scope they
need with :func:`require_scope`, so a key minted for checking cannot start a paid repair or
approve an override.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from plan_model_checker.core.errors import AppError

__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_SCOPES",
    "SCOPES",
    "ApiKeyVerifier",
    "AuthenticatedPrincipal",
    "Principal",
    "extract_api_key",
    "generate_api_key",
    "hash_api_key",
    "parse_key_entry",
    "require_api_key",
    "require_scope",
]

API_KEY_HEADER = "X-API-Key"
_MAX_KEY_LENGTH = 512
SCOPES = frozenset({"check", "monitor", "repair", "override", "approve"})
DEFAULT_SCOPES = frozenset({"check", "monitor"})


def parse_key_entry(entry: str) -> tuple[bytes, frozenset[str]]:
    """Split ``<sha256>[:<scope>+<scope>]`` into the digest bytes and its scopes."""
    digest, _, scope_text = entry.partition(":")
    raw = bytes.fromhex(digest)
    if len(raw) != 32:
        raise ValueError("API key digests must be SHA-256")
    scopes = frozenset(s for s in scope_text.split("+") if s) if scope_text else DEFAULT_SCOPES
    unknown = scopes - SCOPES
    if unknown:
        raise ValueError(f"unknown API key scopes: {sorted(unknown)}")
    return raw, scopes


def hash_api_key(raw_key: str) -> str:
    """Lowercase hex SHA-256 of an API key, as stored in ``APP_API_KEY_SHA256``."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return a new (raw_key, sha256_hex) pair. Show the raw key once; store only the digest."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller, identified by a non-secret prefix of its key digest."""

    key_id: str
    scopes: frozenset[str] = DEFAULT_SCOPES

    @property
    def name(self) -> str:
        return f"key:{self.key_id}"


class ApiKeyVerifier:
    def __init__(self, sha256_hex_digests: Iterable[str]) -> None:
        self._entries = tuple(parse_key_entry(item) for item in sha256_hex_digests)
        self._digests = tuple(digest for digest, _ in self._entries)

    @property
    def configured(self) -> bool:
        return bool(self._digests)

    def verify(self, presented: str | None) -> Principal | None:
        """Return the principal for a valid key, else None. Fails closed with no keys set."""
        if not presented or len(presented) > _MAX_KEY_LENGTH:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        matched: tuple[bytes, frozenset[str]] | None = None
        for candidate, scopes in self._entries:
            if hmac.compare_digest(digest, candidate):
                matched = (candidate, scopes)
        if matched is None:
            return None
        return Principal(key_id=matched[0].hex()[:12], scopes=matched[1])


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Read the key from ``X-API-Key`` or ``Authorization: Bearer <key>``.

    Starlette's ``Headers`` is case-insensitive; plain mappings must use lowercase names.
    """
    value = headers.get("x-api-key")
    if value:
        return value
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def require_api_key(request: Request) -> Principal:
    """FastAPI dependency: authenticate the caller or raise a uniform 401."""
    verifier: ApiKeyVerifier = request.app.state.api_key_verifier
    principal = verifier.verify(extract_api_key(request.headers))
    if principal is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


AuthenticatedPrincipal = Annotated[Principal, Depends(require_api_key)]


def require_scope(scope: str) -> Callable[[Request], Principal]:
    """FastAPI dependency factory: a valid key that also carries ``scope`` (else 401/403)."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope}")

    def dependency(request: Request) -> Principal:
        principal = require_api_key(request)
        if scope not in principal.scopes:
            raise AppError(403, "forbidden", f"This API key lacks the '{scope}' scope.")
        return principal

    return dependency
