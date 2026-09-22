"""Ed25519-signed, hash-pinned effect registries.

A registry bundle carries the declarations, their canonical SHA-256 and an Ed25519 signature over
a domain-separated message containing that digest. Loading a bundle recomputes the digest from the
declarations (so an edited declaration no longer matches), checks it against an optional pin
(so a validly signed but *different* registry is refused too), and verifies the signature with a
trusted public key selected by key id. Anything else raises :class:`RegistryIntegrityError`; there
is no "load anyway" path.
"""

from __future__ import annotations

import base64
import hmac
import os
from collections.abc import Mapping
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from plan_model_checker.core.canonical import canonical_json, sha256_hex
from plan_model_checker.effects import EffectRegistry, EffectRegistryModel, RegistryError

__all__ = [
    "SIGNATURE_DOMAIN",
    "RegistryIntegrityError",
    "SignedRegistry",
    "generate_signing_key",
    "key_id_of",
    "load_private_key",
    "load_signed_registry",
    "public_key_from_hex",
    "public_key_hex",
    "save_private_key",
    "sign_registry",
    "verify_signed_registry",
]

SIGNATURE_DOMAIN = b"plan-model-checker/effect-registry/v1\x00"


class RegistryIntegrityError(Exception):
    """The registry bundle is unsigned, tampered, signed by an unknown key or not the pinned one."""


class SignedRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    registry: EffectRegistryModel
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    key_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    signature: str = Field(min_length=1, max_length=200)


def _registry_digest(model: EffectRegistryModel) -> str:
    return sha256_hex(canonical_json(model.model_dump(mode="json")))


def _message(digest: str) -> bytes:
    return SIGNATURE_DOMAIN + digest.encode("ascii")


def public_key_hex(key: Ed25519PublicKey) -> str:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def public_key_from_hex(value: str) -> Ed25519PublicKey:
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise RegistryIntegrityError("public key must be 64 hex characters") from exc
    if len(raw) != 32:
        raise RegistryIntegrityError("public key must be 64 hex characters")
    return Ed25519PublicKey.from_public_bytes(raw)


def key_id_of(key: Ed25519PublicKey) -> str:
    """Stable, non-secret identifier of a public key (first 16 hex chars of its SHA-256)."""
    return sha256_hex(bytes.fromhex(public_key_hex(key)))[:16]


def generate_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def save_private_key(key: Ed25519PrivateKey, path: Path) -> None:
    """Write the key as unencrypted PKCS#8 PEM, readable only by the owner where supported."""
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem)


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise RegistryIntegrityError("signing key is not an Ed25519 private key")
    return key


def sign_registry(model: EffectRegistryModel, key: Ed25519PrivateKey) -> SignedRegistry:
    EffectRegistry(model)  # refuse to sign declarations that do not validate
    digest = _registry_digest(model)
    signature = key.sign(_message(digest))
    return SignedRegistry(
        registry=model,
        sha256=digest,
        key_id=key_id_of(key.public_key()),
        signature=base64.b64encode(signature).decode("ascii"),
    )


def verify_signed_registry(
    bundle: SignedRegistry,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    *,
    pinned_sha256: str | None = None,
) -> EffectRegistry:
    """Return the registry only if digest, pin and signature all check out."""
    digest = _registry_digest(bundle.registry)
    if not hmac.compare_digest(digest, bundle.sha256):
        raise RegistryIntegrityError("registry declarations do not match the signed digest")
    if pinned_sha256 is not None and not hmac.compare_digest(digest, pinned_sha256):
        raise RegistryIntegrityError("registry digest does not match the pinned digest")
    key = trusted_keys.get(bundle.key_id)
    if key is None:
        raise RegistryIntegrityError(f"registry is signed by an untrusted key {bundle.key_id}")
    try:
        signature = base64.b64decode(bundle.signature, validate=True)
        key.verify(signature, _message(digest))
    except (InvalidSignature, ValueError) as exc:
        raise RegistryIntegrityError("registry signature is invalid") from exc
    try:
        return EffectRegistry(bundle.registry)
    except RegistryError as exc:
        raise RegistryIntegrityError(f"signed registry is inconsistent: {exc}") from exc


def load_signed_registry(
    path: Path,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    *,
    pinned_sha256: str | None = None,
) -> EffectRegistry:
    try:
        bundle = SignedRegistry.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise RegistryIntegrityError(f"{path.name} is not a signed registry bundle") from exc
    return verify_signed_registry(bundle, trusted_keys, pinned_sha256=pinned_sha256)
