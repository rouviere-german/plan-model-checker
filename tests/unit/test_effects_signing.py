from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from plan_model_checker.catalog import file_maintenance_registry_model
from plan_model_checker.effects import (
    EffectRegistry,
    EffectRegistryModel,
    RegistryError,
    UnknownToolError,
)
from plan_model_checker.props import GroundProp
from plan_model_checker.signing import (
    RegistryIntegrityError,
    SignedRegistry,
    generate_signing_key,
    key_id_of,
    load_private_key,
    load_signed_registry,
    public_key_from_hex,
    public_key_hex,
    save_private_key,
    sign_registry,
    verify_signed_registry,
)

pytestmark = pytest.mark.unit


def model_json() -> dict[str, Any]:
    return file_maintenance_registry_model().model_dump(mode="json")


def mutate(change: Any) -> EffectRegistryModel:
    data = model_json()
    change(data)
    return EffectRegistryModel.model_validate(data)


def tool(data: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in data["tools"] if item["name"] == name)


def test_grounding_events_outcomes_and_approvals(registry: EffectRegistry) -> None:
    args = {"path": "/a"}
    assert registry.ground_event("delete_dir", args) == GroundProp("delete", ("/a",))
    outcomes = registry.ground_outcomes("request_approval", args)
    assert [o.name for o in outcomes] == ["granted", "pending"]
    assert outcomes[0].updates == ((GroundProp("approved", ("/a",)), True),)
    assert registry.ground_approval("delete_dir", args) == GroundProp("approved", ("/a",))
    assert registry.ground_approval("snapshot_dir", args) is None
    assert registry.param_sorts("drain_service") == {"service": "service"}
    assert registry.initial(GroundProp("backup_ok", ("/a",))) is False
    with pytest.raises(UnknownToolError):
        registry.tool("rm_rf")
    with pytest.raises(RegistryError):
        registry.prop("nope")


def test_destructive_tools_yield_implicit_approval_specs(registry: EffectRegistry) -> None:
    assert registry.implicit_specs() == (
        ("approval_delete_dir", "G(delete[path] -> Y approved[path])"),
    )


def test_establishers_name_tools_that_make_a_proposition_true(registry: EffectRegistry) -> None:
    assert [e.tool for e in registry.establishers("backup_ok", True)] == ["snapshot_dir"]
    assert [e.tool for e in registry.establishers("lock_held", False)] == ["release_lock"]
    assert [e.tool for e in registry.establishers("drain", True)] == ["drain_service"]
    approval = registry.establishers("approved", True)
    assert approval[0].tool == "request_approval"
    assert approval[0].always is False  # it may stay pending


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: tool(d, "snapshot_dir")["params"][0].update(sort="disk"), "unknown sort"),
        (lambda d: tool(d, "snapshot_dir")["event"].update(prop="backup_ok"), "not a declared"),
        (lambda d: tool(d, "snapshot_dir")["event"].update(args=[]), "arity"),
        (lambda d: tool(d, "snapshot_dir")["event"].update(prop="start", args=[]), "reserved"),
        (
            lambda d: tool(d, "drain_service")["outcomes"][0]["effects"][0].update(
                prop="backup_ok"
            ),
            "expects sort",
        ),
        (
            lambda d: tool(d, "snapshot_dir")["outcomes"][0]["effects"][0].update(args=["other"]),
            "unknown parameter",
        ),
        (lambda d: tool(d, "write_file")["event"].update(prop="delete"), "destructive"),
        (lambda d: d["props"].append({"name": "end", "kind": "event"}), "reserved"),
        (lambda d: d["props"].append(d["props"][0]), "reserved or duplicated"),
        (lambda d: d["props"].append({"name": "x", "kind": "event", "initial": True}), "initial"),
        (lambda d: d["props"].append({"name": "y", "kind": "fluent", "params": ["disk"]}), "sorts"),
        (lambda d: d["tools"].append(d["tools"][0]), "twice"),
        (lambda d: d.update(sorts=["path", "path", "job", "service"]), "duplicate sort"),
    ],
)
def test_inconsistent_declarations_are_rejected(change: Any, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        EffectRegistry(mutate(change))


def test_destructive_tools_must_declare_an_approval() -> None:
    def drop_approval(data: dict[str, Any]) -> None:
        tool(data, "delete_dir")["approval"] = None

    with pytest.raises(ValidationError, match="approval"):
        mutate(drop_approval)


def test_signed_registry_round_trips_through_a_file(tmp_path: Path) -> None:
    key = generate_signing_key()
    save_private_key(key, tmp_path / "keys" / "signing.pem")
    loaded_key = load_private_key(tmp_path / "keys" / "signing.pem")
    bundle = sign_registry(file_maintenance_registry_model(), loaded_key)
    path = tmp_path / "registry.signed.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    public = public_key_from_hex(public_key_hex(key.public_key()))
    registry = load_signed_registry(path, {key_id_of(public): public}, pinned_sha256=bundle.sha256)
    assert registry.sha256 == bundle.sha256
    assert bundle.key_id == key_id_of(key.public_key())


def test_modified_unknown_unpinned_or_forged_registries_are_refused(tmp_path: Path) -> None:
    key = generate_signing_key()
    trusted = {key_id_of(key.public_key()): key.public_key()}
    bundle = sign_registry(file_maintenance_registry_model(), key)
    assert verify_signed_registry(bundle, trusted).sha256 == bundle.sha256

    def no_approval(data: dict[str, Any]) -> None:
        tool(data, "delete_dir").update(destructive=False, approval=None)

    edited = bundle.model_copy(update={"registry": mutate(no_approval)})
    with pytest.raises(RegistryIntegrityError, match="do not match"):
        verify_signed_registry(edited, trusted)

    other = generate_signing_key()
    resigned = sign_registry(mutate(no_approval), other)
    with pytest.raises(RegistryIntegrityError, match="untrusted key"):
        verify_signed_registry(resigned, trusted)
    # Trusted signer, but not the pinned registry.
    legit_but_different = sign_registry(mutate(no_approval), key)
    with pytest.raises(RegistryIntegrityError, match="pinned"):
        verify_signed_registry(legit_but_different, trusted, pinned_sha256=bundle.sha256)

    forged = bundle.model_copy(update={"signature": base64.b64encode(b"\x00" * 64).decode("ascii")})
    with pytest.raises(RegistryIntegrityError, match="signature is invalid"):
        verify_signed_registry(forged, trusted)
    garbage = bundle.model_copy(update={"signature": "not base64!"})
    with pytest.raises(RegistryIntegrityError, match="signature is invalid"):
        verify_signed_registry(garbage, trusted)

    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"registry": {}}', encoding="utf-8")
    with pytest.raises(RegistryIntegrityError, match="not a signed registry"):
        load_signed_registry(bad_file, trusted)


def test_public_keys_must_be_raw_ed25519_hex() -> None:
    for bad in ("zz", "ab" * 10):
        with pytest.raises(RegistryIntegrityError):
            public_key_from_hex(bad)


def test_signing_refuses_invalid_declarations() -> None:
    key = generate_signing_key()

    def broken(data: dict[str, Any]) -> None:
        tool(data, "snapshot_dir")["event"]["prop"] = "backup_ok"

    with pytest.raises(RegistryError):
        sign_registry(mutate(broken), key)


def test_signed_bundles_are_closed_models() -> None:
    with pytest.raises(ValidationError):
        SignedRegistry.model_validate(
            {
                "registry": model_json(),
                "sha256": "0" * 64,
                "key_id": "0" * 16,
                "signature": "x",
                "unsigned_extra": True,
            }
        )
