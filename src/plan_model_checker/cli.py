"""Command line entry point: ``plan-model-checker`` or ``python -m plan_model_checker``.

Commands:
  serve                     run the HTTP service
  demo                      offline end-to-end demo (JSON summary on stdout)
  check PLAN.json           check a plan (exit 0 safe, 1 violation, 2 rejected)
  generate-api-key          print a new API key and its SHA-256 entry (with scopes)
  keygen --out KEY.pem      create an Ed25519 registry signing key
  sign-registry             sign the built-in (or a given) effect registry
  audit verify FILE         verify a hash-chained audit log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from plan_model_checker.api import create_service_app  # noqa: PLC0415
    from plan_model_checker.core.config import get_settings  # noqa: PLC0415
    from plan_model_checker.core.logging import configure_logging  # noqa: PLC0415
    from plan_model_checker.core.tracing import install_global_tracer_provider  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_service_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _demo(workdir: Path) -> int:
    from plan_model_checker.core.logging import configure_logging  # noqa: PLC0415
    from plan_model_checker.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    summary = run_demo(workdir)
    for line in summary["story"] if isinstance(summary["story"], list) else []:
        print(f"* {line}", file=sys.stderr)
    print(json.dumps({"demo": "plan-model-checker", **summary}, indent=2, sort_keys=True))
    return 0


def _check(path: Path, specs: list[str] | None) -> int:
    from pydantic import ValidationError  # noqa: PLC0415

    from plan_model_checker.catalog import (  # noqa: PLC0415
        file_maintenance_registry,
        protocol_specs,
    )
    from plan_model_checker.checker import CheckBudgetExceededError, ModelChecker  # noqa: PLC0415
    from plan_model_checker.explain import explain  # noqa: PLC0415
    from plan_model_checker.ir import Plan, PlanError  # noqa: PLC0415
    from plan_model_checker.spec.specs import SpecError  # noqa: PLC0415

    registry = file_maintenance_registry()
    checker = ModelChecker(registry, protocol_specs(registry))
    try:
        plan = Plan.model_validate_json(path.read_bytes())
        result = checker.check(plan, specs)
    except (ValidationError, PlanError, SpecError, CheckBudgetExceededError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)[:300]}))
        return 2
    payload = result.to_json()
    for violation, source in zip(payload["violations"], result.violations, strict=True):
        hint = explain(plan, registry, source.counterexample)
        violation["hint"] = hint.to_json() if hint is not None else None
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 1


def _keygen(out: Path) -> int:
    from plan_model_checker.signing import (  # noqa: PLC0415
        generate_signing_key,
        key_id_of,
        public_key_hex,
        save_private_key,
    )

    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    key = generate_signing_key()
    save_private_key(key, out)
    public = key.public_key()
    print(json.dumps({"public_key": public_key_hex(public), "key_id": key_id_of(public)}, indent=2))
    return 0


def _sign(key_path: Path, out: Path, registry_path: Path | None) -> int:
    from plan_model_checker.catalog import file_maintenance_registry_model  # noqa: PLC0415
    from plan_model_checker.effects import EffectRegistryModel  # noqa: PLC0415
    from plan_model_checker.signing import load_private_key, sign_registry  # noqa: PLC0415

    model = (
        EffectRegistryModel.model_validate_json(registry_path.read_bytes())
        if registry_path is not None
        else file_maintenance_registry_model()
    )
    bundle = sign_registry(model, load_private_key(key_path))
    out.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": bundle.sha256, "key_id": bundle.key_id, "out": str(out)}))
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one return per command
    parser = argparse.ArgumentParser(prog="plan-model-checker")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    check = sub.add_parser("check", help="model-check a plan JSON file")
    check.add_argument("plan", type=Path)
    check.add_argument("--spec", action="append", dest="specs", default=None)

    key = sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 entry")
    key.add_argument("--scopes", default="check+monitor", help="e.g. check+monitor+repair")

    keygen = sub.add_parser("keygen", help="create an Ed25519 registry signing key")
    keygen.add_argument("--out", type=Path, required=True)

    sign = sub.add_parser("sign-registry", help="sign an effect registry")
    sign.add_argument("--key", type=Path, required=True)
    sign.add_argument("--out", type=Path, required=True)
    sign.add_argument(
        "--registry", type=Path, default=None, help="registry JSON (default: built-in)"
    )

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "demo":
        return _demo(args.workdir)
    if args.command == "check":
        return _check(args.plan, args.specs)
    if args.command == "keygen":
        return _keygen(args.out)
    if args.command == "sign-registry":
        return _sign(args.key, args.out, args.registry)
    if args.command == "generate-api-key":
        from plan_model_checker.core.security import (  # noqa: PLC0415
            generate_api_key,
            parse_key_entry,
        )

        raw, digest = generate_api_key()
        entry = f"{digest}:{args.scopes}"
        parse_key_entry(entry)  # rejects unknown scopes
        print(json.dumps({"api_key": raw, "api_key_sha256": entry}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    from plan_model_checker.agentic.audit import main as audit_main  # noqa: PLC0415

    forwarded = ["verify", str(args.path)]
    if args.expect_head:
        forwarded += ["--expect-head", args.expect_head]
    return audit_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
