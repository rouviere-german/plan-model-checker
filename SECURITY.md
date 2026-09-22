# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component (checker, spec parser, runtime monitor, API, registry signing);
- steps to reproduce, ideally as a failing test or a plan/spec pair;
- the impact you expect.

Of particular interest: a plan the checker reports `safe` that has a violating trace under the
declared effects (compare with `oracle_check`), a counterexample that does not replay, a way to
load a registry that was not signed by a trusted key or does not match the pin, a way to run a
destructive tool without a human approval, and a way for model or tool output to change specs.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration.
Out of scope: denial of service by traffic volume, findings that need a compromised host or a
leaked registry signing key, and issues in third-party dependencies that are already publicly
tracked (please report those upstream). Guarantees are relative to the declared tool effects;
see [docs/SOUNDNESS.md](docs/SOUNDNESS.md) before reporting a "missed" violation caused by a tool
behaving differently from its declaration (that is what the runtime monitor is for).

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- Two enforcement layers: static model checking before execution and a runtime monitor on every
  tool call, compiled from the same specs.
- The effect registry is Ed25519-signed and SHA-256-pinned; tools are pinned to their declaration.
- Destructive tools need an approval proposition that only a human decision can set; overrides of
  counterexamples need a recorded justification and a different approver.
- Budgets on plan size, loop bounds, explored states, spec instances, repair attempts, tokens and
  USD; exhaustion never yields a `safe` verdict.
- Scoped API keys, closed request schemas, rate limiting, body limits, PII redaction in logs and
  in the hash-chained audit log.
- CI runs ruff, mypy --strict, pytest with a coverage floor, the 5,000-case differential oracle
  job, bandit, semgrep (registry packs plus `.semgrep/rules.yml`), pip-audit, gitleaks, a trivy
  image scan and a syft CycloneDX SBOM.
- Secrets come only from the environment; only `.env.example` is committed. JSON only, no pickle.
