# plan-model-checker

Checks LLM-generated agent plans against temporal resource-protocol rules *before anything runs*,
returns the shortest counterexample so the plan can be repaired, and enforces the same rules at
runtime on what the tools actually did.

## Problem

Operational rules constrain the *order* of steps, not single calls: back up before deleting,
release every lock on every path, check a job's status before retrying a non-idempotent call,
drain a service before restarting it and put it back afterwards. A per-call guardrail cannot
express "the backup must still be valid at the moment of the delete", and a rule written into a
prompt is a request, not a guarantee.

This repo treats a plan as a small program and model-checks it:

- the planner emits a **typed plan IR** (calls, branches on tracked facts or opaque data, bounded
  loops);
- every tool **declares its effects** on finite propositions (`backup_ok[path]`, `lock_held[path]`,
  `drained[service]`), in an **Ed25519-signed, hash-pinned registry**;
- rules are **past-time LTL** under a top-level `G`, e.g. `G(delete[r] -> Y backup_ok[r])`;
- an **explicit-state checker** explores every path (both branches of opaque predicates, every
  declared tool outcome, every loop count up to the bound) and returns the **shortest violating
  trace**;
- a **bounded repair loop** feeds that counterexample back to the planner; if repair fails, only a
  **human override with a recorded justification** lets the plan run;
- a **runtime monitor** compiled from the same specs wraps every tool call and blocks the step that
  would break a rule when a tool's real behaviour diverges from its declaration.

## Architecture

```mermaid
flowchart TB
    task[Task] --> planner["Planner (claude-sonnet-5 live / scripted offline)"]
    planner -->|flat JSON plan, structured output| ir["Plan IR (pydantic, closed schema, size limits)"]
    subgraph static["Static layer"]
        ir --> cfg["Ground + unroll to control-flow graph<br/>(collection sizes 0..bound)"]
        registry[("Effect registry<br/>Ed25519 signed, SHA-256 pinned")] --> cfg
        specs["Spec DSL (Lark) -> typed past-time LTL<br/>+ implicit approval specs"] --> mon["Compiled monitors<br/>(per-step bit vectors)"]
        cfg --> bfs["0-1 BFS over (pc, fluents, monitor state)<br/>memoized, cone of influence"]
        mon --> bfs
        bfs -->|safe| ok[Verified plan]
        bfs -->|violation| cex["Shortest counterexample<br/>+ repair hint (missing step, where)"]
    end
    cex --> repair["Repair loop (<= 3 attempts, token/USD budget)<br/>claude-opus-5 live / scripted offline"]
    repair --> ir
    repair -->|still violating| override["Human override<br/>justification + different approver"]
    override --> ok
    subgraph runtime["Runtime layer"]
        ok --> exec["Plan executor"]
        exec --> gate["Tool gateway<br/>(registered + declaration-pinned tools)"]
        gate --> tools["Mock file-maintenance tools<br/>(least-privilege world views)"]
        tools -->|observed effects + sensor re-read| rmon["Runtime monitor<br/>(same compiled specs)"]
        rmon -->|would violate| block[Block step]
    end
    bfs -.-> audit[("Hash-chained audit log")]
    rmon -.-> audit
    override -.-> audit
```

How one check works (`src/plan_model_checker/checker.py`):

1. Structural limits (nodes, nesting, loop bounds, unrolled size) are enforced first; every call
   is typed against the signed registry (unregistered tools, wrong parameters, sort clashes fail).
2. For each assignment of sizes to data-dependent collections, the plan is unrolled into a
   control-flow graph. `If` on a tracked fluent becomes a pair of guarded edges (path-sensitive);
   `If` on an opaque predicate becomes a nondeterministic choice; each declared tool outcome is a
   separate transition.
3. Specs are instantiated over the resources the plan mentions (active-domain semantics) and each
   instance is compiled into a per-step monitor (Havelund & Roşu's past-time algorithm).
4. A 0-1 BFS per instance (choices cost 0, calls cost 1) over `(control point, relevant fluents,
   monitor state)` finds the shortest trace that ends in a violating state. Paths that reach a
   seen state are pruned, so cost follows distinct states, not paths.

The trace model (positions, `start`/`end`, post-state fluents) and exactly what `safe` guarantees
are in [docs/SOUNDNESS.md](docs/SOUNDNESS.md). Design decisions:
[ADR 0002](docs/adr/0002-past-time-ltl-instead-of-full-ltlf.md) past-time LTL instead of LTLf,
[ADR 0003](docs/adr/0003-explicit-state-search-with-memoization.md) explicit-state search,
[ADR 0004](docs/adr/0004-declared-effects-plus-runtime-monitor.md) declared effects plus a
runtime monitor, [ADR 0005](docs/adr/0005-collections-as-snapshots-with-enumerated-sizes.md)
data-dependent loops.

The spec language (`src/plan_model_checker/spec/`): `Y` yesterday, `Z` weak yesterday, `O` once,
`H` historically, `S` since, boolean connectives, and a top-level `G`. The built-in rules for the
file-maintenance domain:

| Spec | Formula |
|---|---|
| backup_before_delete | `G(delete[r] -> Y backup_ok[r])` |
| release_all_locks | `G(end -> !lock_held[r])` |
| mutate_under_lock | `G(mutate[r] -> Y lock_held[r])` |
| no_double_acquire | `G(acquire[r] -> Y !lock_held[r])` |
| status_before_retry | `G(retry[j] -> Y status_checked[j])` |
| drain_before_restart | `G(restart[s] -> Y(drained[s] & (!undrain[s] S drain[s])))` |
| undrain_at_end | `G(end -> !drained[s])` |
| archive_before_delete | `G(delete[r] -> O archive[r])` |
| approval_delete_dir (implicit, from the registry) | `G(delete[path] -> Y approved[path])` |

Layout:

```text
src/plan_model_checker/
  ir.py  props.py  effects.py  signing.py  catalog.py      plan IR, effect model, signed registry
  spec/  (ast, parser, specs, monitor, semantics)          DSL, typing, compilation, reference semantics
  checker.py  explain.py  oracle.py  gen.py                 model checker, repair hints, oracle, generators
  runtime.py  world.py                                      runtime monitor, executor, mock tools
  planner.py  wire.py  override.py                          planner/repair loop, flat LLM schema, overrides
  engine.py  api.py  cli.py  demo.py  corpus.py            service, HTTP API, CLI, demo, labeled corpus
  core/  agentic/                                           shared baseline (security, logging, providers)
bench/bench_checker.py  bench/bench_differential.py  bench/bench_baseline.py
docs/SOUNDNESS.md  docs/THREAT_MODEL.md  docs/adr/  CLAIMS.md
```

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. Everything below runs offline and
never needs an API key.

```bash
uv sync --frozen
uv run python tasks.py demo    # the end-to-end story below, JSON summary on stdout
uv run python tasks.py check   # ruff, mypy --strict, pytest + coverage, bandit, pip-audit, baseline
uv run python tasks.py bench   # benchmarks with regression thresholds
```

On Windows with a small repository drive, keep the environment elsewhere first:
`$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/plan-model-checker"`.

The demo (scripted planner and repairer, mock tools) prints:

```text
* registry cab3bc95adca verified; tampered copy refused=True
* check: violation (backup_before_delete); counterexample: request_approval(path=/srv/logs/2025)[granted] -> archive_dir(path=/srv/logs/2025)[ok] -> delete_dir(path=/srv/logs/2025)[ok]
* hint: backup_ok[/srv/logs/2025] must hold immediately before the call at steps/1/then/1; candidate steps: snapshot_dir(path=/srv/logs/2025)
* repair: verified after 1 attempt(s); plan now runs request_approval, archive_dir, snapshot_dir, delete_dir
* run (healthy tools): completed; deleted=['/srv/logs/2025']; backups=['/srv/logs/2025']
* run (snapshot silently fails): blocked; blocked delete_dir(path=/srv/logs/2025) by backup_before_delete; deleted=[]
* audit: 17 entries verified=True; tampered copy detected=True
```

Check a plan file (exit code 0 safe, 1 violation, 2 rejected):

```bash
uv run plan-model-checker check plan.json
```

Run the service and call it:

```bash
uv run plan-model-checker generate-api-key --scopes check+monitor+repair   # key + digest entry
APP_API_KEY_SHA256=<digest entry> uv run plan-model-checker serve
curl -s -H "X-API-Key: <key>" -H "Content-Type: application/json" \
     -d @check-request.json http://127.0.0.1:8000/v1/check
```

Endpoints: `POST /v1/check` (scope `check`), `POST /v1/repair` (`repair`), `POST /v1/monitor`
(`monitor`: replay observed steps and ask whether the next call is allowed), `POST /v1/overrides`
(`override`) and `POST /v1/overrides/{id}/decision` (`approve`), plus `/healthz`, `/readyz`
(specs compiled, registry signature verified) and `/metrics`.

Production registry: `uv run plan-model-checker keygen --out signing.pem`, then
`sign-registry --key signing.pem --out registry.signed.json`, and set `APP_REGISTRY_PATH`,
`APP_REGISTRY_PUBLIC_KEYS` and `APP_REGISTRY_SHA256`. Without them, the built-in registry is signed
with an ephemeral key at start-up and `/readyz` fails in `prod`.

Live models: set `APP_REPAIRER=anthropic` and `ANTHROPIC_API_KEY`; the planner role uses
`claude-sonnet-5` and the repairer role `claude-opus-5` (both configurable), with structured output,
adaptive thinking and the server-side refusal fallback on Opus 5. No live run is part of this
repo's results.

Containers: `docker compose up --build` runs the non-root, read-only image. Build, smoke test,
trivy scan and SBOM run in CI (`.github/workflows/ci.yml`, job `docker`); they were not run
locally.

## Results

Measured with `uv run python tasks.py bench` (which runs `bench/bench_baseline.py --check`,
`bench/bench_checker.py --check` and `bench/bench_differential.py --check`) on 2026-09-22:
Windows 10 (10.0.19045), Intel64 Family 6 Model 158, 6 logical CPUs, Python 3.12.14, CPU only.
All three printed "within thresholds" / "agrees with the oracle on every case".

**Differential correctness** (`bench/bench_differential.py`, 5,000 seeded generated cases: plans
with nesting depth <= 6 and at most 8 unrolled calls, two generated specs each):

```text
cases: 5000 generated plans x 2 specs = 10000 verdicts (5708 violated, 4292 satisfied)
verdict mismatches: 0
counterexamples: 5708; replay failures: 0; not shortest: 0
oracle enumerated 22779 traces (max 8 unrolled calls per plan); checker 4.4 s, oracle 4.76 s
```

The hypothesis version of the same test also passed with 5,000 examples
(`DIFF_EXAMPLES=5000 uv run python tasks.py differential`: 4 passed in 51.98 s), and a seeded
one-operator bug in the monitor is caught by it (`tests/unit/test_differential.py`).

**Labeled corpus, repair and runtime** (`bench/bench_checker.py`):

```text
corpus: 60/60 verdicts correct (30 violating, 30 safe)
check latency over 1200 corpus checks: p50 0.582 ms, p95 2.022 ms, max 6.157 ms; max states 600
repair: 30/30 fixed within 3 attempts, attempts histogram {'1': 30} (scripted (inserts the step named by the counterexample))
runtime: 30/30 divergences blocked (30 plans statically verified); by fault: {'snapshot_silently_fails': 5, 'snapshot_overwritten': 5, 'lock_not_taken': 4, 'lock_lease_expires': 4, 'release_is_a_noop': 4, 'status_check_has_no_effect': 4, 'drain_has_no_effect': 4}
```

The repair rate measures the loop with a scripted repairer that inserts the step the
counterexample's hint names. It says nothing about how well an LLM repairs plans; no live-model
repair rates were recorded.

**State count and check time against plan size** (same run). `delete_chain` repeats
"request approval; if approved: archive, snapshot, delete" for n directories; `diamond_chain` has
n opaque branches between a job submission and a retry. Paths are execution paths (2^n); states
are what the search actually popped, summed over spec instances.

| family | n | calls | paths | instances | states | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| delete_chain | 1 | 4 | 2 | 6 | 44 | 0.615 |
| delete_chain | 2 | 8 | 4 | 12 | 172 | 1.729 |
| delete_chain | 4 | 16 | 16 | 24 | 680 | 5.718 |
| delete_chain | 8 | 32 | 256 | 48 | 2704 | 20.691 |
| delete_chain | 16 | 64 | 65536 | 96 | 10784 | 80.274 |
| delete_chain | 32 | 128 | 4294967296 | 192 | 43072 | 319.998 |
| diamond_chain | 1 | 4 | 2 | 1 | 6 | 0.205 |
| diamond_chain | 2 | 6 | 4 | 1 | 9 | 0.271 |
| diamond_chain | 4 | 10 | 16 | 1 | 15 | 0.375 |
| diamond_chain | 8 | 18 | 256 | 1 | 27 | 0.573 |
| diamond_chain | 16 | 34 | 65536 | 1 | 51 | 1.005 |
| diamond_chain | 32 | 66 | 4294967296 | 1 | 99 | 1.907 |

States stay linear in plan length per instance, but the number of instances grows with the number
of resources, so `delete_chain` grows roughly quadratically and passes the 50 ms budget between
n = 8 and n = 16 (32 and 64 call sites). The corpus plans are well inside it.

**Baseline micro-benchmarks** (`bench/bench_baseline.py`, microseconds): redact_text_1kb p95
402.9, rate_limiter_check p95 2.1, llm_request_sha256_10_messages p95 130.3, audit_append_fsync
p95 8923.0 (one fsync per entry, dominated by the disk), audit_verify_1000_entries p95 36195.4.

Tests: `uv run python tasks.py test` ran 533 tests (1 skipped: a template-only check) with 97.10%
line+branch coverage. These are single-machine numbers for spotting regressions, not performance
claims.

## Security

Threats, controls and the tests that prove them are in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md);
every claim here links to evidence in [CLAIMS.md](CLAIMS.md).

- **Two enforcement layers.** Plans are model-checked before execution; every tool call is then
  checked by the runtime monitor against observed effects. A model that obeys injected "the rule
  is waived" text still produces a plan the checker rejects.
- **Signed, pinned registry.** Declarations are verified against an Ed25519 signature from a
  trusted key and an optional SHA-256 pin; executable tools are pinned to the declaration hash
  they implement. Unregistered or modified tools are refused.
- **Human approval where it matters.** Destructive tools need an approval proposition that only a
  human decision (bound to the exact call, different principal, single use) can set; the registry
  turns this into an implicit spec that cannot be deselected. Overriding a counterexample needs a
  justification of at least 20 characters and a different approver, and is bound to one plan hash
  and the named spec instances.
- **Model and tool output are data.** Specs and registry are server-side and immutable; the only
  thing read back from a model is a plan in a closed schema. Rejected plans, counterexamples and
  tool outputs enter repair prompts inside untrusted-data envelopes.
- **Budgets.** Plan nodes, nesting, loop bounds, unrolled size, explored states, spec instances,
  collection-size assignments, repair attempts, steps, tokens and USD are all capped; exhausting a
  check budget is an error, never a "safe" verdict.
- **Least privilege.** Mock tools get a world view with only their own operations; the world
  refuses deletes without a granted approval.
- **Service hardening.** Scoped API keys (stored as SHA-256 digests), closed request schemas,
  rate limiting, body limits, security headers, non-leaking errors, PII redaction in logs and in
  the hash-chained audit log (plans, verdicts, counterexamples, runs, overrides).
- **CI.** ruff, mypy --strict, pytest with an 85% coverage floor, the 5,000-case differential job,
  benchmark thresholds, bandit, semgrep, pip-audit, gitleaks, hadolint, trivy and a CycloneDX SBOM.
  See [SECURITY.md](SECURITY.md) for reporting.

## Non-goals

- **Guarantees beyond the declared effects.** A `safe` verdict is about the registry's abstraction
  of the tools ([SOUNDNESS.md](docs/SOUNDNESS.md)). The runtime monitor covers divergences it can
  observe; effects that tools neither report nor expose to the sensor are invisible to both layers.
- **Liveness and future-time rules.** Only positive combinations of `G(past-time)` are accepted;
  "eventually" properties and full LTLf are out (ADR 0002). End-of-plan obligations are expressed
  with the `end` event.
- **Data-flow and confidentiality properties** (what may flow where); see the taint-flow-agent
  project.
- **Blocking end-state violations at runtime.** A violation that only appears at `end` (a release
  that silently did nothing) is detected and the run marked `violated`; nothing can be blocked.
- **Large-scale plans.** Explicit-state search is fine for agent-sized plans; the delete-chain row
  above passes 50 ms between 32 and 64 call sites. No symbolic (BDD/SAT) engine (ADR 0003).
- **Collections that depend on a loop variable** (instances of each service): collections are
  plan-wide snapshots (ADR 0005). No unbounded loops.
- **LLM repair quality.** Offline, plans come from the labeled corpus and repairs from a scripted
  repairer; the 30/30 repair rate tests the loop. No live-model repair rates or cassettes are
  recorded in this repo, and `tasks.py record` refuses to run.
- **Real tools and a real file system.** The demo agent runs against in-memory mock tools.
- **Tamper-proof state files.** The audit log is tamper-evident; the approval store and budget
  ledger are not signed. Distributed rate limiting, key rotation and multi-tenant isolation are
  out of scope.

## References

- K. Havelund and G. Roşu, "Synthesizing Monitors for Safety Properties", TACAS 2002 (per-step
  past-time monitors).
- Z. Manna and A. Pnueli, *Temporal Verification of Reactive Systems: Safety*, Springer 1995.
- G. De Giacomo and M. Y. Vardi, "Linear Temporal Logic and Linear Dynamic Logic on Finite
  Traces", IJCAI 2013 (the LTLf alternative).
- F. Chen and G. Roşu, "Parametric Trace Slicing and Monitoring", TACAS 2009 (parametric specs).
- E. M. Clarke, O. Grumberg and D. Peled, *Model Checking*, MIT Press 1999 (explicit-state search,
  counterexamples, cone of influence).
