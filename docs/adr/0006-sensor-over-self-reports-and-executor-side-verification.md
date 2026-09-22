# 6. The sensor wins over tool reports; the executor verifies what it runs

- Status: accepted
- Date: 2026-09-22

## Context

An adversarial review of the runtime layer ([ADR 0004](0004-declared-effects-plus-runtime-monitor.md))
found that it was only as strong as the honesty of the tools and of its callers:

- the monitor merged the sensor's readings *under* the tool's report, so a snapshot tool that did
  nothing but reported `backup_ok[path] = true` got the following delete through, although a
  sensor was configured and would have read `false`;
- any tool could report any declared fluent, so an archive tool could report `approved[path]`;
- the monitor started from the registry's declared initial values, not from the world, so a lock
  another process already held was invisible;
- `PlanExecutor.run` executed any plan, verified or not, and took a waiver set from its caller
  instead of from an approved override.

## Decision

- **The sensor wins.** With a sensor, after every call the executor re-reads every fluent a spec
  instance depends on, every fluent the call may change and every fluent the tool reported. The
  sensed value is what the monitor records. A reported value the sensor contradicts is a
  *misreport*: it is audited (`tool.misreported`) and the run stops, because nothing else the
  tool says can be trusted either.
- **Declared effects only.** A report may only name fluents that some declared outcome of that
  call can set. Anything else (an approval from an archive tool, an undeclared proposition, two
  values for one fluent) is rejected before it is applied, in the executor and in `/v1/monitor`.
- **Sensed initial state.** With a sensor, the fluents a new spec instance reads are sensed before
  the first position (and, for a resource first mentioned later, when it appears; it is assumed
  to have kept that value since the start).
- **The executor verifies.** `PlanExecutor.run` checks the plan against its own specs and
  registry and refuses it (`refused`) unless it verifies, or unless an approved override is named
  and the executor itself consumes it for this plan hash and it covers every violated instance.

## Consequences

- The monitor no longer depends on tools telling the truth about anything the sensor can read.
  Without a sensor it still has to trust reports, within the call's declared effects; the README
  says so and `tests/unit/test_runtime_adversarial.py` pins that behaviour down.
- The benchmark runs every divergence scenario a second time with the faulty tool lying (it
  reports its declared effects): with the sensor all 30 are stopped before the violating step,
  without it the violating step runs in all 30.
- `/v1/monitor` has no sensor. It rejects undeclared effects but otherwise judges the effects the
  client reports, from the registry's initial values.
- Running a plan costs one extra check (milliseconds for corpus plans). A partial override (one
  that waives only some of the violated instances) no longer lets a plan start.
