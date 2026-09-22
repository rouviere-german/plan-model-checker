# 4. Check against declared effects, enforce against observed effects

- Status: accepted
- Date: 2026-09-22

## Context

Static checking needs a model of what each tool does. Two options:

1. **Trust nothing statically**: only monitor at runtime. Violations are caught, but only after
   the plan has already done some of its work (a half-finished maintenance job), and the planner
   gets no feedback before execution.
2. **Abstract interpretation of tools** into declared effects over finite propositions: the plan
   can be verified, and repaired, before anything runs. The price is that the verdict is only as
   good as the declarations, and real tools fail in ways their declarations do not mention.

Declarations can model known failure modes as extra outcomes (`request_approval` has `granted`
and `pending`), which the checker explores. But declaring every possible failure as an outcome
makes almost every useful plan "unsafe" statically, and unknown failure modes cannot be declared
at all.

## Decision

Both layers, compiled from the same specs:

- the **checker** verifies plans against the declared effects in a signed, hash-pinned registry;
  declarations may stay optimistic where modelling every failure would make planning useless
  (`snapshot_dir` declares only success);
- the **runtime monitor** wraps every tool call. Before a call it predicts the next position from
  the *observed* state for every declared outcome and blocks the call if any spec would become
  false. After the call it applies the effects the tool reported plus a sensor re-read of every
  fluent a live spec instance depends on, flags a divergence when the observed change is not
  exactly a declared outcome, and stops the run if the observed state already violates a spec.
  ([ADR 0006](0006-sensor-over-self-reports-and-executor-side-verification.md) tightens this:
  the sensor's reading wins over the report, a contradicted report stops the run, and tools may
  only report their declared effects.)
- Tool implementations are pinned to the hash of the declaration they were written against, so a
  registry that silently changes a tool's declared effects cannot be paired with the old code.

## Consequences

- Plans are repaired before execution; divergences are contained during execution. The corpus
  benchmark measures both: 60/60 static verdicts and 30/30 divergence scenarios blocked before the
  violating step.
- Guarantees are stated relative to the declarations ([SOUNDNESS.md](../SOUNDNESS.md)).
- Specs must name *fluents* when they depend on an effect having happened: an event such as
  `drain[s]` records that the call was made, not that it worked. `drain_before_restart` uses both.
- The sensor is a trust boundary: with a sensor the monitor is as good as what can be re-read;
  without one it is only as good as what tools report. Effects that are neither reported nor
  sensed stay invisible.
- End-of-plan obligations can only be detected at the end, not blocked.
