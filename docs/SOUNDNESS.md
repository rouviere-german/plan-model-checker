# What a "safe" verdict means

The checker proves a precise statement, and only that statement. This page spells it out so a
green verdict is not read as more than it is.

## The guarantee

For a plan `P`, a spec set `S` and a verified effect registry `R`:

> If `ModelChecker.check(P)` returns `safe`, then every trace that `P` can produce **under the
> abstraction `R`** satisfies every spec in `S` (and every implicit approval spec derived from
> `R`), for every assignment of sizes to the plan's data-dependent collections up to their
> declared bounds, with every spec variable ranging over the resources the plan mentions.

If it returns `violation`, the attached counterexample is a trace of `P` under `R` that falsifies
the spec, and no shorter such trace exists (length = number of tool calls). If the check exceeds
a budget (states, spec instances, size assignments) it raises instead of answering: a plan that
could not be checked is never reported safe.

Some spec instances are not searched because they cannot fail on the plan: an event that no call
in the control-flow graph emits is false at every position, a fluent that no declared outcome can
set keeps its initial value, and if folding those constants into the instance gives `true`
(`src/plan_model_checker/spec/simplify.py`, rules that are exact at every position) the instance
holds on every trace. This changes the cost of a check, not its answer; the differential run
below covers it (4,813 of 19,369 instances skipped in the 5,000 cases, 0 disagreements).

## The trace model

A trace of a plan with `n` tool calls has `n + 2` positions:

| position | event | fluents |
|---|---|---|
| 0 | `start` | declared initial values |
| `i` (1..n) | the event of call `i`, e.g. `delete[/srv/logs]` | values **after** call `i`'s effects |
| `n + 1` | `end` | same as position `n` |

Exactly one event holds at each position. Fluents are the registry's state propositions
(`backup_ok[r]`, `lock_held[r]`, ...). A formula `G ψ` holds when `ψ` holds at every position;
the checker evaluates the equivalent `H ψ` at the last position, incrementally
([ADR 0002](adr/0002-past-time-ltl-instead-of-full-ltlf.md)).

"Traces `P` can produce under `R`" means:

- every **declared outcome** of every call is possible (a tool with outcomes `granted` and
  `pending` branches);
- an `If` on a **tracked proposition** takes the branch that proposition's abstract value
  selects (path-sensitive);
- an `If` on an **opaque** predicate can take either branch;
- a loop over **items** runs once per listed item; a loop over a **collection** runs 0 to
  `max_iterations` times over fresh symbolic items, and every loop over the same collection label
  sees the same items (a collection is a snapshot of one run).

## Where the guarantee stops

1. **Effect declarations are trusted by the checker.** If `snapshot_dir` declares
   `backup_ok[path] := true` and the real tool silently fails, the static verdict is about a world
   that did not happen. This is the reason for the runtime monitor
   ([ADR 0004](adr/0004-declared-effects-plus-runtime-monitor.md),
   [ADR 0006](adr/0006-sensor-over-self-reports-and-executor-side-verification.md)):
   - With a **sensor**, the monitor records what the sensor reads for every fluent a spec depends
     on, every fluent the call may change and every fluent the tool reported. A report the sensor
     contradicts stops the run. So a tool that fails *and* claims success is caught, as long as
     the sensor can read the fluent.
   - **Without a sensor**, the monitor can only apply what tools report. A tool that lies about a
     declared effect is believed (`tests/unit/test_runtime_adversarial.py::test_without_a_sensor_the_monitor_can_only_trust_self_reports`).
   - In both cases a tool may only report fluents its declaration says the call can change; an
     approval reported by anything but `request_approval` is rejected.
   - An effect that is neither reported nor readable by the sensor is invisible to both layers.
2. **Events are trusted as calls, not as effects.** `drain[s]` holds because `drain_service` was
   called, whether or not it drained anything. Specs that need an effect should name the fluent
   (`drained[s]`), as `drain_before_restart` does; event-only specs cannot detect a no-op tool.
3. **Active-domain quantification.** `G(end -> !lock_held[r])` is checked for every `r` the plan
   mentions (in tool arguments or guards), not for every resource that exists. A spec that
   requires something of resources the plan never mentions is vacuous for them.
4. **Symbolic collection items are assumed fresh and distinct.** The checker treats
   `stale_dirs#0` as different from `/srv/logs` and from `other#0`. The executor enforces this at
   run start: a collection with duplicates, items that equal any constant the plan names (call
   arguments, guard arguments, listed loop items) or a spec constant, items shared with another
   collection, items that are not well-formed resource names, or more items than the bound
   aborts the run before any call.
5. **Resource names are compared as strings.** `/srv/logs` and `/srv/logs/` are different
   resources to the checker. Names with `.` or `..` path segments are refused everywhere a name
   can enter (plans, collections, monitor reports, spec constants), but real tools must still
   resolve paths against an allowed root; the mock tools here have no file system.
6. **Only safety properties.** Specs are positive combinations of `G(past-time formula)`. Nothing
   of the form "eventually X" (liveness), and no data-flow or confidentiality property (see
   taint-flow-agent). `G(end -> X)` expresses "X holds when the plan ends", which covers the usual
   release/undrain obligations for terminating plans.
7. **Bounded loops only.** Loops are unrolled to their declared bound; unbounded iteration is not
   expressible in the IR.
8. **The runtime monitor blocks steps, not plan ends.** A violation that only appears at `end`
   (a lock still held because `release_lock` was a no-op) is detected when the run finishes and
   the run is marked `violated`; nothing can be "blocked" at that point.
9. **The initial state.** The checker assumes the registry's declared initial values. At runtime,
   with a sensor, the monitor senses the fluents each spec instance reads before the first
   position, so a lock another holder already has blocks `acquire_lock`; a resource the run first
   mentions later is assumed to have had its sensed value since the start. Without a sensor (and
   in `/v1/monitor`) the declared initial values are assumed.
10. **Overrides are explicit and consumed by the executor.** `PlanExecutor.run` re-checks every
    plan and refuses one that does not verify unless an approved override is named, the executor
    consumes it for exactly this plan hash, and it waives every violated instance. The monitor
    still evaluates waived instances and records each waived violation in the audit log.

## How the implementation is checked

- The compiled monitor is compared with a direct, textbook evaluator of the formula at every
  position of random traces (`tests/unit/test_spec.py`).
- The checker's verdicts, counterexample validity and counterexample length are compared with a
  brute-force oracle that enumerates every trace of the plan and evaluates the future-time `G`
  definition directly (`tests/unit/test_differential.py`, `bench/bench_differential.py`: 5,000
  generated cases per CI run, 0 disagreements allowed).
- A seeded bug (Y evaluated as Z) is caught by the same harness
  (`tests/unit/test_differential.py::test_the_oracle_catches_a_seeded_monitor_bug`), which shows
  that the comparison is sensitive enough to notice a one-operator mistake.
- Constant folding is checked against the textbook evaluator at every position of random traces
  (`tests/unit/test_spec.py::test_folding_fixed_propositions_preserves_the_value_at_every_position`).
- The incremental runtime monitor (instances created as resources appear, partial replay, only
  instances whose letter can change are stepped) is compared with a naive reference that folds
  every instance over the whole trace, including specs with several variables and constants
  (`tests/unit/test_runtime_monitor.py::test_the_incremental_monitor_agrees_with_a_naive_reference`).

The oracle shares two things with the checker: the effect registry's grounding functions and the
spec instantiation (substituting resources for variables). A bug there would affect both sides
equally; both are small, deterministic data transformations with their own unit tests.
