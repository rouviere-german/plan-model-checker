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

1. **Effect declarations are trusted.** If `snapshot_dir` declares `backup_ok[path] := true` and
   the real tool silently fails, the static verdict is about a world that did not happen. This is
   the reason for the runtime monitor, which applies the *observed* effects and blocks the next
   step that would violate a spec (the demo shows exactly this). The monitor only sees what tools
   report and what the sensor re-reads; an effect that is neither reported nor sensed is
   invisible to both layers.
2. **Events are trusted as calls, not as effects.** `drain[s]` holds because `drain_service` was
   called, whether or not it drained anything. Specs that need an effect should name the fluent
   (`drained[s]`), as `drain_before_restart` does; event-only specs cannot detect a no-op tool.
3. **Active-domain quantification.** `G(end -> !lock_held[r])` is checked for every `r` the plan
   mentions (in tool arguments or guards), not for every resource that exists. A spec that
   requires something of resources the plan never mentions is vacuous for them.
4. **Symbolic collection items are assumed fresh and distinct.** The checker treats
   `stale_dirs#0` as different from `/srv/logs` and from `other#0`. The executor enforces this at
   run start: a collection with duplicates, items that equal a plan or spec constant, items shared
   with another collection, or more items than the bound aborts the run before any call.
5. **Only safety properties.** Specs are positive combinations of `G(past-time formula)`. Nothing
   of the form "eventually X" (liveness), and no data-flow or confidentiality property (see
   taint-flow-agent). `G(end -> X)` expresses "X holds when the plan ends", which covers the usual
   release/undrain obligations for terminating plans.
6. **Bounded loops only.** Loops are unrolled to their declared bound; unbounded iteration is not
   expressible in the IR.
7. **The runtime monitor blocks steps, not plan ends.** A violation that only appears at `end`
   (a lock still held because `release_lock` was a no-op) is detected when the run finishes and
   the run is marked `violated`; nothing can be "blocked" at that point.
8. **Overrides are explicit.** An approved override waives named spec instances for one plan
   hash; the monitor still evaluates them and records each waived violation in the audit log.

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

The oracle shares two things with the checker: the effect registry's grounding functions and the
spec instantiation (substituting resources for variables). A bug there would affect both sides
equally; both are small, deterministic data transformations with their own unit tests.
