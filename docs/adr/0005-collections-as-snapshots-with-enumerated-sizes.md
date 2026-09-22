# 5. Data-dependent loops: snapshot collections with enumerated sizes

- Status: accepted
- Date: 2026-09-22

## Context

Agents iterate over things they only learn at runtime ("every stale directory"). The plan IR
needs a loop whose item count is unknown statically but bounded. Options for the checker:

1. **Independent nondeterministic count per loop**, fresh symbolic items per loop instance.
   Simple, but two loops over the same list ("back up every stale directory, then delete every
   stale directory") would be checked as if they iterated over different directories, so the
   second loop's deletions could never see the first loop's backups: a false alarm on the most
   common safe pattern.
2. **Collections as values in the state** (count chosen at first use, stored in the search
   state). Precise, but complicates the state space and every consumer of the control-flow graph.
3. **Snapshot semantics with enumerated sizes**: a collection label denotes one list for the
   whole run; the checker enumerates every size assignment (0..bound per label) and builds one
   deterministic control-flow graph per assignment, with items `label#0 .. label#k-1` shared by
   every loop over that label.

## Decision

Option 3. Loops over the same label must declare the same bound (checked). Items are symbolic
resources assumed distinct from each other and from every constant in the plan and the specs.
At runtime the executor resolves every collection once, before the first call, and aborts the run
if a collection exceeds its bound, contains duplicates, or aliases a constant or another
collection's items.

## Consequences

- Backup-all-then-delete-all verifies; the corpus contains such plans.
- The number of control-flow graphs is the product of (bound + 1) over labels; it is capped
  (at most 3 labels and 256 assignments by default), and exceeding it raises.
- Nested loops over the same label see the same items on every outer iteration. A collection
  whose contents depend on the outer item (instances of each service) cannot be expressed; this
  is listed as a non-goal.
- The runtime checks close the gap between symbolic items and real ones; they are tested in
  `tests/unit/test_runtime.py::test_collections_are_bounded_distinct_and_fresh`.
