# 3. Explicit-state 0-1 BFS with memoization instead of symbolic model checking

- Status: accepted
- Date: 2026-09-22

## Context

A plan unrolls to a control-flow graph with nondeterministic choice points (opaque branches,
tool outcomes, collection sizes). The number of paths is exponential in the number of choice
points: a plan with 32 approval requests has 2^32 paths. Three families of techniques were
considered:

1. **Path enumeration** (what the brute-force oracle does): simple and obviously correct, but
   exponential; unusable beyond toy plans.
2. **Symbolic model checking** (BDDs, or SAT/SMT-based bounded model checking with a solver such
   as Z3): scales to very large state spaces, but adds a heavy native dependency, makes shortest
   counterexamples an extra optimization problem, and makes the implementation much harder to
   audit in a portfolio-sized codebase.
3. **Explicit-state search over (control point, tracked fluents, monitor state)** with a visited
   set: every path that reaches an already-seen state is pruned, so the cost is bounded by the
   number of distinct states rather than paths.

For plans an agent writes (tens of calls, a handful of resources), the number of distinct states
per spec instance is small because only the fluents that the instance or a branch guard reads
are tracked (cone of influence) and violated monitor states collapse into one sink.

## Decision

Explicit-state search, one search per spec instance, with:

- 0-1 BFS: choice edges cost 0 and tool calls cost 1, so the first violating end state popped is
  a shortest violating trace in number of calls;
- memoization on `(pc, fluent bits, monitor state)`;
- cone-of-influence projection of fluents per instance;
- budgets on states, instances and collection-size assignments that raise instead of answering.

## Consequences

- Shortest counterexamples come for free and are small enough to put into a repair prompt.
- Measured on the development machine (README Results): 60 corpus plans at p95 2.0 ms per check;
  a chain of 32 guarded deletions (128 call sites, 2^32 paths) explores 43,072 states in about
  0.32 s; 32 opaque diamonds (2^32 paths) explore 99 states.
- Cost grows with the number of spec instances (resources times specs): the delete chain grows
  roughly quadratically in n because every new directory adds instances *and* lengthens every
  instance's search. That is the scaling limit we accept; a plan with thousands of resources
  would need per-resource slicing of the control-flow graph or a symbolic engine.
- The implementation is plain Python, deterministic, and differential-tested against path
  enumeration on 5,000 generated cases per CI run.
