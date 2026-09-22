# 2. Past-time LTL under a top-level G, not full LTL on finite traces

- Status: accepted
- Date: 2026-09-22

## Context

Resource-protocol rules ("back up before deleting", "release every lock before the end", "check
status before retrying") need a temporal logic over finite traces. The two common choices are:

- **Full LTLf** (De Giacomo & Vardi, IJCAI 2013): future operators (X, F, G, U) over finite
  traces. Checking needs a translation to automata (LTLf to DFA is worst-case doubly exponential)
  and a product construction; runtime enforcement of liveness-like formulas cannot block a step,
  because "eventually" can always still be satisfied later.
- **Past-time LTL** (Manna & Pnueli; Havelund & Roşu, TACAS 2002): Y, Z, O, H, S. A safety
  property is written `G ψ` with `ψ` past-time. The value of every subformula at a position is a
  function of the current letter and the previous position's values, so a monitor is a
  bit-vector updated once per step, compiled in linear time from the formula.

The rules we need are safety rules whose natural reading is "at the moment X happens, the past
must look like Y". End-of-plan obligations become safety rules too, because the trace has an
explicit final position marked by the `end` event: `G(end -> !lock_held[r])`.

## Decision

Specs are positive boolean combinations (`&`, `|`) of `G ψ` terms with `ψ` pure past-time. G may
not appear under a temporal operator, a negation or the left side of an implication; the parser
and type checker reject anything else. Checking and monitoring use `H ψ` evaluated at the last
position, which is equivalent on finite traces.

## Consequences

- One compiled artifact serves both layers: the checker steps the same `CompiledMonitor` along
  control-flow paths that the runtime monitor steps along real events. There is no automaton
  construction and no determinization.
- Monitor state is the vector of subformula bits read through `pre[]`; explicit-state search over
  `(control point, fluents, monitor state)` terminates and memoizes well.
- Every allowed spec is monotone: once false after a prefix, it is false for every extension.
  The runtime monitor can therefore block the first step that would falsify a spec, and the
  checker can collapse violated monitor states into one sink.
- We give up liveness ("every deletion is eventually followed by a notification"), which a
  plan checker for terminating plans can mostly express with `G(end -> O notify[r])`-style
  rules anyway, and anything that genuinely needs future operators. This is listed in the README
  non-goals.
- Correctness of the G-to-H rewrite and of the incremental compilation is tested against direct
  implementations of the textbook definitions, including the future-time G
  (`tests/unit/test_spec.py`, `tests/unit/test_differential.py`).
