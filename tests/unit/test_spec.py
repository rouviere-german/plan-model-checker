from __future__ import annotations

import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from plan_model_checker.effects import EffectRegistry
from plan_model_checker.gen import diff_registry, random_formula, random_spec
from plan_model_checker.props import END, START, GroundProp
from plan_model_checker.spec.ast import (
    Always,
    And,
    Atom,
    Const,
    Formula,
    Historically,
    Implies,
    Not,
    Or,
    Since,
    Var,
    Yesterday,
    atoms,
    depth,
    ground_prop,
    render,
    size,
    substitute,
    variables,
)
from plan_model_checker.spec.monitor import compile_monitor
from plan_model_checker.spec.parser import (
    MAX_FORMULA_CHARS,
    SpecSyntaxError,
    parse_document,
    parse_formula,
)
from plan_model_checker.spec.semantics import Evaluator, satisfies
from plan_model_checker.spec.specs import (
    InstanceBudgetExceededError,
    SpecError,
    SpecSet,
    build_spec,
    final_form,
    infer_sorts,
)

pytestmark = pytest.mark.unit

A = Atom("a", (Var("r"),))
P = Atom("p", (Var("r"),))


def test_precedence_and_associativity() -> None:
    assert parse_formula("G(delete[r] -> Y backup_ok[r])") == Always(
        Implies(Atom("delete", (Var("r"),)), Yesterday(Atom("backup_ok", (Var("r"),))))
    )
    assert parse_formula("a S b S c") == Since(Atom("a"), Since(Atom("b"), Atom("c")))
    assert parse_formula("!a S b") == Since(Not(Atom("a")), Atom("b"))
    assert parse_formula("a | b & c") == Or(Atom("a"), And(Atom("b"), Atom("c")))
    assert parse_formula("a -> b -> c") == Implies(Atom("a"), Implies(Atom("b"), Atom("c")))
    assert parse_formula('G(restart["svc:api"] -> H x)') == Always(
        Implies(Atom("restart", (Const("svc:api"),)), Historically(Atom("x")))
    )


@given(st.randoms(use_true_random=False))
def test_render_then_parse_is_the_identity(rng: random.Random) -> None:
    formula = random_spec(rng)
    assert parse_formula(render(formula)) == formula


@pytest.mark.parametrize(
    "text",
    ["G(", "a b", "A", "a[", "a[R]", 'x["has#hash"]', "Y", "a -> ", "spec x: a;"],
)
def test_syntax_errors_are_reported_without_crashing(text: str) -> None:
    with pytest.raises(SpecSyntaxError):
        parse_formula(text)


def test_size_and_nesting_limits_protect_the_parser() -> None:
    with pytest.raises(SpecSyntaxError, match="longer"):
        parse_formula("a & " * (MAX_FORMULA_CHARS // 4) + "a")
    with pytest.raises(SpecSyntaxError, match="nested"):
        parse_formula("!" * 900 + "a")
    with pytest.raises(SpecSyntaxError, match="nodes"):
        parse_formula(" & ".join(f"a{i}" for i in range(300)))


def test_spec_documents() -> None:
    decls = parse_document(
        '# comment\nspec one "first rule": G(a -> Y b);\nspec two: G(end -> !c);\n'
    )
    assert [(d.name, d.description) for d in decls] == [("one", "first rule"), ("two", "")]
    assert decls[0].text == "G(a -> Y b)"
    with pytest.raises(SpecSyntaxError, match="twice"):
        parse_document("spec a: G b; spec a: G c;")
    with pytest.raises(SpecSyntaxError):
        parse_document("spec a G b;")


def test_ast_helpers() -> None:
    formula = parse_formula('G(a[r] & b[s, "k"] -> O c)')
    assert variables(formula) == {"r", "s"}
    assert size(formula) == 7
    assert depth(formula) == 4
    ground = substitute(formula, {"r": "x", "s": "y"})
    assert variables(ground) == set()
    assert sorted(str(ground_prop(atom)) for atom in atoms(ground)) == ["a[x]", "b[y,k]", "c"]
    with pytest.raises(TypeError):
        ground_prop(Atom("a", (Var("r"),)))


def test_g_is_only_allowed_as_positive_top_level_combination(
    registry: EffectRegistry,
) -> None:
    for text in [
        "G(end -> !lock_held[r])",
        "G(delete[r]) & G(write[r])",
        "G(delete[r]) | G(write[r])",
        "true",
    ]:
        build_spec("ok", parse_formula(text), registry)
    for text in [
        "end -> !lock_held[r]",
        "!G(delete[r])",
        "G(delete[r]) -> G(backup[r])",
        "Y G(delete[r])",
        "G(G(delete[r]))",
    ]:
        with pytest.raises(SpecError):
            build_spec("bad", parse_formula(text), registry)


def test_unknown_propositions_arity_and_sort_clashes_are_errors(registry: EffectRegistry) -> None:
    with pytest.raises(SpecError, match="not declared"):
        infer_sorts(parse_formula("G(delet[r] -> Y backup_ok[r])"), registry)
    with pytest.raises(SpecError, match="argument"):
        infer_sorts(parse_formula("G(delete[r, r])"), registry)
    with pytest.raises(SpecError, match="sorts"):
        infer_sorts(parse_formula("G(delete[r] -> drained[r])"), registry)
    with pytest.raises(SpecError, match="at most"):
        infer_sorts(parse_formula("G(delete[a] & delete[b] & delete[c] & delete[d])"), registry)
    assert infer_sorts(parse_formula("G(restart[s] -> Y drained[s])"), registry) == {"s": "service"}


def test_instances_quantify_over_the_active_domain(registry: EffectRegistry) -> None:
    spec_set = SpecSet.parse(
        "spec pair: G(acquire[a] -> Y !lock_held[b]);\nspec none: G(end -> !drained[s]);",
        registry,
    )
    assert spec_set.names == ("pair", "none", "approval_delete_dir")
    instances = spec_set.instances({"path": ("/x", "/y")})
    keys = [instance.key for instance in instances]
    assert keys[:4] == [
        "pair[a=/x, b=/x]",
        "pair[a=/x, b=/y]",
        "pair[a=/y, b=/x]",
        "pair[a=/y, b=/y]",
    ]
    assert not any(key.startswith("none") for key in keys)  # no service in the plan
    with pytest.raises(InstanceBudgetExceededError):
        spec_set.instances({"path": ("/x", "/y")}, max_instances=3)
    selected = spec_set.select(["none"])
    assert selected.names == ("none", "approval_delete_dir")
    assert spec_set.select(None) is spec_set
    with pytest.raises(SpecError):
        spec_set.select(["missing"])
    with pytest.raises(SpecError, match="twice"):
        SpecSet((spec_set.specs[0], spec_set.specs[0]))
    assert instances[0].to_json()["binding"] == {"a": "/x", "b": "/x"}


def test_final_form_rewrites_only_top_level_g() -> None:
    formula = parse_formula("G(a -> Y b) & G(c) | G(d)")
    assert final_form(formula) == Or(
        And(Historically(parse_formula("a -> Y b")), Historically(Atom("c"))),
        Historically(Atom("d")),
    )


def _random_trace(rng: random.Random, props: list[GroundProp]) -> list[frozenset[GroundProp]]:
    length = rng.randint(1, 7)
    trace = []
    for position in range(length):
        letter = {prop for prop in props if rng.random() < 0.4}
        letter.discard(START)
        letter.discard(END)
        if position == 0:
            letter.add(START)
        if position == length - 1:
            letter.add(END)
        trace.append(frozenset(letter))
    return trace


@given(st.randoms(use_true_random=False))
def test_compiled_monitor_matches_the_reference_semantics_at_every_position(
    rng: random.Random,
) -> None:
    formula: Formula = random_formula(rng, depth=5, variables=())
    monitor = compile_monitor(formula)
    props = list(monitor.props) or [GroundProp("a", ("r1",))]
    trace = _random_trace(rng, props)
    evaluator = Evaluator(trace)
    bits = monitor.first(monitor.letter(trace[0]))
    assert monitor.holds(bits) == evaluator.holds(formula, 0)
    for position in range(1, len(trace)):
        bits = monitor.step(monitor.state(bits), monitor.letter(trace[position]))
        assert monitor.holds(bits) == evaluator.holds(formula, position)


@given(st.randoms(use_true_random=False))
def test_g_at_position_zero_equals_h_at_the_last_position(rng: random.Random) -> None:
    source = substitute(random_spec(rng), {"x": "r1", "y": "r2"})
    props = [GroundProp(n, (r,)) for n in ("p", "q", "a", "b") for r in ("r1", "r2")] + [
        GroundProp("s"),
        GroundProp("c"),
    ]
    trace = _random_trace(rng, props)
    evaluator = Evaluator(trace)
    assert satisfies(source, trace) == evaluator.holds(final_form(source), len(trace) - 1)


def test_monitor_refuses_g_and_the_evaluator_refuses_empty_traces() -> None:
    with pytest.raises(TypeError):
        compile_monitor(Always(Atom("a")))
    with pytest.raises(ValueError, match="at least one"):
        Evaluator([])


def test_synthetic_registry_supports_every_generated_spec() -> None:
    registry = diff_registry()
    rng = random.Random(3)
    for _ in range(50):
        build_spec("s", random_spec(rng), registry)
