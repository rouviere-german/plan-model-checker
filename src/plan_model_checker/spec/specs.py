"""Typed, compiled spec sets and their instantiation over a plan's resources.

A spec is a *positive boolean combination of* ``G(ψ)`` *terms with ψ pure past-time*. For a
finite trace of length n, ``(w, 0) ⊨ G ψ`` iff ``(w, n-1) ⊨ H ψ``; replacing each top-level G by
H therefore gives a pure past-time formula whose value at the last position is the verdict, and
which can be evaluated incrementally. Because every term is monotone (once ``H ψ`` is false it
stays false) and they are combined with & and | only, a spec that is false after some prefix is
false for every extension: violations are detected at the earliest step, which lets the runtime
monitor block a step before it runs.

Specs are parametric: free variables are universally quantified over the resources of their sort
that the plan mentions (active-domain semantics, as in parametric trace slicing; Chen & Roşu,
"Parametric Trace Slicing and Monitoring", TACAS 2009). Every proposition a spec mentions must be
declared in the effect registry, so a misspelled proposition is an error rather than a spec that
is silently always true.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from plan_model_checker.effects import EffectRegistry
from plan_model_checker.spec.ast import (
    Always,
    And,
    Bool,
    Const,
    Formula,
    Historically,
    Or,
    Var,
    atoms,
    children,
    render,
    substitute,
)
from plan_model_checker.spec.parser import SpecDecl, parse_document, parse_formula

__all__ = [
    "MAX_SPEC_VARIABLES",
    "InstanceBudgetExceededError",
    "Spec",
    "SpecError",
    "SpecInstance",
    "SpecSet",
    "build_spec",
    "final_form",
    "infer_sorts",
]

MAX_SPEC_VARIABLES = 3


class SpecError(ValueError):
    """A spec is ill-typed, uses an undeclared proposition, or places G incorrectly."""


class InstanceBudgetExceededError(Exception):
    """Instantiating the specs over the plan's resources would exceed the instance budget."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"more than {limit} spec instances")
        self.limit = limit


def _contains_always(formula: Formula) -> bool:
    stack = [formula]
    while stack:
        node = stack.pop()
        if isinstance(node, Always):
            return True
        stack.extend(children(node))
    return False


def _check_placement(formula: Formula) -> None:
    stack = [formula]
    while stack:
        node = stack.pop()
        if isinstance(node, And | Or):
            stack.extend((node.left, node.right))
        elif isinstance(node, Always):
            if _contains_always(node.arg):
                raise SpecError("G may not be nested inside another G")
        elif isinstance(node, Bool):
            continue
        else:
            raise SpecError(
                "a spec must be G(<past-time formula>), or a combination of such terms with & "
                "and |; the only future operator allowed is a top-level G"
            )


def final_form(formula: Formula) -> Formula:
    """Replace each top-level ``G ψ`` by ``H ψ``: the formula to evaluate at the last position."""
    if isinstance(formula, Always):
        return Historically(formula.arg)
    if isinstance(formula, And):
        return And(final_form(formula.left), final_form(formula.right))
    if isinstance(formula, Or):
        return Or(final_form(formula.left), final_form(formula.right))
    return formula


def infer_sorts(formula: Formula, registry: EffectRegistry) -> dict[str, str]:
    """Map each variable to its resource sort, checking every atom against the registry."""
    sorts: dict[str, str] = {}
    for atom in atoms(formula):
        if not registry.has_prop(atom.name):
            raise SpecError(f"proposition {atom.name} is not declared in the effect registry")
        decl = registry.prop(atom.name)
        if len(atom.args) != len(decl.params):
            raise SpecError(
                f"proposition {atom.name} takes {len(decl.params)} argument(s), "
                f"got {len(atom.args)}"
            )
        for position, term in enumerate(atom.args):
            if not isinstance(term, Var):
                continue
            expected = decl.params[position]
            previous = sorts.setdefault(term.name, expected)
            if previous != expected:
                raise SpecError(
                    f"variable {term.name} is used with sorts {previous} and {expected}"
                )
    if len(sorts) > MAX_SPEC_VARIABLES:
        raise SpecError(f"a spec may use at most {MAX_SPEC_VARIABLES} variables")
    return sorts


@dataclass(frozen=True, slots=True)
class Spec:
    name: str
    formula: Formula
    var_sorts: tuple[tuple[str, str], ...]
    description: str = ""
    implicit: bool = False

    @property
    def text(self) -> str:
        return render(self.formula)

    @property
    def constants(self) -> frozenset[str]:
        return frozenset(
            term.value
            for atom in atoms(self.formula)
            for term in atom.args
            if isinstance(term, Const)
        )


def build_spec(
    name: str,
    formula: Formula,
    registry: EffectRegistry,
    *,
    description: str = "",
    implicit: bool = False,
) -> Spec:
    _check_placement(formula)
    sorts = infer_sorts(formula, registry)
    return Spec(
        name=name,
        formula=formula,
        var_sorts=tuple(sorted(sorts.items())),
        description=description,
        implicit=implicit,
    )


@dataclass(frozen=True, slots=True)
class SpecInstance:
    """A spec with its variables bound to concrete resources."""

    spec: str
    binding: tuple[tuple[str, str], ...]
    source: Formula
    final: Formula

    @property
    def key(self) -> str:
        if not self.binding:
            return self.spec
        return f"{self.spec}[{', '.join(f'{var}={value}' for var, value in self.binding)}]"

    def to_json(self) -> dict[str, Any]:
        return {"spec": self.spec, "binding": dict(self.binding), "formula": render(self.source)}


@dataclass(frozen=True)
class SpecSet:
    specs: tuple[Spec, ...]
    _by_name: dict[str, Spec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name: dict[str, Spec] = {}
        for spec in self.specs:
            if spec.name in by_name:
                raise SpecError(f"spec {spec.name} is declared twice")
            by_name[spec.name] = spec
        object.__setattr__(self, "_by_name", by_name)

    @classmethod
    def from_decls(
        cls,
        decls: Iterable[SpecDecl],
        registry: EffectRegistry,
        *,
        include_implicit: bool = True,
    ) -> SpecSet:
        specs = [
            build_spec(decl.name, decl.formula, registry, description=decl.description)
            for decl in decls
        ]
        if include_implicit:
            for name, text in registry.implicit_specs():
                specs.append(
                    build_spec(
                        name,
                        parse_formula(text),
                        registry,
                        description="implicit: destructive tool needs its approval proposition",
                        implicit=True,
                    )
                )
        return cls(tuple(specs))

    @classmethod
    def parse(
        cls, text: str, registry: EffectRegistry, *, include_implicit: bool = True
    ) -> SpecSet:
        return cls.from_decls(parse_document(text), registry, include_implicit=include_implicit)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def get(self, name: str) -> Spec:
        spec = self._by_name.get(name)
        if spec is None:
            raise SpecError(f"unknown spec {name}")
        return spec

    def select(self, names: Sequence[str] | None) -> SpecSet:
        """A subset by name. Implicit (approval) specs are always kept."""
        if names is None:
            return self
        wanted = {self.get(name).name for name in names}
        return SpecSet(tuple(s for s in self.specs if s.name in wanted or s.implicit))

    def constants(self) -> frozenset[str]:
        found: set[str] = set()
        for spec in self.specs:
            found |= spec.constants
        return frozenset(found)

    def instances(
        self, universe: Mapping[str, Sequence[str]], *, max_instances: int = 10_000
    ) -> list[SpecInstance]:
        """Ground every spec over the resources of each variable's sort (active domain)."""
        result: list[SpecInstance] = []
        for spec in self.specs:
            names = [var for var, _ in spec.var_sorts]
            domains = [sorted(set(universe.get(sort, ()))) for _, sort in spec.var_sorts]
            for values in itertools.product(*domains):
                if len(result) >= max_instances:
                    raise InstanceBudgetExceededError(max_instances)
                result.append(_instance(spec, tuple(zip(names, values, strict=True))))
        return result

    def new_instances(
        self,
        universe: Mapping[str, Collection[str]],
        added: Mapping[str, Iterable[str]],
        *,
        max_instances: int = 10_000,
    ) -> list[SpecInstance]:
        """The instances over ``universe`` whose binding uses at least one ``added`` resource.

        ``universe`` already includes ``added``. Each such binding is produced exactly once: it
        is enumerated at the first variable that takes a new value (earlier variables range over
        the old resources only). This lets the runtime monitor grow its active domain one call at
        a time at a cost proportional to the new instances, instead of re-enumerating all of
        them for every new resource.
        """
        result: list[SpecInstance] = []
        fresh_by_sort = {sort: sorted(set(values)) for sort, values in added.items()}
        for spec in self.specs:
            names = [var for var, _ in spec.var_sorts]
            sorts = [sort for _, sort in spec.var_sorts]
            new = [fresh_by_sort.get(sort, []) for sort in sorts]
            if not any(new):
                continue
            # Only specs with several variables need the old and full domains (sorting them is
            # what would make every registration cost as much as the whole universe).
            full = [sorted(universe.get(sort, ())) if len(sorts) > 1 else [] for sort in sorts]
            old = [
                [value for value in domain if value not in set(fresh)]
                for domain, fresh in zip(full, new, strict=True)
            ]
            for first, fresh in enumerate(new):
                if not fresh:
                    continue
                domains = [*old[:first], fresh, *full[first + 1 :]]
                for values in itertools.product(*domains):
                    if len(result) >= max_instances:
                        raise InstanceBudgetExceededError(max_instances)
                    result.append(_instance(spec, tuple(zip(names, values, strict=True))))
        return result


def _instance(spec: Spec, binding: tuple[tuple[str, str], ...]) -> SpecInstance:
    source = substitute(spec.formula, dict(binding))
    return SpecInstance(spec=spec.name, binding=binding, source=source, final=final_form(source))
