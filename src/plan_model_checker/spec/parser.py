"""Lark (LALR) parser for spec formulas and spec documents.

Precedence, loosest first: ``<->``, ``->`` (right associative), ``|``, ``&``, ``S`` (right
associative), then the prefix operators ``! Y Z O H G``. Propositions are lowercase identifiers
with optional arguments in brackets; bare lowercase arguments are variables and double-quoted ones
are resource constants: ``G(delete[r] -> Y backup_ok[r])``, ``G(restart["svc:api"] -> Y
drained["svc:api"])``.

A spec document is a sequence of ``spec <name> ["description"]: <formula>;`` declarations with
``#`` comments.

Input size and nesting are bounded (DoS protection): text longer than ``MAX_FORMULA_CHARS`` or
formulas deeper than ``MAX_FORMULA_DEPTH`` are rejected with :class:`SpecSyntaxError`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError, UnexpectedInput, VisitError

from plan_model_checker.ir import is_valid_resource
from plan_model_checker.spec.ast import (
    Always,
    And,
    Atom,
    Bool,
    Const,
    Formula,
    Historically,
    Iff,
    Implies,
    Not,
    Once,
    Or,
    Since,
    Term,
    Var,
    WeakYesterday,
    Yesterday,
    depth,
    render,
    size,
)

__all__ = [
    "MAX_DOCUMENT_CHARS",
    "MAX_FORMULA_CHARS",
    "MAX_FORMULA_DEPTH",
    "MAX_FORMULA_SIZE",
    "SpecDecl",
    "SpecSyntaxError",
    "parse_document",
    "parse_formula",
]

MAX_FORMULA_CHARS = 2_000
MAX_DOCUMENT_CHARS = 100_000
MAX_FORMULA_DEPTH = 32
MAX_FORMULA_SIZE = 256

_GRAMMAR = r"""
?formula: iff_

?iff_: impl_
     | impl_ "<->" impl_              -> iff

?impl_: or_
      | or_ "->" impl_                -> implies

?or_: and_
    | or_ "|" and_                    -> or_op

?and_: since_
     | and_ "&" since_                -> and_op

?since_: unary_
       | unary_ "S" since_            -> since

?unary_: atom_
       | "!" unary_                   -> not_op
       | "Y" unary_                   -> yesterday
       | "Z" unary_                   -> weak_yesterday
       | "O" unary_                   -> once
       | "H" unary_                   -> historically
       | "G" unary_                   -> always

?atom_: "true"                        -> true_
      | "false"                       -> false_
      | NAME "[" term ("," term)* "]" -> atom
      | NAME                          -> atom
      | "(" formula ")"

term: NAME                            -> var
    | ESCAPED_STRING                  -> const

document: spec_decl*
spec_decl: "spec" NAME [ESCAPED_STRING] ":" formula ";"

NAME: /[a-z][a-z0-9_]*/
COMMENT: /#[^\n]*/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
%ignore COMMENT
"""

_SPEC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SpecSyntaxError(ValueError):
    """The spec text does not parse, or exceeds a size limit."""


@dataclass(frozen=True, slots=True)
class SpecDecl:
    name: str
    description: str
    formula: Formula
    text: str


def _unquote(token: Token) -> str:
    value = json.loads(str(token))
    if not isinstance(value, str):  # pragma: no cover - ESCAPED_STRING always decodes to str
        raise SpecSyntaxError("constant must be a string")
    return value


@v_args(inline=True)
class _ToAst(Transformer[Token, Any]):
    def iff(self, left: Formula, right: Formula) -> Formula:
        return Iff(left, right)

    def implies(self, left: Formula, right: Formula) -> Formula:
        return Implies(left, right)

    def or_op(self, left: Formula, right: Formula) -> Formula:
        return Or(left, right)

    def and_op(self, left: Formula, right: Formula) -> Formula:
        return And(left, right)

    def since(self, left: Formula, right: Formula) -> Formula:
        return Since(left, right)

    def not_op(self, arg: Formula) -> Formula:
        return Not(arg)

    def yesterday(self, arg: Formula) -> Formula:
        return Yesterday(arg)

    def weak_yesterday(self, arg: Formula) -> Formula:
        return WeakYesterday(arg)

    def once(self, arg: Formula) -> Formula:
        return Once(arg)

    def historically(self, arg: Formula) -> Formula:
        return Historically(arg)

    def always(self, arg: Formula) -> Formula:
        return Always(arg)

    def true_(self) -> Formula:
        return Bool(True)

    def false_(self) -> Formula:
        return Bool(False)

    def atom(self, name: Token, *terms: Term) -> Formula:
        return Atom(str(name), tuple(terms))

    def var(self, name: Token) -> Term:
        return Var(str(name))

    def const(self, token: Token) -> Term:
        value = _unquote(token)
        if not is_valid_resource(value):
            raise SpecSyntaxError(f"constant {value!r} is not a valid resource name")
        return Const(value)

    def spec_decl(self, name: Token, description: Token | None, formula: Formula) -> SpecDecl:
        return SpecDecl(
            name=str(name),
            description=_unquote(description) if description is not None else "",
            formula=formula,
            text=render(formula),
        )

    def document(self, *decls: SpecDecl) -> list[SpecDecl]:
        return list(decls)


@lru_cache(maxsize=1)
def _parser() -> Lark:
    return Lark(_GRAMMAR, start=["formula", "document"], parser="lalr", maybe_placeholders=True)


def _check_bounds(formula: Formula) -> Formula:
    if size(formula) > MAX_FORMULA_SIZE:
        raise SpecSyntaxError(f"formula has more than {MAX_FORMULA_SIZE} nodes")
    if depth(formula) > MAX_FORMULA_DEPTH:
        raise SpecSyntaxError(f"formula is nested deeper than {MAX_FORMULA_DEPTH}")
    return formula


def _transform(text: str, start: str) -> Any:
    try:
        tree = _parser().parse(text, start=start)
        return _ToAst().transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, SpecSyntaxError):
            raise exc.orig_exc from exc
        raise SpecSyntaxError("spec could not be parsed") from exc
    except UnexpectedInput as exc:
        raise SpecSyntaxError(f"syntax error at line {exc.line}, column {exc.column}") from exc
    except LarkError as exc:
        raise SpecSyntaxError("spec could not be parsed") from exc
    except RecursionError as exc:
        raise SpecSyntaxError("formula is nested too deeply") from exc


def parse_formula(text: str) -> Formula:
    if len(text) > MAX_FORMULA_CHARS:
        raise SpecSyntaxError(f"formula longer than {MAX_FORMULA_CHARS} characters")
    result = _transform(text, "formula")
    if not isinstance(result, Formula):  # pragma: no cover - the grammar only produces formulas
        raise SpecSyntaxError("spec could not be parsed")
    return _check_bounds(result)


def parse_document(text: str) -> list[SpecDecl]:
    if len(text) > MAX_DOCUMENT_CHARS:
        raise SpecSyntaxError(f"spec document longer than {MAX_DOCUMENT_CHARS} characters")
    decls = _transform(text, "document")
    seen: set[str] = set()
    result: list[SpecDecl] = []
    for decl in decls:
        if not _SPEC_NAME.fullmatch(decl.name):
            raise SpecSyntaxError(f"invalid spec name {decl.name!r}")
        if decl.name in seen:
            raise SpecSyntaxError(f"spec {decl.name} is declared twice")
        seen.add(decl.name)
        _check_bounds(decl.formula)
        result.append(decl)
    return result
