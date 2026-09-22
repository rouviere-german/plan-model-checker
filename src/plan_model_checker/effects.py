"""Effect declarations: the abstract interpretation of tools over finite propositions.

Every tool declares:

- typed parameters (each parameter has a resource *sort* such as ``path`` or ``service``);
- the *event* a call emits (an event proposition over its parameters, true only at the call's
  trace position);
- one or more *outcomes*, each a list of fluent updates. Several outcomes make the tool
  nondeterministic for the checker, which explores all of them;
- for destructive tools, an *approval proposition* that must hold immediately before the call.
  The registry turns every such requirement into an implicit spec, so no spec set can forget it.

Guarantees of the checker hold relative to these declarations (see docs/SOUNDNESS.md); the
runtime monitor covers the gap when a tool's real behaviour diverges from what it declares.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from plan_model_checker.core.canonical import canonical_json, sha256_hex
from plan_model_checker.ir import IDENT_PATTERN, TOOL_PATTERN
from plan_model_checker.props import RESERVED_PROPS, GroundProp

__all__ = [
    "ApprovalDecl",
    "EffectDecl",
    "EffectRegistry",
    "EffectRegistryModel",
    "Establisher",
    "EventDecl",
    "GroundOutcome",
    "OutcomeDecl",
    "ParamDecl",
    "PropDecl",
    "RegistryError",
    "ToolEffects",
    "UnknownToolError",
]


class RegistryError(ValueError):
    """The effect declarations are inconsistent (unknown sort, arity or kind mismatch...)."""


class UnknownToolError(RegistryError):
    """A plan or a runtime call names a tool that is not in the signed registry."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PropDecl(_Frozen):
    name: str = Field(pattern=IDENT_PATTERN)
    kind: Literal["fluent", "event"]
    params: tuple[str, ...] = Field(default=(), max_length=3)
    initial: bool = False
    description: str = Field(default="", max_length=300)


class ParamDecl(_Frozen):
    name: str = Field(pattern=IDENT_PATTERN)
    sort: str = Field(pattern=IDENT_PATTERN)


class EffectDecl(_Frozen):
    prop: str = Field(pattern=IDENT_PATTERN)
    args: tuple[str, ...] = Field(default=(), max_length=3)
    value: bool


class OutcomeDecl(_Frozen):
    name: str = Field(pattern=IDENT_PATTERN)
    effects: tuple[EffectDecl, ...] = ()


class EventDecl(_Frozen):
    prop: str = Field(pattern=IDENT_PATTERN)
    args: tuple[str, ...] = Field(default=(), max_length=3)


class ApprovalDecl(_Frozen):
    prop: str = Field(pattern=IDENT_PATTERN)
    args: tuple[str, ...] = Field(default=(), max_length=3)


class ToolEffects(_Frozen):
    name: str = Field(pattern=TOOL_PATTERN)
    description: str = Field(default="", max_length=500)
    params: tuple[ParamDecl, ...] = Field(default=(), max_length=6)
    event: EventDecl
    outcomes: tuple[OutcomeDecl, ...] = Field(min_length=1, max_length=4)
    destructive: bool = False
    approval: ApprovalDecl | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ToolEffects:
        names = [param.name for param in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"tool {self.name}: duplicate parameter names")
        outcome_names = [outcome.name for outcome in self.outcomes]
        if len(outcome_names) != len(set(outcome_names)):
            raise ValueError(f"tool {self.name}: duplicate outcome names")
        if self.destructive and self.approval is None:
            raise ValueError(f"tool {self.name}: destructive tools must declare an approval")
        return self


class EffectRegistryModel(_Frozen):
    version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    sorts: tuple[str, ...] = Field(min_length=1, max_length=16)
    props: tuple[PropDecl, ...] = Field(max_length=128)
    tools: tuple[ToolEffects, ...] = Field(min_length=1, max_length=128)


@dataclass(frozen=True, slots=True)
class GroundOutcome:
    name: str
    updates: tuple[tuple[GroundProp, bool], ...]


@dataclass(frozen=True, slots=True)
class Establisher:
    """A tool that can make ``prop`` take ``value`` (for repair suggestions).

    ``param_for_arg[i]`` names the tool parameter that fills argument ``i`` of the proposition.
    ``always`` is True when every declared outcome establishes it.
    """

    tool: str
    param_for_arg: tuple[str, ...]
    always: bool


_BUILTIN_EVENTS = (
    PropDecl(name="start", kind="event", description="holds only at the first position"),
    PropDecl(name="end", kind="event", description="holds only at the final position"),
)


class EffectRegistry:
    """Validated, indexed, immutable effect declarations."""

    def __init__(self, model: EffectRegistryModel) -> None:
        self._model = model
        self._sha256 = sha256_hex(canonical_json(model.model_dump(mode="json")))
        sorts = set(model.sorts)
        if len(sorts) != len(model.sorts):
            raise RegistryError("duplicate sort names")
        props: dict[str, PropDecl] = {decl.name: decl for decl in _BUILTIN_EVENTS}
        for decl in model.props:
            if decl.name in RESERVED_PROPS or decl.name in props:
                raise RegistryError(f"proposition {decl.name} is reserved or duplicated")
            unknown = [sort for sort in decl.params if sort not in sorts]
            if unknown:
                raise RegistryError(f"proposition {decl.name}: unknown sorts {unknown}")
            if decl.kind == "event" and decl.initial:
                raise RegistryError(f"event {decl.name} cannot have an initial value")
            props[decl.name] = decl
        self._props = props
        tools: dict[str, ToolEffects] = {}
        for tool in model.tools:
            if tool.name in tools:
                raise RegistryError(f"tool {tool.name} is declared twice")
            self._check_tool(tool, sorts)
            tools[tool.name] = tool
        self._tools = tools
        self._check_destructive_events()
        self._tool_hashes = {
            name: sha256_hex(canonical_json(tool.model_dump(mode="json")))
            for name, tool in tools.items()
        }

    # --- validation -------------------------------------------------------------------------

    def _check_ref(
        self,
        tool: ToolEffects,
        prop_name: str,
        args: tuple[str, ...],
        kind: Literal["fluent", "event"],
        what: str,
    ) -> None:
        decl = self._props.get(prop_name)
        if decl is None or decl.kind != kind:
            raise RegistryError(f"tool {tool.name}: {what} {prop_name} is not a declared {kind}")
        if len(args) != len(decl.params):
            raise RegistryError(f"tool {tool.name}: {what} {prop_name} has the wrong arity")
        params = {param.name: param.sort for param in tool.params}
        for position, arg in enumerate(args):
            if arg not in params:
                raise RegistryError(f"tool {tool.name}: {what} uses unknown parameter {arg}")
            if params[arg] != decl.params[position]:
                raise RegistryError(
                    f"tool {tool.name}: {what} {prop_name} argument {position} expects sort "
                    f"{decl.params[position]}, parameter {arg} has sort {params[arg]}"
                )

    def _check_tool(self, tool: ToolEffects, sorts: set[str]) -> None:
        for param in tool.params:
            if param.sort not in sorts:
                raise RegistryError(f"tool {tool.name}: unknown sort {param.sort}")
        if tool.event.prop in {"start", "end"}:
            raise RegistryError(f"tool {tool.name}: start/end are reserved events")
        self._check_ref(tool, tool.event.prop, tool.event.args, "event", "event")
        for outcome in tool.outcomes:
            for effect in outcome.effects:
                self._check_ref(tool, effect.prop, effect.args, "fluent", "effect")
        if tool.approval is not None:
            self._check_ref(tool, tool.approval.prop, tool.approval.args, "fluent", "approval")

    def _check_destructive_events(self) -> None:
        owners: dict[str, list[ToolEffects]] = {}
        for tool in self._tools.values():
            owners.setdefault(tool.event.prop, []).append(tool)
        for event, tools in owners.items():
            if any(tool.destructive for tool in tools) and len(tools) > 1:
                raise RegistryError(
                    f"event {event} of a destructive tool must not be shared with other tools"
                )

    # --- lookup -----------------------------------------------------------------------------

    @property
    def model(self) -> EffectRegistryModel:
        return self._model

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def name(self) -> str:
        return self._model.name

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def tool_sha256(self, name: str) -> str:
        self.tool(name)
        return self._tool_hashes[name]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def tool(self, name: str) -> ToolEffects:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(f"tool {name} is not in the effect registry")
        return tool

    def prop(self, name: str) -> PropDecl:
        decl = self._props.get(name)
        if decl is None:
            raise RegistryError(f"proposition {name} is not declared")
        return decl

    def has_prop(self, name: str) -> bool:
        return name in self._props

    def props(self) -> tuple[PropDecl, ...]:
        return tuple(self._props.values())

    def param_sorts(self, tool_name: str) -> dict[str, str]:
        return {param.name: param.sort for param in self.tool(tool_name).params}

    def initial(self, prop: GroundProp) -> bool:
        decl = self._props.get(prop.name)
        return bool(decl is not None and decl.kind == "fluent" and decl.initial)

    # --- grounding --------------------------------------------------------------------------

    @staticmethod
    def _ground(name: str, params: tuple[str, ...], args: Mapping[str, str]) -> GroundProp:
        return GroundProp(name, tuple(args[param] for param in params))

    def ground_event(self, tool_name: str, args: Mapping[str, str]) -> GroundProp:
        tool = self.tool(tool_name)
        return self._ground(tool.event.prop, tool.event.args, args)

    def ground_outcomes(self, tool_name: str, args: Mapping[str, str]) -> tuple[GroundOutcome, ...]:
        tool = self.tool(tool_name)
        return tuple(
            GroundOutcome(
                name=outcome.name,
                updates=tuple(
                    (self._ground(effect.prop, effect.args, args), effect.value)
                    for effect in outcome.effects
                ),
            )
            for outcome in tool.outcomes
        )

    def ground_approval(self, tool_name: str, args: Mapping[str, str]) -> GroundProp | None:
        approval = self.tool(tool_name).approval
        if approval is None:
            return None
        return self._ground(approval.prop, approval.args, args)

    # --- derived knowledge ------------------------------------------------------------------

    def implicit_specs(self) -> tuple[tuple[str, str], ...]:
        """``(name, formula)`` for every destructive tool: its approval must hold just before."""
        specs: list[tuple[str, str]] = []
        for tool in sorted(self._tools.values(), key=lambda item: item.name):
            if not tool.destructive or tool.approval is None:
                continue
            event = _render(tool.event.prop, tool.event.args)
            approval = _render(tool.approval.prop, tool.approval.args)
            specs.append((f"approval_{tool.name}", f"G({event} -> Y {approval})"))
        return tuple(specs)

    def establishers(self, prop_name: str, value: bool) -> tuple[Establisher, ...]:
        """Tools whose event is ``prop_name`` (value True) or whose effects set it to ``value``."""
        found: list[Establisher] = []
        for tool in sorted(self._tools.values(), key=lambda item: item.name):
            if tool.event.prop == prop_name and value:
                found.append(Establisher(tool.name, tool.event.args, always=True))
                continue
            mappings: list[tuple[str, ...]] = []
            for outcome in tool.outcomes:
                for effect in outcome.effects:
                    if effect.prop == prop_name and effect.value == value:
                        mappings.append(effect.args)
            if mappings:
                always = len(mappings) >= len(tool.outcomes) and all(
                    any(
                        effect.prop == prop_name and effect.value == value
                        for effect in outcome.effects
                    )
                    for outcome in tool.outcomes
                )
                found.append(Establisher(tool.name, mappings[0], always=always))
        found.sort(key=lambda item: (not item.always, item.tool))
        return tuple(found)


def _render(prop: str, args: tuple[str, ...]) -> str:
    return f"{prop}[{', '.join(args)}]" if args else prop
