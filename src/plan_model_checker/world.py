"""An in-memory file-maintenance world and least-privilege mock tools for it.

Tools never report their effects from intent. Each one performs its action through a
:class:`ScopedWorld` that only exposes the operations that tool is allowed to use, and then the
world is *observed*: the tool reports the current value of every fluent its declaration mentions.
If a fault is injected (for example a snapshot that silently does nothing), the observation shows
it and the runtime monitor sees the real state, not the declared one.

Every tool is pinned to the hash of the declaration it was written against (taken from the
catalog, not from whatever registry is loaded at runtime), so a registry that changes a tool's
declared effects cannot be paired with an implementation written for the old ones.

Deletion is doubly gated: the plan must pass ``approved[path]`` (a proposition only
``request_approval`` can set, after a human approved that exact call in the approval store),
and the world itself refuses to delete a directory without a granted, unconsumed approval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from plan_model_checker.agentic.approval import ApprovalRecord, ApprovalStore
from plan_model_checker.agentic.providers.base import ToolUseBlock
from plan_model_checker.agentic.tools import Capability, ToolDefinition, ToolRegistry
from plan_model_checker.catalog import file_maintenance_registry
from plan_model_checker.effects import EffectRegistry
from plan_model_checker.props import GroundProp
from plan_model_checker.runtime import Observation

__all__ = [
    "TOOL_OPERATIONS",
    "FileMaintenanceWorld",
    "HumanReviewer",
    "ScopedWorld",
    "WorldTool",
    "build_tools",
]

# Least privilege: the world operations each tool may use.
TOOL_OPERATIONS: Mapping[str, frozenset[str]] = {
    "snapshot_dir": frozenset({"read_dir", "write_backup"}),
    "request_approval": frozenset({"approvals"}),
    "delete_dir": frozenset({"delete_dir"}),
    "write_file": frozenset({"write_dir"}),
    "acquire_lock": frozenset({"locks"}),
    "release_lock": frozenset({"locks"}),
    "compact_dir": frozenset({"write_dir"}),
    "archive_dir": frozenset({"archive"}),
    "drain_service": frozenset({"services"}),
    "undrain_service": frozenset({"services"}),
    "restart_service": frozenset({"services"}),
    "submit_job": frozenset({"jobs"}),
    "check_status": frozenset({"jobs"}),
    "retry_job": frozenset({"jobs"}),
}

HumanReviewer = Callable[[ApprovalRecord, ApprovalStore], None]


class _PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=128)


def _approval_registry() -> ToolRegistry:
    """The destructive call, as the template approval gate sees it (bound to its exact args)."""
    registry = ToolRegistry({Capability.WRITE_DATA})
    registry.register(
        ToolDefinition(
            name="delete_dir",
            description="Delete a directory.",
            args_model=_PathArgs,
            capabilities=frozenset({Capability.WRITE_DATA}),
            handler=lambda args: None,
        )
    )
    return registry


@dataclass
class FileMaintenanceWorld:
    dirs: dict[str, int] = field(default_factory=dict)
    services: set[str] = field(default_factory=set)
    collections: dict[str, list[str]] = field(default_factory=dict)
    opaque: dict[str, bool] = field(default_factory=dict)
    faults: dict[str, str] = field(default_factory=dict)
    approvals: ApprovalStore = field(default_factory=ApprovalStore)
    reviewer: HumanReviewer | None = None
    agent: str = "agent:maintenance"
    backups: dict[str, int] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    locks: set[str] = field(default_factory=set)
    archive: set[str] = field(default_factory=set)
    drained: set[str] = field(default_factory=set)
    submissions: dict[str, int] = field(default_factory=dict)
    checked: set[str] = field(default_factory=set)
    granted: set[str] = field(default_factory=set)
    restarts: dict[str, int] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def fault(self, tool: str, resource: str) -> str | None:
        return self.faults.get(f"{tool}:{resource}", self.faults.get(tool))

    def sense(self, prop: GroundProp) -> bool:  # noqa: PLR0911
        """Observe one fluent in the real world state."""
        target = prop.args[0] if prop.args else ""
        if prop.name == "backup_ok":
            return target in self.backups and self.backups[target] == self.dirs.get(target)
        if prop.name == "deleted":
            return target in self.deleted
        if prop.name == "approved":
            return target in self.granted
        if prop.name == "lock_held":
            return target in self.locks
        if prop.name == "archived":
            return target in self.archive
        if prop.name == "drained":
            return target in self.drained
        if prop.name == "submitted":
            return self.submissions.get(target, 0) > 0
        if prop.name == "status_checked":
            return target in self.checked
        raise KeyError(f"no sensor for {prop.name}")

    def collection(self, label: str) -> Sequence[str]:
        return tuple(self.collections.get(label, ()))

    def decide(self, label: str, path: str) -> bool:
        del path
        return self.opaque.get(label, False)


class ScopedWorld:
    """The only view of the world a tool gets: operations outside its allowlist raise."""

    def __init__(self, world: FileMaintenanceWorld, allowed: frozenset[str]) -> None:
        self._world = world
        self._allowed = allowed

    def _require(self, operation: str) -> FileMaintenanceWorld:
        if operation not in self._allowed:
            raise PermissionError(f"operation {operation} is not granted to this tool")
        return self._world

    def snapshot(self, path: str) -> None:
        world = self._require("read_dir")
        self._require("write_backup")
        if path in world.dirs:
            world.backups[path] = world.dirs[path]

    def write(self, path: str) -> None:
        world = self._require("write_dir")
        if path in world.deleted:
            raise FileNotFoundError(path)
        world.dirs[path] = world.dirs.get(path, 0) + 1

    def delete(self, path: str) -> None:
        world = self._require("delete_dir")
        if path not in world.granted:
            raise PermissionError(f"deleting {path} needs a granted approval")
        world.granted.discard(path)
        world.deleted.add(path)

    def lock(self, path: str, *, held: bool) -> None:
        world = self._require("locks")
        if held:
            world.locks.add(path)
        else:
            world.locks.discard(path)

    def archive(self, path: str) -> None:
        self._require("archive").archive.add(path)

    def service(self, name: str, action: str) -> None:
        world = self._require("services")
        if action == "drain":
            world.drained.add(name)
        elif action == "undrain":
            world.drained.discard(name)
        else:
            world.restarts[name] = world.restarts.get(name, 0) + 1

    def job(self, name: str, action: str) -> None:
        world = self._require("jobs")
        if action == "submit":
            world.submissions[name] = world.submissions.get(name, 0) + 1
            world.checked.discard(name)
        elif action == "check":
            world.checked.add(name)
        else:
            world.submissions[name] = world.submissions.get(name, 0) + 1
            world.checked.discard(name)

    def request_approval(self, path: str) -> bool:
        world = self._require("approvals")
        registry = _approval_registry()
        call = registry.validate_call(
            ToolUseBlock(id="approval", name="delete_dir", arguments={"path": path})
        )
        if world.approvals.consume(call.call_hash, principal=world.agent) is None:
            record = world.approvals.request(call, requested_by=world.agent)
            if world.reviewer is not None:
                world.reviewer(record, world.approvals)
            if world.approvals.consume(call.call_hash, principal=world.agent) is None:
                return False
        world.granted.add(path)
        return True


_Action = Callable[[ScopedWorld, Mapping[str, str]], str]


def _simple(method: Callable[[ScopedWorld, str], None], param: str) -> _Action:
    def action(scoped: ScopedWorld, args: Mapping[str, str]) -> str:
        method(scoped, args[param])
        return "ok"

    return action


def _approval(scoped: ScopedWorld, args: Mapping[str, str]) -> str:
    return "granted" if scoped.request_approval(args["path"]) else "pending"


_ACTIONS: Mapping[str, _Action] = {
    "snapshot_dir": _simple(ScopedWorld.snapshot, "path"),
    "request_approval": _approval,
    "delete_dir": _simple(ScopedWorld.delete, "path"),
    "write_file": _simple(ScopedWorld.write, "path"),
    "acquire_lock": _simple(lambda s, p: s.lock(p, held=True), "path"),
    "release_lock": _simple(lambda s, p: s.lock(p, held=False), "path"),
    "compact_dir": _simple(ScopedWorld.write, "path"),
    "archive_dir": _simple(ScopedWorld.archive, "path"),
    "drain_service": _simple(lambda s, n: s.service(n, "drain"), "service"),
    "undrain_service": _simple(lambda s, n: s.service(n, "undrain"), "service"),
    "restart_service": _simple(lambda s, n: s.service(n, "restart"), "service"),
    "submit_job": _simple(lambda s, n: s.job(n, "submit"), "job"),
    "check_status": _simple(lambda s, n: s.job(n, "check"), "job"),
    "retry_job": _simple(lambda s, n: s.job(n, "retry"), "job"),
}


@dataclass
class WorldTool:
    """One executable tool: scoped action, then observation of the declared fluents."""

    name: str
    declared_sha256: str
    world: FileMaintenanceWorld
    registry: EffectRegistry
    action: _Action

    def __call__(self, args: Mapping[str, str]) -> Observation:
        resource = next(iter(args.values()), "")
        fault = self.world.fault(self.name, resource)
        scoped = ScopedWorld(self.world, TOOL_OPERATIONS[self.name])
        outcome = "ok" if fault == "noop" else self.action(scoped, args)
        if fault == "concurrent_write" and resource in self.world.dirs:
            self.world.dirs[resource] += 1  # another process writes right after the call
        elif fault == "lock_lost":
            self.world.locks.discard(resource)  # the lock lease expires during the call
        self.world.log.append(f"{self.name}({resource}){' [fault]' if fault else ''}")
        watched: dict[GroundProp, None] = {}
        for declared in self.registry.ground_outcomes(self.name, args):
            for prop, _ in declared.updates:
                watched[prop] = None
        effects = tuple((prop, self.world.sense(prop)) for prop in watched)
        output: JsonValue = {"tool": self.name, "resource": resource, "status": outcome}
        return Observation(outcome=outcome, effects=effects, output=output)


def build_tools(
    world: FileMaintenanceWorld,
    registry: EffectRegistry,
    *,
    pins: Mapping[str, str] | None = None,
) -> list[WorldTool]:
    """Executable tools for ``world``, pinned to the catalog declarations unless ``pins`` given."""
    implemented = file_maintenance_registry()
    pinned = (
        dict(pins)
        if pins is not None
        else {name: implemented.tool_sha256(name) for name in implemented.tool_names}
    )
    return [
        WorldTool(
            name=name,
            declared_sha256=pinned[name],
            world=world,
            registry=registry,
            action=_ACTIONS[name],
        )
        for name in sorted(pinned)
        if name in _ACTIONS and registry.has_tool(name)
    ]
