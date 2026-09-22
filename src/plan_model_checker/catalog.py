"""The file-maintenance domain: effect declarations and the eight resource-protocol specs.

Sorts: ``path`` (directories), ``service`` and ``job``. Fluents track what the specs care about
(a fresh backup, a held lock, a drained service, a checked job status, a granted approval); events
name what each call does. ``delete_dir`` is destructive: it needs ``approved[path]``, which only
``request_approval`` can set, and consumes it.

The declarations are deliberately *optimistic* in one place: ``snapshot_dir`` declares a single
successful outcome. The static checker trusts that; the runtime monitor observes what actually
happened and blocks a later deletion when a snapshot silently failed (the demo's divergence).
"""

from __future__ import annotations

from plan_model_checker.effects import (
    ApprovalDecl,
    EffectDecl,
    EffectRegistry,
    EffectRegistryModel,
    EventDecl,
    OutcomeDecl,
    ParamDecl,
    PropDecl,
    ToolEffects,
)
from plan_model_checker.spec.specs import SpecSet

__all__ = [
    "PROTOCOL_SPECS",
    "file_maintenance_registry",
    "file_maintenance_registry_model",
    "protocol_specs",
]

PROTOCOL_SPECS = """\
# Resource-protocol rules for the file-maintenance agent. Variables are quantified over the
# resources a plan mentions.
spec backup_before_delete "a directory is deleted only right after a still-valid backup":
    G(delete[r] -> Y backup_ok[r]);
spec release_all_locks "every lock is released on every path before the plan ends":
    G(end -> !lock_held[r]);
spec mutate_under_lock "a directory is compacted only while its lock is held":
    G(mutate[r] -> Y lock_held[r]);
spec no_double_acquire "a held lock is never acquired again":
    G(acquire[r] -> Y !lock_held[r]);
spec status_before_retry "a non-idempotent job is retried only after checking its status":
    G(retry[j] -> Y status_checked[j]);
spec drain_before_restart "restart only while drained, with no undrain since the last drain":
    G(restart[s] -> Y(drained[s] & (!undrain[s] S drain[s])));
spec undrain_at_end "no service is left out of rotation when the plan ends":
    G(end -> !drained[s]);
spec archive_before_delete "a directory is recorded in the archive index before deletion":
    G(delete[r] -> O archive[r]);
"""


def _path(name: str = "path") -> tuple[ParamDecl, ...]:
    return (ParamDecl(name=name, sort="path"),)


def _set(prop: str, value: bool, *args: str) -> EffectDecl:
    return EffectDecl(prop=prop, args=args or ("path",), value=value)


def file_maintenance_registry_model() -> EffectRegistryModel:
    fluents = [
        PropDecl(name="backup_ok", kind="fluent", params=("path",), description="valid backup"),
        PropDecl(name="deleted", kind="fluent", params=("path",)),
        PropDecl(name="approved", kind="fluent", params=("path",), description="human approval"),
        PropDecl(name="lock_held", kind="fluent", params=("path",)),
        PropDecl(name="archived", kind="fluent", params=("path",)),
        PropDecl(name="drained", kind="fluent", params=("service",)),
        PropDecl(name="submitted", kind="fluent", params=("job",)),
        PropDecl(name="status_checked", kind="fluent", params=("job",)),
    ]
    events = [
        PropDecl(name=name, kind="event", params=(sort,))
        for name, sort in (
            ("backup", "path"),
            ("delete", "path"),
            ("write", "path"),
            ("acquire", "path"),
            ("release", "path"),
            ("mutate", "path"),
            ("archive", "path"),
            ("approval_requested", "path"),
            ("drain", "service"),
            ("undrain", "service"),
            ("restart", "service"),
            ("submit", "job"),
            ("check", "job"),
            ("retry", "job"),
        )
    ]
    single = "ok"
    tools = [
        ToolEffects(
            name="snapshot_dir",
            description="Copy a directory to the backup area.",
            params=_path(),
            event=EventDecl(prop="backup", args=("path",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("backup_ok", True),)),),
        ),
        ToolEffects(
            name="request_approval",
            description="Ask a human to approve deleting a directory; may stay pending.",
            params=_path(),
            event=EventDecl(prop="approval_requested", args=("path",)),
            outcomes=(
                OutcomeDecl(name="granted", effects=(_set("approved", True),)),
                OutcomeDecl(name="pending", effects=()),
            ),
        ),
        ToolEffects(
            name="delete_dir",
            description="Delete a directory (destructive).",
            params=_path(),
            event=EventDecl(prop="delete", args=("path",)),
            outcomes=(
                OutcomeDecl(
                    name=single,
                    effects=(_set("deleted", True), _set("approved", False)),
                ),
            ),
            destructive=True,
            approval=ApprovalDecl(prop="approved", args=("path",)),
        ),
        ToolEffects(
            name="write_file",
            description="Write into a directory; any earlier backup becomes stale.",
            params=_path(),
            event=EventDecl(prop="write", args=("path",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("backup_ok", False),)),),
        ),
        ToolEffects(
            name="acquire_lock",
            description="Take the maintenance lock of a directory.",
            params=_path(),
            event=EventDecl(prop="acquire", args=("path",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("lock_held", True),)),),
        ),
        ToolEffects(
            name="release_lock",
            description="Release the maintenance lock of a directory.",
            params=_path(),
            event=EventDecl(prop="release", args=("path",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("lock_held", False),)),),
        ),
        ToolEffects(
            name="compact_dir",
            description="Rewrite a directory in place (needs the lock).",
            params=_path(),
            event=EventDecl(prop="mutate", args=("path",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("backup_ok", False),)),),
        ),
        ToolEffects(
            name="archive_dir",
            description="Record a directory in the archive index.",
            params=_path(),
            event=EventDecl(prop="archive", args=("path",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("archived", True),)),),
        ),
        ToolEffects(
            name="drain_service",
            description="Take a service out of rotation.",
            params=(ParamDecl(name="service", sort="service"),),
            event=EventDecl(prop="drain", args=("service",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("drained", True, "service"),)),),
        ),
        ToolEffects(
            name="undrain_service",
            description="Put a service back into rotation.",
            params=(ParamDecl(name="service", sort="service"),),
            event=EventDecl(prop="undrain", args=("service",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("drained", False, "service"),)),),
        ),
        ToolEffects(
            name="restart_service",
            description="Restart a service process.",
            params=(ParamDecl(name="service", sort="service"),),
            event=EventDecl(prop="restart", args=("service",)),
            outcomes=(OutcomeDecl(name=single, effects=()),),
        ),
        ToolEffects(
            name="submit_job",
            description="Submit a non-idempotent batch job.",
            params=(ParamDecl(name="job", sort="job"),),
            event=EventDecl(prop="submit", args=("job",)),
            outcomes=(
                OutcomeDecl(
                    name=single,
                    effects=(
                        _set("submitted", True, "job"),
                        _set("status_checked", False, "job"),
                    ),
                ),
            ),
        ),
        ToolEffects(
            name="check_status",
            description="Look up whether a submitted job is still running.",
            params=(ParamDecl(name="job", sort="job"),),
            event=EventDecl(prop="check", args=("job",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("status_checked", True, "job"),)),),
        ),
        ToolEffects(
            name="retry_job",
            description="Submit a job again after a failure.",
            params=(ParamDecl(name="job", sort="job"),),
            event=EventDecl(prop="retry", args=("job",)),
            outcomes=(OutcomeDecl(name=single, effects=(_set("status_checked", False, "job"),)),),
        ),
    ]
    return EffectRegistryModel(
        name="file-maintenance",
        sorts=("path", "service", "job"),
        props=(*fluents, *events),
        tools=tuple(tools),
    )


def file_maintenance_registry() -> EffectRegistry:
    return EffectRegistry(file_maintenance_registry_model())


def protocol_specs(registry: EffectRegistry, *, include_implicit: bool = True) -> SpecSet:
    return SpecSet.parse(PROTOCOL_SPECS, registry, include_implicit=include_implicit)
