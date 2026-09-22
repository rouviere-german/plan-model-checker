"""Labeled plan corpus and runtime divergence scenarios for the file-maintenance domain.

**Plans (60).** Thirty correct plans, four or three per protocol spec, cover straight-line code,
opaque branches, guarded branches, fixed loops and data-dependent collections. Each has a
violating twin produced by *omitting one step* (the snapshot before a delete, the release on one
branch, the status check before a retry...). The label of a plan is fixed by that construction,
not by the checker: correct plans are expected ``safe``, twins are expected to violate the spec
they were built for (possibly among others). Tests confirm every label against the brute-force
oracle too.

**Divergence scenarios (30).** A plan that passes the static check runs against a world in which
one tool does not do what its declaration says (a snapshot that silently does nothing or is
immediately overwritten, a lock that is not taken or whose lease expires, a status check or a
drain that has no effect, a release that keeps the lock). Each scenario names the step that must
be blocked and the spec that must block it.

**Hard repair cases (24).** Violating plans whose fix is more than inserting the one step they
omit: several omissions, wrong order, wrong arguments, omissions inside loops and branches, and
plans that need a step moved under an approval guard, which no insertion achieves. They measure
where the scripted, insert-one-step repairer stops working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from plan_model_checker.ir import Plan

__all__ = [
    "CorpusCase",
    "DivergenceScenario",
    "HardCase",
    "build_corpus",
    "build_divergence_scenarios",
    "build_hard_corpus",
    "call",
    "each",
    "each_of",
    "if_opaque",
    "if_prop",
    "plan",
]

Step = dict[str, Any]


def call(tool: str, **args: str) -> Step:
    return {
        "kind": "call",
        "tool": tool,
        "args": [{"name": name, "value": value} for name, value in args.items()],
    }


def if_prop(prop: str, arg: str, then: list[Step], otherwise: list[Step] | None = None) -> Step:
    return {
        "kind": "if",
        "cond": {"kind": "prop", "prop": prop, "args": [arg]},
        "then": then,
        "otherwise": otherwise or [],
    }


def if_opaque(label: str, then: list[Step], otherwise: list[Step] | None = None) -> Step:
    return {
        "kind": "if",
        "cond": {"kind": "opaque", "label": label},
        "then": then,
        "otherwise": otherwise or [],
    }


def each(var: str, items: list[str], body: list[Step]) -> Step:
    return {
        "kind": "foreach",
        "var": var,
        "items": items,
        "max_iterations": len(items),
        "body": body,
    }


def each_of(var: str, collection: str, bound: int, body: list[Step]) -> Step:
    return {
        "kind": "foreach",
        "var": var,
        "collection": collection,
        "max_iterations": bound,
        "body": body,
    }


def plan(goal: str, *steps: Step) -> Plan:
    return Plan.model_validate({"goal": goal, "steps": list(steps)})


def _keep(omit: bool, *steps: Step) -> list[Step]:
    """The steps, or none of them when this is the step the violating twin omits."""
    return [] if omit else list(steps)


@dataclass(frozen=True, slots=True)
class CorpusCase:
    name: str
    spec: str
    plan: Plan
    expected: str
    omitted: str | None = None

    @property
    def violating(self) -> bool:
        return self.expected == "violation"


Variant = Callable[[bool], Plan]


def _guarded_delete(path: str, *, omit_backup: bool = False, omit_archive: bool = False) -> Step:
    return if_prop(
        "approved",
        path,
        [
            *_keep(omit_archive, call("archive_dir", path=path)),
            *_keep(omit_backup, call("snapshot_dir", path=path)),
            call("delete_dir", path=path),
        ],
    )


def _backup_variants() -> list[tuple[str, Variant, str]]:
    def single(omit: bool) -> Plan:
        path = "/srv/logs/2025"
        return plan(
            "Delete last year's logs.",
            call("request_approval", path=path),
            _guarded_delete(path, omit_backup=omit),
        )

    def fixed_loop(omit: bool) -> Plan:
        return plan(
            "Delete two scratch directories.",
            each(
                "d",
                ["/scratch/build-a", "/scratch/build-b"],
                [call("request_approval", path="$d"), _guarded_delete("$d", omit_backup=omit)],
            ),
        )

    def collection(omit: bool) -> Plan:
        return plan(
            "Delete every stale directory (at most three).",
            each_of(
                "d",
                "stale_dirs",
                3,
                [call("request_approval", path="$d"), _guarded_delete("$d", omit_backup=omit)],
            ),
        )

    def stale_backup(omit: bool) -> Plan:
        path = "/srv/reports"
        return plan(
            "Write a final report, then delete the report directory.",
            call("request_approval", path=path),
            if_prop(
                "approved",
                path,
                [
                    call("archive_dir", path=path),
                    call("snapshot_dir", path=path),
                    call("write_file", path=path),
                    *_keep(omit, call("snapshot_dir", path=path)),
                    call("delete_dir", path=path),
                ],
            ),
        )

    return [
        ("delete_single", single, "snapshot_dir"),
        ("delete_fixed_loop", fixed_loop, "snapshot_dir"),
        ("delete_collection", collection, "snapshot_dir"),
        ("delete_after_write", stale_backup, "second snapshot_dir"),
    ]


def _release_variants() -> list[tuple[str, Variant, str]]:
    def single(omit: bool) -> Plan:
        path = "/var/lib/cache"
        return plan(
            "Compact the cache.",
            call("acquire_lock", path=path),
            call("compact_dir", path=path),
            *_keep(omit, call("release_lock", path=path)),
        )

    def branch(omit: bool) -> Plan:
        path = "/var/lib/queue"
        return plan(
            "Compact the queue if it is large.",
            call("acquire_lock", path=path),
            if_opaque(
                "queue_is_large",
                [call("compact_dir", path=path), *_keep(omit, call("release_lock", path=path))],
                [call("release_lock", path=path)],
            ),
        )

    def fixed_loop(omit: bool) -> Plan:
        return plan(
            "Compact two caches.",
            each(
                "c",
                ["/var/cache/x", "/var/cache/y"],
                [
                    call("acquire_lock", path="$c"),
                    call("compact_dir", path="$c"),
                    *_keep(omit, call("release_lock", path="$c")),
                ],
            ),
        )

    def collection(omit: bool) -> Plan:
        return plan(
            "Compact dirty shards.",
            each_of(
                "s",
                "shards",
                2,
                [
                    call("acquire_lock", path="$s"),
                    if_opaque("shard_is_dirty", [call("compact_dir", path="$s")]),
                    *_keep(omit, call("release_lock", path="$s")),
                ],
            ),
        )

    return [
        ("release_single", single, "release_lock"),
        ("release_branch", branch, "release_lock on one branch"),
        ("release_fixed_loop", fixed_loop, "release_lock"),
        ("release_collection", collection, "release_lock"),
    ]


def _mutate_variants() -> list[tuple[str, Variant, str]]:
    def twice(omit: bool) -> Plan:
        path = "/data/index"
        return plan(
            "Compact the index twice.",
            *_keep(omit, call("acquire_lock", path=path)),
            call("compact_dir", path=path),
            call("compact_dir", path=path),
            call("release_lock", path=path),
        )

    def branch(omit: bool) -> Plan:
        path = "/data/segments"
        return plan(
            "Compact segments if fragmented.",
            *_keep(omit, call("acquire_lock", path=path)),
            if_opaque("fragmented", [call("compact_dir", path=path)]),
            call("release_lock", path=path),
        )

    def fixed_loop(omit: bool) -> Plan:
        return plan(
            "Compact the two mail spools.",
            each(
                "m",
                ["/var/mail/a", "/var/mail/b"],
                [
                    *_keep(omit, call("acquire_lock", path="$m")),
                    call("compact_dir", path="$m"),
                    call("release_lock", path="$m"),
                ],
            ),
        )

    def collection(omit: bool) -> Plan:
        return plan(
            "Compact every index listed by the catalog.",
            each_of(
                "i",
                "indexes",
                2,
                [
                    *_keep(omit, call("acquire_lock", path="$i")),
                    call("compact_dir", path="$i"),
                    call("release_lock", path="$i"),
                ],
            ),
        )

    return [
        ("mutate_twice", twice, "acquire_lock"),
        ("mutate_branch", branch, "acquire_lock"),
        ("mutate_fixed_loop", fixed_loop, "acquire_lock"),
        ("mutate_collection", collection, "acquire_lock"),
    ]


def _double_acquire_variants() -> list[tuple[str, Variant, str]]:
    def windows(omit: bool) -> Plan:
        path = "/data/warehouse"
        return plan(
            "Two maintenance windows on the warehouse.",
            call("acquire_lock", path=path),
            call("compact_dir", path=path),
            *_keep(omit, call("release_lock", path=path)),
            call("acquire_lock", path=path),
            call("compact_dir", path=path),
            call("release_lock", path=path),
        )

    def retry_branch(omit: bool) -> Plan:
        path = "/data/journal"
        return plan(
            "Compact the journal, re-taking the lock if the first attempt stalls.",
            call("acquire_lock", path=path),
            if_opaque(
                "stalled",
                [*_keep(omit, call("release_lock", path=path)), call("acquire_lock", path=path)],
            ),
            call("compact_dir", path=path),
            call("release_lock", path=path),
        )

    def loop_then_again(omit: bool) -> Plan:
        return plan(
            "Compact two tables, then the first one again.",
            each(
                "t",
                ["/db/t1", "/db/t2"],
                [
                    call("acquire_lock", path="$t"),
                    call("compact_dir", path="$t"),
                    *_keep(omit, call("release_lock", path="$t")),
                ],
            ),
            call("acquire_lock", path="/db/t1"),
            call("release_lock", path="/db/t1"),
        )

    def collection(omit: bool) -> Plan:
        return plan(
            "Two passes over every table.",
            each_of(
                "t",
                "tables",
                2,
                [
                    call("acquire_lock", path="$t"),
                    call("compact_dir", path="$t"),
                    *_keep(omit, call("release_lock", path="$t")),
                    call("acquire_lock", path="$t"),
                    call("release_lock", path="$t"),
                ],
            ),
        )

    return [
        ("reacquire_windows", windows, "first release_lock"),
        ("reacquire_branch", retry_branch, "release_lock before re-acquiring"),
        ("reacquire_after_loop", loop_then_again, "release_lock in the loop"),
        ("reacquire_collection", collection, "first release_lock"),
    ]


def _retry_variants() -> list[tuple[str, Variant, str]]:
    def on_failure(omit: bool) -> Plan:
        job = "job-nightly-export"
        return plan(
            "Submit the export and retry it if it failed.",
            call("submit_job", job=job),
            if_opaque(
                "export_failed",
                [*_keep(omit, call("check_status", job=job)), call("retry_job", job=job)],
            ),
        )

    def two_retries(omit: bool) -> Plan:
        job = "job-reindex"
        return plan(
            "Submit the reindex and retry it twice.",
            call("submit_job", job=job),
            call("check_status", job=job),
            call("retry_job", job=job),
            *_keep(omit, call("check_status", job=job)),
            call("retry_job", job=job),
        )

    def fixed_loop(omit: bool) -> Plan:
        return plan(
            "Submit both billing jobs, retrying failures.",
            each(
                "j",
                ["job-billing-eu", "job-billing-us"],
                [
                    call("submit_job", job="$j"),
                    if_opaque(
                        "job_failed",
                        [*_keep(omit, call("check_status", job="$j")), call("retry_job", job="$j")],
                    ),
                ],
            ),
        )

    def stuck(omit: bool) -> Plan:
        job = "job-thumbnails"
        return plan(
            "Submit the thumbnail job and retry it if it is stuck.",
            call("submit_job", job=job),
            *_keep(omit, call("check_status", job=job)),
            if_opaque("job_stuck", [call("retry_job", job=job)]),
        )

    return [
        ("retry_on_failure", on_failure, "check_status"),
        ("retry_twice", two_retries, "second check_status"),
        ("retry_fixed_loop", fixed_loop, "check_status"),
        ("retry_if_stuck", stuck, "check_status"),
    ]


def _restart_variants() -> list[tuple[str, Variant, str]]:
    def single(omit: bool) -> Plan:
        svc = "svc:api"
        return plan(
            "Restart the API.",
            *_keep(omit, call("drain_service", service=svc)),
            call("restart_service", service=svc),
            call("undrain_service", service=svc),
        )

    def redrain(omit: bool) -> Plan:
        svc = "svc:search"
        return plan(
            "Probe, then restart the search service.",
            call("drain_service", service=svc),
            call("undrain_service", service=svc),
            *_keep(omit, call("drain_service", service=svc)),
            call("restart_service", service=svc),
            call("undrain_service", service=svc),
        )

    def fixed_loop(omit: bool) -> Plan:
        return plan(
            "Rolling restart of the web tier.",
            each(
                "s",
                ["svc:web-1", "svc:web-2"],
                [
                    *_keep(omit, call("drain_service", service="$s")),
                    call("restart_service", service="$s"),
                    call("undrain_service", service="$s"),
                ],
            ),
        )

    def on_change(omit: bool) -> Plan:
        svc = "svc:gateway"
        return plan(
            "Restart the gateway if its config changed.",
            *_keep(omit, call("drain_service", service=svc)),
            if_opaque("config_changed", [call("restart_service", service=svc)]),
            call("undrain_service", service=svc),
        )

    return [
        ("restart_single", single, "drain_service"),
        ("restart_after_probe", redrain, "second drain_service"),
        ("restart_rolling", fixed_loop, "drain_service"),
        ("restart_on_change", on_change, "drain_service"),
    ]


def _undrain_variants() -> list[tuple[str, Variant, str]]:
    def single(omit: bool) -> Plan:
        svc = "svc:worker"
        return plan(
            "Restart the worker.",
            call("drain_service", service=svc),
            call("restart_service", service=svc),
            *_keep(omit, call("undrain_service", service=svc)),
        )

    def branch(omit: bool) -> Plan:
        svc = "svc:cache"
        return plan(
            "Drain the cache; restart it unless it is healthy.",
            call("drain_service", service=svc),
            if_opaque(
                "healthy",
                [call("undrain_service", service=svc)],
                [
                    call("restart_service", service=svc),
                    *_keep(omit, call("undrain_service", service=svc)),
                ],
            ),
        )

    def collection(omit: bool) -> Plan:
        return plan(
            "Restart every pool member.",
            each_of(
                "s",
                "pool",
                2,
                [
                    call("drain_service", service="$s"),
                    call("restart_service", service="$s"),
                    *_keep(omit, call("undrain_service", service="$s")),
                ],
            ),
        )

    return [
        ("undrain_single", single, "undrain_service"),
        ("undrain_branch", branch, "undrain_service on one branch"),
        ("undrain_collection", collection, "undrain_service"),
    ]


def _archive_variants() -> list[tuple[str, Variant, str]]:
    def single(omit: bool) -> Plan:
        path = "/srv/uploads/tmp"
        return plan(
            "Delete the temporary uploads.",
            call("request_approval", path=path),
            _guarded_delete(path, omit_archive=omit),
        )

    def early(omit: bool) -> Plan:
        path = "/srv/exports/old"
        return plan(
            "Archive, then delete old exports once approved.",
            *_keep(omit, call("archive_dir", path=path)),
            call("request_approval", path=path),
            if_prop(
                "approved",
                path,
                [call("snapshot_dir", path=path), call("delete_dir", path=path)],
            ),
        )

    def fixed_loop(omit: bool) -> Plan:
        return plan(
            "Delete two retired home directories.",
            each(
                "h",
                ["/home/retired-1", "/home/retired-2"],
                [call("request_approval", path="$h"), _guarded_delete("$h", omit_archive=omit)],
            ),
        )

    return [
        ("archive_single", single, "archive_dir"),
        ("archive_early", early, "archive_dir"),
        ("archive_fixed_loop", fixed_loop, "archive_dir"),
    ]


_FAMILIES: tuple[tuple[str, Callable[[], list[tuple[str, Variant, str]]]], ...] = (
    ("backup_before_delete", _backup_variants),
    ("release_all_locks", _release_variants),
    ("mutate_under_lock", _mutate_variants),
    ("no_double_acquire", _double_acquire_variants),
    ("status_before_retry", _retry_variants),
    ("drain_before_restart", _restart_variants),
    ("undrain_at_end", _undrain_variants),
    ("archive_before_delete", _archive_variants),
)


def build_corpus() -> list[CorpusCase]:
    """30 safe plans and their 30 single-omission twins, across the 8 protocol specs."""
    cases: list[CorpusCase] = []
    for spec, variants in _FAMILIES:
        for name, variant, omitted in variants():
            cases.append(CorpusCase(f"{name}/safe", spec, variant(False), "safe"))
            cases.append(
                CorpusCase(f"{name}/omits", spec, variant(True), "violation", omitted=omitted)
            )
    return cases


@dataclass(frozen=True, slots=True)
class DivergenceScenario:
    name: str
    plan: Plan
    faults: dict[str, str]
    blocked_step: str
    blocked_spec: str
    dirs: dict[str, int] = field(default_factory=dict)
    collections: dict[str, list[str]] = field(default_factory=dict)
    opaque: dict[str, bool] = field(default_factory=dict)

    @property
    def faulted_tools(self) -> frozenset[str]:
        """The tools the faults apply to (fault keys are ``tool`` or ``tool:resource``)."""
        return frozenset(key.partition(":")[0] for key in self.faults)


def _delete_plan(paths: list[str], *, write_first: bool = False) -> Plan:
    steps: list[Step] = []
    for path in paths:
        body = [call("archive_dir", path=path)]
        if write_first:
            body.append(call("write_file", path=path))
        body += [call("snapshot_dir", path=path), call("delete_dir", path=path)]
        steps += [call("request_approval", path=path), if_prop("approved", path, body)]
    return plan("Delete directories after backing them up.", *steps)


def _compact_plan(paths: list[str], passes: int) -> Plan:
    steps: list[Step] = []
    for path in paths:
        steps.append(call("acquire_lock", path=path))
        steps += [call("compact_dir", path=path) for _ in range(passes)]
        steps.append(call("release_lock", path=path))
    return plan("Compact directories under their locks.", *steps)


def _windows_plan(path: str) -> Plan:
    return plan(
        "Two maintenance windows.",
        call("acquire_lock", path=path),
        call("compact_dir", path=path),
        call("release_lock", path=path),
        call("acquire_lock", path=path),
        call("compact_dir", path=path),
        call("release_lock", path=path),
    )


def _retry_plan(jobs: list[str]) -> Plan:
    steps: list[Step] = []
    for job in jobs:
        steps += [
            call("submit_job", job=job),
            call("check_status", job=job),
            call("retry_job", job=job),
        ]
    return plan("Submit jobs and retry them once after checking.", *steps)


def _restart_plan(services: list[str]) -> Plan:
    steps: list[Step] = []
    for svc in services:
        steps += [
            call("drain_service", service=svc),
            call("restart_service", service=svc),
            call("undrain_service", service=svc),
        ]
    return plan("Rolling restart.", *steps)


def build_divergence_scenarios() -> list[DivergenceScenario]:
    """30 runs where a tool's real behaviour departs from its declaration."""
    scenarios: list[DivergenceScenario] = []
    delete_sets = [
        ["/srv/a"],
        ["/srv/b"],
        ["/srv/c", "/srv/d"],
        ["/srv/e", "/srv/f"],
        ["/srv/g", "/srv/h", "/srv/i"],
    ]
    for index, paths in enumerate(delete_sets):
        target = paths[-1]
        dirs = dict.fromkeys(paths, 1)
        scenarios.append(
            DivergenceScenario(
                name=f"snapshot_silently_fails_{index}",
                plan=_delete_plan(paths),
                faults={f"snapshot_dir:{target}": "noop"},
                blocked_step=f"delete_dir(path={target})",
                blocked_spec="backup_before_delete",
                dirs=dirs,
            )
        )
        scenarios.append(
            DivergenceScenario(
                name=f"snapshot_overwritten_{index}",
                plan=_delete_plan(paths, write_first=index % 2 == 1),
                faults={f"snapshot_dir:{target}": "concurrent_write"},
                blocked_step=f"delete_dir(path={target})",
                blocked_spec="backup_before_delete",
                dirs=dirs,
            )
        )
    lock_sets = [["/idx/a"], ["/idx/b"], ["/idx/c", "/idx/d"], ["/idx/e", "/idx/f"]]
    for index, paths in enumerate(lock_sets):
        target = paths[-1]
        scenarios.append(
            DivergenceScenario(
                name=f"lock_not_taken_{index}",
                plan=_compact_plan(paths, passes=1 + index % 2),
                faults={f"acquire_lock:{target}": "noop"},
                blocked_step=f"compact_dir(path={target})",
                blocked_spec="mutate_under_lock",
            )
        )
        scenarios.append(
            DivergenceScenario(
                name=f"lock_lease_expires_{index}",
                plan=_compact_plan(paths, passes=2),
                faults={f"compact_dir:{target}": "lock_lost"},
                blocked_step=f"compact_dir(path={target})",
                blocked_spec="mutate_under_lock",
            )
        )
    for index, path in enumerate(["/wh/a", "/wh/b", "/wh/c", "/wh/d"]):
        scenarios.append(
            DivergenceScenario(
                name=f"release_is_a_noop_{index}",
                plan=_windows_plan(path),
                faults={f"release_lock:{path}": "noop"},
                blocked_step=f"acquire_lock(path={path})",
                blocked_spec="no_double_acquire",
            )
        )
    job_sets = [["job-a"], ["job-b"], ["job-c", "job-d"], ["job-e", "job-f"]]
    for index, jobs in enumerate(job_sets):
        target = jobs[-1]
        scenarios.append(
            DivergenceScenario(
                name=f"status_check_has_no_effect_{index}",
                plan=_retry_plan(jobs),
                faults={f"check_status:{target}": "noop"},
                blocked_step=f"retry_job(job={target})",
                blocked_spec="status_before_retry",
            )
        )
    service_sets = [["svc:a"], ["svc:b"], ["svc:c", "svc:d"], ["svc:e", "svc:f"]]
    for index, services in enumerate(service_sets):
        target = services[-1]
        scenarios.append(
            DivergenceScenario(
                name=f"drain_has_no_effect_{index}",
                plan=_restart_plan(services),
                faults={f"drain_service:{target}": "noop"},
                blocked_step=f"restart_service(service={target})",
                blocked_spec="drain_before_restart",
            )
        )
    return scenarios


@dataclass(frozen=True, slots=True)
class HardCase:
    """A violating plan whose fix is harder than inserting the one step it omits."""

    name: str
    category: str
    plan: Plan
    violated: frozenset[str]


def _approve_then(path: str, body: list[Step]) -> list[Step]:
    return [call("request_approval", path=path), if_prop("approved", path, body)]


def _multi_omission() -> list[HardCase]:
    return [
        HardCase(
            "delete_without_archive_or_snapshot",
            "multi_omission",
            plan("Delete /m1.", *_approve_then("/m1", [call("delete_dir", path="/m1")])),
            frozenset({"backup_before_delete", "archive_before_delete"}),
        ),
        HardCase(
            "compact_without_lock_or_release",
            "multi_omission",
            plan("Compact /m2.", call("compact_dir", path="/m2")),
            frozenset({"mutate_under_lock"}),
        ),
        HardCase(
            "restart_without_drain_or_undrain",
            "multi_omission",
            plan("Restart svc:m3.", call("restart_service", service="svc:m3")),
            frozenset({"drain_before_restart"}),
        ),
        HardCase(
            "retry_and_restart_with_three_omissions",
            "multi_omission",
            plan(
                "Retry job-m4 and restart svc:m4.",
                call("submit_job", job="job-m4"),
                call("retry_job", job="job-m4"),
                call("restart_service", service="svc:m4"),
            ),
            frozenset({"status_before_retry", "drain_before_restart"}),
        ),
        HardCase(
            "delete_and_compact_with_four_omissions",
            "multi_omission",
            plan(
                "Delete /m5 and compact /m6.",
                *_approve_then("/m5", [call("delete_dir", path="/m5")]),
                call("compact_dir", path="/m6"),
            ),
            frozenset({"backup_before_delete", "archive_before_delete", "mutate_under_lock"}),
        ),
    ]


def _wrong_order() -> list[HardCase]:
    return [
        HardCase(
            "snapshot_after_delete",
            "wrong_order",
            plan(
                "Delete /o1.",
                *_approve_then(
                    "/o1",
                    [
                        call("archive_dir", path="/o1"),
                        call("delete_dir", path="/o1"),
                        call("snapshot_dir", path="/o1"),
                    ],
                ),
            ),
            frozenset({"backup_before_delete"}),
        ),
        HardCase(
            "release_before_compact",
            "wrong_order",
            plan(
                "Compact /o2.",
                call("acquire_lock", path="/o2"),
                call("release_lock", path="/o2"),
                call("compact_dir", path="/o2"),
            ),
            frozenset({"mutate_under_lock"}),
        ),
        HardCase(
            "archive_after_delete",
            "wrong_order",
            plan(
                "Delete /o3.",
                *_approve_then(
                    "/o3",
                    [
                        call("snapshot_dir", path="/o3"),
                        call("delete_dir", path="/o3"),
                        call("archive_dir", path="/o3"),
                    ],
                ),
            ),
            frozenset({"archive_before_delete"}),
        ),
        HardCase(
            "check_after_retry",
            "wrong_order",
            plan(
                "Retry job-o4.",
                call("submit_job", job="job-o4"),
                call("retry_job", job="job-o4"),
                call("check_status", job="job-o4"),
            ),
            frozenset({"status_before_retry"}),
        ),
    ]


def _wrong_args() -> list[HardCase]:
    return [
        HardCase(
            "snapshot_of_another_directory",
            "wrong_args",
            plan(
                "Delete /w1.",
                *_approve_then(
                    "/w1",
                    [
                        call("archive_dir", path="/w1"),
                        call("snapshot_dir", path="/w2"),
                        call("delete_dir", path="/w1"),
                    ],
                ),
            ),
            frozenset({"backup_before_delete"}),
        ),
        HardCase(
            "lock_on_another_path",
            "wrong_args",
            plan(
                "Compact /w4.",
                call("acquire_lock", path="/w3"),
                call("compact_dir", path="/w4"),
                call("release_lock", path="/w3"),
            ),
            frozenset({"mutate_under_lock"}),
        ),
        HardCase(
            "drain_of_another_service",
            "wrong_args",
            plan(
                "Restart svc:w6.",
                call("drain_service", service="svc:w5"),
                call("restart_service", service="svc:w6"),
                call("undrain_service", service="svc:w5"),
            ),
            frozenset({"drain_before_restart"}),
        ),
        HardCase(
            "status_of_another_job",
            "wrong_args",
            plan(
                "Retry job-w7.",
                call("submit_job", job="job-w7"),
                call("check_status", job="job-w8"),
                call("retry_job", job="job-w7"),
            ),
            frozenset({"status_before_retry"}),
        ),
    ]


def _in_loops_and_branches() -> list[HardCase]:
    return [
        HardCase(
            "snapshot_missing_in_collection_loop",
            "in_loop",
            plan(
                "Delete every stale directory.",
                each_of(
                    "d",
                    "stale",
                    3,
                    _approve_then(
                        "$d", [call("archive_dir", path="$d"), call("delete_dir", path="$d")]
                    ),
                ),
            ),
            frozenset({"backup_before_delete"}),
        ),
        HardCase(
            "release_missing_in_fixed_loop",
            "in_loop",
            plan(
                "Compact two shards.",
                each(
                    "d",
                    ["/l1", "/l2"],
                    [call("acquire_lock", path="$d"), call("compact_dir", path="$d")],
                ),
            ),
            frozenset({"release_all_locks"}),
        ),
        HardCase(
            "undrain_missing_in_collection_loop",
            "in_loop",
            plan(
                "Restart every service.",
                each_of(
                    "s",
                    "services",
                    2,
                    [
                        call("drain_service", service="$s"),
                        call("restart_service", service="$s"),
                    ],
                ),
            ),
            frozenset({"undrain_at_end"}),
        ),
        HardCase(
            "shared_lock_taken_every_iteration",
            "in_loop",
            plan(
                "Archive two directories under one lock.",
                each(
                    "i",
                    ["/x1", "/x2"],
                    [
                        call("acquire_lock", path="/lk"),
                        call("compact_dir", path="/lk"),
                        call("archive_dir", path="$i"),
                    ],
                ),
            ),
            frozenset({"no_double_acquire", "release_all_locks"}),
        ),
        HardCase(
            "release_missing_on_one_branch",
            "in_branch",
            plan(
                "Compact /b1 if it is big.",
                if_opaque(
                    "big",
                    [
                        call("acquire_lock", path="/b1"),
                        call("compact_dir", path="/b1"),
                        call("release_lock", path="/b1"),
                    ],
                    [call("acquire_lock", path="/b1"), call("compact_dir", path="/b1")],
                ),
            ),
            frozenset({"release_all_locks"}),
        ),
        HardCase(
            "check_missing_on_one_branch",
            "in_branch",
            plan(
                "Retry job-b2 now or later.",
                call("submit_job", job="job-b2"),
                if_opaque(
                    "now",
                    [call("retry_job", job="job-b2")],
                    [call("check_status", job="job-b2"), call("retry_job", job="job-b2")],
                ),
            ),
            frozenset({"status_before_retry"}),
        ),
        HardCase(
            "snapshot_missing_in_nested_branch",
            "in_branch",
            plan(
                "Delete /b3.",
                *_approve_then(
                    "/b3",
                    [
                        call("archive_dir", path="/b3"),
                        if_opaque(
                            "fast",
                            [call("delete_dir", path="/b3")],
                            [call("snapshot_dir", path="/b3"), call("delete_dir", path="/b3")],
                        ),
                    ],
                ),
            ),
            frozenset({"backup_before_delete"}),
        ),
        HardCase(
            "undrain_missing_in_branch",
            "in_branch",
            plan(
                "Restart svc:b4 during maintenance.",
                if_opaque(
                    "maintenance",
                    [
                        call("drain_service", service="svc:b4"),
                        call("restart_service", service="svc:b4"),
                    ],
                ),
            ),
            frozenset({"undrain_at_end"}),
        ),
    ]


def _needs_restructure() -> list[HardCase]:
    return [
        HardCase(
            "delete_outside_the_approval_guard",
            "needs_restructure",
            plan(
                "Delete /r1.",
                call("request_approval", path="/r1"),
                call("archive_dir", path="/r1"),
                call("snapshot_dir", path="/r1"),
                call("delete_dir", path="/r1"),
            ),
            frozenset({"approval_delete_dir"}),
        ),
        HardCase(
            "loop_deletes_without_approval_guard",
            "needs_restructure",
            plan(
                "Delete every stale directory.",
                each_of(
                    "d",
                    "stale",
                    2,
                    [
                        call("request_approval", path="$d"),
                        call("archive_dir", path="$d"),
                        call("snapshot_dir", path="$d"),
                        call("delete_dir", path="$d"),
                    ],
                ),
            ),
            frozenset({"approval_delete_dir"}),
        ),
        HardCase(
            "second_delete_reuses_a_consumed_approval",
            "needs_restructure",
            plan(
                "Delete /r2 twice.",
                *_approve_then(
                    "/r2",
                    [
                        call("archive_dir", path="/r2"),
                        call("snapshot_dir", path="/r2"),
                        call("delete_dir", path="/r2"),
                        call("delete_dir", path="/r2"),
                    ],
                ),
            ),
            frozenset({"approval_delete_dir"}),
        ),
    ]


def build_hard_corpus() -> list[HardCase]:
    """24 violating plans for the repair loop, harder than one omitted step.

    Several omissions in one plan, steps in the wrong order, steps on the wrong resource, steps
    missing inside loops and branches, and plans that no insertion can fix because the unsafe
    step has to move under an approval guard. ``violated`` is the set of specs the plan is built
    to break; the checker must report at least those. What the scripted repairer makes of each
    case is measured by bench/bench_checker.py, not assumed here.
    """
    return [
        *_multi_omission(),
        *_wrong_order(),
        *_wrong_args(),
        *_in_loops_and_branches(),
        *_needs_restructure(),
    ]
