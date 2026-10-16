"""The scheduler: which tasks may start *right now*, as one pure decision.

Design §5.2. A task is **Ready** when every dependency is Integrated; at most
`concurrency` tasks may be in flight; and two tasks whose ``files_allowed``
overlap never run at the same time — the later one stays Ready but is *deferred*
(deterministic pick: plan order). No LLM, no I/O, no clock: `compute_dispatch`
is a total function of the plan and the task state, so the policy is testable
without a graph, a repo, or a token.

The returned `state` is what the graph routes on:

* ``"escalated"`` — some task escalated; nothing is dispatched, the run stops.
* ``"done"`` — every task is Integrated; the run is over.
* ``"deadlock"`` — nothing to dispatch, nothing in flight, and unfinished tasks
  remain (e.g. a dependency cycle the validator missed); only a human editing
  the plan can unstick it.
* ``"ok"`` — dispatch this (possibly empty) list. An empty list *with* work in
  flight is normal: the graph waits for an in-flight task's ``integrate`` to
  finish and re-enter the scheduler.

"In flight" therefore means every status that has work outstanding in the graph
but is not yet Integrated: `IMPLEMENTING`, `REVIEW`, `FIXING` and `APPROVED`. An
Approved task is reviewed but not merged — its ``integrate`` branch may still be
pending, so it occupies a concurrency slot (and its files stay claimed) and must
never be mistaken for a deadlock; see `_IN_FLIGHT`.
"""

from pathlib import PurePosixPath
from typing import Literal, NamedTuple

from orch.state import TStatus

_IN_FLIGHT = frozenset(
    {TStatus.IMPLEMENTING, TStatus.REVIEW, TStatus.FIXING, TStatus.APPROVED}
)


class DispatchDecision(NamedTuple):
    """Task ids to start now, plus the graph-level state that decision implies."""

    dispatch: list[str]
    state: Literal["ok", "done", "deadlock", "escalated"]


def _status(value: object) -> TStatus:
    """Normalize a task status: a `TStatus` passes through, a serialized string is parsed.

    A checkpoint round trip hands the raw ``"pending"`` string back, so both
    forms must be accepted. Anything else raises rather than being silently
    treated as PENDING — dispatching a task that is already mid-flight would
    double it up.
    """
    if isinstance(value, TStatus):
        return value
    try:
        return TStatus(value)
    except ValueError as e:
        raise ValueError(f"unknown task status {value!r}") from e


def _patterns_overlap(a: str, b: str) -> bool:
    """Do the two ``files_allowed`` globs denote any path in common?

    These are patterns, not paths: a plan declares ownership as globs
    (``src/**``, ``db/migrations/**``). Two globs conflict when either matches
    the other taken as a path, and each pattern is also tried in its directory
    form (``src`` claims ``src/**``) so a directory entry and an explicit file
    beneath it are not scheduled together. Erring toward conflict only costs
    parallelism; missing one costs a merge conflict on the integration branch.
    """
    for x in (a, f"{a}/**"):
        for y in (b, f"{b}/**"):
            if x == y or PurePosixPath(y).full_match(x) or PurePosixPath(x).full_match(y):
                return True
    return False


def _claims_any(files: list[str], claims: list[list[str]]) -> bool:
    """Does this task's declared ownership intersect any set already claimed?"""
    return any(_patterns_overlap(f, c) for claim in claims for f in files for c in claim)


def compute_dispatch(
    *, tasks: dict[str, dict], task_state: dict[str, dict], concurrency: int
) -> DispatchDecision:
    """Decide which of ``tasks`` to dispatch against the current ``task_state``.

    ``tasks`` holds ``TaskDef.model_dump()``-shaped records in plan order; that
    order decides who wins both the concurrency cap and a file-ownership
    conflict. ``task_state`` may hold ``TStatus`` members or their serialized
    strings. See the module docstring for the meaning of the returned state.
    """
    statuses = {
        tid: _status((task_state.get(tid) or {}).get("status", TStatus.PENDING)) for tid in tasks
    }

    if any(status is TStatus.ESCALATED for status in statuses.values()):
        return DispatchDecision([], "escalated")
    if all(status is TStatus.INTEGRATED for status in statuses.values()):
        return DispatchDecision([], "done")

    in_flight = [tid for tid in tasks if statuses[tid] in _IN_FLIGHT]
    claimed = [list(tasks[tid].get("files_allowed", [])) for tid in in_flight]

    def ready(tid: str) -> bool:
        return statuses[tid] is TStatus.PENDING and all(
            statuses.get(dep) is TStatus.INTEGRATED for dep in tasks[tid].get("depends_on", [])
        )

    dispatch: list[str] = []
    slots = concurrency - len(in_flight)
    for tid in tasks:
        if len(dispatch) >= slots:
            break
        if not ready(tid):
            continue
        files = list(tasks[tid].get("files_allowed", []))
        if _claims_any(files, claimed):
            continue
        dispatch.append(tid)
        claimed.append(files)

    if dispatch or in_flight:
        return DispatchDecision(dispatch, "ok")
    return DispatchDecision([], "deadlock")
