"""Merge an Approved task into the integration branch and re-run its checks.

The integration worktree is shared, so only one merge — and its post-merge
checks — may touch it at a time: parallel task branches queue on a module lock.
Merging is idempotent: a branch already reachable from the integration ``HEAD``
is recognized by ``is_ancestor`` and never merged twice (Review Focus #5).

When a merge conflicts or the post-merge checks come back red, the failure is
treated as clearly local to the one task: :func:`spawn_repair` adds a Repair task
on ``agent/<TASK-ID>-repair`` (the ordinary pipeline, run in its own worktree)
and the scheduler picks it up on the next loop. The repair is **bounded**: one
per failure event (the failing branch's head sha at failure time), and a repair
may never spawn a repair — a second failure escalates ``repair_exhausted``
instead. When a repair does integrate, it resolves its parent to ``INTEGRATED``
too, since the repair is what actually landed the parent's work.
"""

import threading
from pathlib import Path

from orch.adapters.base import AgentAdapter
from orch.artifacts import RunDir
from orch.checks import checks_green, run_checks
from orch.config import Config
from orch.errors import GitError
from orch.gitops import head_sha, is_ancestor, merge_no_ff
from orch.nodes import escalation_record, passthrough_env
from orch.nodes.task import task_branch
from orch.plan import TaskDef
from orch.state import TStatus, new_task_state

#: One git working tree, one writer: integrations run one at a time.
_INTEGRATION_LOCK = threading.Lock()

INTEGRATION_WORKTREE = "orch-integration"


def integration_worktree(cfg: Config) -> Path:
    """``<target>/<worktree_root>/orch-integration`` — the shared integration tree."""
    return cfg.target_repo / cfg.worktree_root / INTEGRATION_WORKTREE


def is_approved(value) -> bool:
    """True for an ``APPROVED`` status, whether a ``TStatus`` or its string form."""
    return value == TStatus.APPROVED


def failure_event_id(state: dict, task_id: str) -> str:
    """The failing task's branch head at failure time — its repair event id.

    The task worktree (recorded by ``t_implement``) is checked out on the task
    branch, so its ``HEAD`` is the branch tip the failure was observed against.
    The same id identifies a second failure of *the same* attempt, which is what
    the one-repair-per-event cap keys on.
    """
    record = (state.get("task_state") or {}).get(task_id) or {}
    worktree = record.get("worktree")
    if not worktree:
        raise GitError(
            f"task {task_id} has no recorded worktree; cannot compute its repair event"
        )
    return head_sha(Path(worktree))


def spawn_repair(state: dict, *, parent_id: str, reason: str) -> dict:
    """Build the state patch that adds one Repair task for ``parent_id``.

    The repair mirrors its parent's contract — same ``files_allowed``, ``checks``
    and ``post_merge_checks`` — with ``depends_on=[]``: the parent is
    Approved-but-not-Integrated, so listing it as a dependency would block the
    repair forever. Its ``task_state`` records ``origin="repair"``, ``parent`` and
    the ``repair_event`` the cap is keyed on.
    """
    parent = TaskDef(**state["tasks"][parent_id])
    rep_id = f"{parent_id}-repair"
    data = parent.model_dump()
    data.update(
        {
            "id": rep_id,
            "title": f"Repair: {reason} for {parent_id}",
            "description": f"Repair the {reason} integration failure of {parent_id}.",
            "depends_on": [],
            "acceptance_criteria": [
                *parent.acceptance_criteria,
                f"Repair resolves the {reason} integration failure",
            ],
        }
    )
    repair = TaskDef(**data)

    record = new_task_state(rep_id, origin="repair", parent=parent_id)
    record["repair_event"] = failure_event_id(state, parent_id)
    return {
        "tasks": {**state["tasks"], rep_id: repair.model_dump(mode="json")},
        "task_state": {rep_id: record},
    }


def _repair_or_escalate(state: dict, tid: str, reason: str, *, detail: str = "") -> dict:
    """One Repair per failure event; a repair never spawns a nested repair.

    Any refusal escalates ``repair_exhausted`` for ``tid``. Otherwise the patch
    from :func:`spawn_repair` is returned with no escalation set, so the existing
    ``integrate → scheduler`` route picks the repair up.
    """
    record = state["task_state"].get(tid) or {}
    if record.get("origin") == "repair":
        return {
            "escalation": escalation_record(
                "repair_exhausted",
                f"repair {tid} failed integration ({reason}): {detail}; no nested repair",
                tid,
            )
        }

    event_id = failure_event_id(state, tid)
    for other, rec in (state.get("task_state") or {}).items():
        if rec.get("origin") == "repair" and rec.get("repair_event") == event_id:
            return {
                "escalation": escalation_record(
                    "repair_exhausted",
                    f"{detail}; a repair ({other}) already covers failure event {event_id}",
                    tid,
                )
            }
    return spawn_repair(state, parent_id=tid, reason=reason)


def _integrated_update(state: dict, tid: str, merge_sha: str | None) -> dict:
    """Mark ``tid`` Integrated — and resolve its parent when ``tid`` is a repair.

    ``merge_sha`` is the integration-branch merge commit to append to the task's
    ``commits``; ``None`` on the idempotent path, where the merge already
    happened and its sha was recorded on the first pass.
    """
    record = state["task_state"].get(tid) or {}
    patch: dict = {"status": TStatus.INTEGRATED}
    if merge_sha is not None:
        patch["commits"] = [*(record.get("commits") or []), merge_sha]
    update: dict = {tid: patch}

    parent = record.get("parent") if record.get("origin") == "repair" else None
    if parent and parent in (state.get("tasks") or {}):
        p_record = state["task_state"].get(parent) or {}
        p_patch: dict = {"status": TStatus.INTEGRATED}
        if merge_sha is not None:
            p_patch["commits"] = [*(p_record.get("commits") or []), merge_sha]
        update[parent] = p_patch
    return update


def integrate(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> dict:
    """Merge one Approved task, then re-run its checks against the merged result.

    Returns a plain state patch (the graph routes on ``state["escalation"]``):
    green post-merge checks → ``INTEGRATED``, a conflict or a red post-merge run →
    a Repair task, or — once the failure event is already repaired — a
    ``repair_exhausted`` escalation.
    """
    tid = payload["task_id"]
    record = state["task_state"].get(tid) or {}
    if not is_approved(record.get("status")):
        # Nothing to merge: the task is already Integrated, or the run is stopping.
        return {}

    worktree = integration_worktree(cfg)
    branch = task_branch(tid)

    with _INTEGRATION_LOCK:
        if is_ancestor(worktree, branch):
            return {"task_state": _integrated_update(state, tid, None)}

        ok, conflicts = merge_no_ff(worktree, branch)
        if not ok:
            return _repair_or_escalate(
                state,
                tid,
                "merge_conflict",
                detail=f"merging {branch} conflicted: {conflicts}",
            )

        task = TaskDef(**state["tasks"][tid])
        checks = dict(task.checks)
        if task.post_merge_checks:
            checks.update(task.post_merge_checks)
        try:
            results = run_checks(
                checks,
                worktree=worktree,
                evidence_dir=run_dir.evidence,
                task_id=f"{tid}.postmerge",
                env=passthrough_env(cfg),
                timeout_s=cfg.timeout_minutes * 60,
            )
        except OSError as e:
            return {"escalation": escalation_record("check_spawn_error", str(e), tid)}

        if not checks_green(results):
            failed = [r.name for r in results if r.exit_code != 0]
            return _repair_or_escalate(
                state,
                tid,
                "post_merge_checks",
                detail=f"post-merge checks failed: {', '.join(failed)}",
            )

        return {"task_state": _integrated_update(state, tid, head_sha(worktree))}
