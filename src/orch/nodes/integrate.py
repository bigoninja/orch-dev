"""Merge an Approved task into the integration branch and re-run its checks.

The integration worktree is shared, so only one merge — and its post-merge
checks — may touch it at a time: parallel task branches queue on a module lock.
Merging is idempotent: a branch already reachable from the integration ``HEAD``
is recognized by ``is_ancestor`` and never merged twice (Review Focus #5).
"""

import threading
from pathlib import Path

from orch.adapters.base import AgentAdapter
from orch.artifacts import RunDir
from orch.checks import checks_green, run_checks
from orch.config import Config
from orch.gitops import head_sha, is_ancestor, merge_no_ff
from orch.nodes import escalation_record, passthrough_env
from orch.nodes.task import task_branch
from orch.plan import TaskDef
from orch.state import TStatus

#: One git working tree, one writer: integrations run one at a time.
_INTEGRATION_LOCK = threading.Lock()

INTEGRATION_WORKTREE = "orch-integration"


def integration_worktree(cfg: Config) -> Path:
    """``<target>/<worktree_root>/orch-integration`` — the shared integration tree."""
    return cfg.target_repo / cfg.worktree_root / INTEGRATION_WORKTREE


def is_approved(value) -> bool:
    """True for an ``APPROVED`` status, whether a ``TStatus`` or its string form."""
    return value == TStatus.APPROVED


def integrate(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> dict:
    """Merge one Approved task, then re-run its checks against the merged result.

    Returns a plain state patch (the graph routes on ``state["escalation"]``):
    green post-merge checks → ``INTEGRATED``, anything else → an escalation.
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
            return {"task_state": {tid: {"status": TStatus.INTEGRATED}}}

        ok, conflicts = merge_no_ff(worktree, branch)
        if not ok:
            return {
                "escalation": escalation_record(
                    "merge_conflict", f"merging {branch} conflicted: {conflicts}", tid
                )
            }

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
            return {
                "escalation": escalation_record(
                    "post_merge_checks", f"post-merge checks failed: {', '.join(failed)}", tid
                )
            }

        commits = [*(record.get("commits") or []), head_sha(worktree)]
        return {"task_state": {tid: {"status": TStatus.INTEGRATED, "commits": commits}}}
