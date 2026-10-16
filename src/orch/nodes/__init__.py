"""Node helpers shared across the orchestrator's pipeline nodes.

Nothing here knows about LangGraph: these are the small pure-ish utilities the
task/integrate/report nodes all need — the passthrough environment, the plan
brief a snapshot needs, and a reducer-faithful view of "state plus this node's
pending updates" used both for routing decisions and for ``snapshot_state``.
"""

import os

from orch.artifacts import RunDir, snapshot_state
from orch.config import Config
from orch.errors import PlanValidationError
from orch.plan import load_plan
from orch.state import escalation_reducer, merge_task_state


def escalation_record(kind: str, message: str, task_id: str | None) -> dict:
    """The canonical escalation payload: ``{"kind", "message", "task_id"}``."""
    return {"kind": kind, "message": message, "task_id": task_id}


def passthrough_env(cfg: Config) -> dict[str, str]:
    """The configured environment variables that actually exist in ``os.environ``.

    Secrets reach a subprocess only because the operator listed them in
    ``cfg.env_passthrough``; a missing name is skipped rather than exported empty.
    """
    return {k: os.environ[k] for k in cfg.env_passthrough if k in os.environ}


def brief_of(run_dir: RunDir) -> str:
    """The plan's brief, for the ``status.md`` header.

    ``brief`` is not part of the graph state (``GState`` is pinned), so snapshots
    re-read it from ``plan.yaml`` — which is immutable after approval. A missing
    or unreadable plan degrades to an empty brief instead of failing a snapshot.
    """
    try:
        return load_plan(run_dir.plan_path).brief
    except PlanValidationError:
        return ""


def merge_updates(state: dict, updates: dict) -> dict:
    """``state`` folded with ``updates`` the way the graph's reducers would fold them.

    A node decides its route and refreshes its snapshot *before* LangGraph commits
    its writes, so both need the projected state, not the stale one. ``task_state``
    is shallow-merged per task, ``escalation`` keeps the first value, and every
    other key is replaced.
    """
    view = dict(state)
    task_state = updates.get("task_state")
    if task_state is not None:
        view["task_state"] = merge_task_state(view.get("task_state") or {}, task_state)
    escalation = updates.get("escalation")
    if escalation is not None:
        view["escalation"] = escalation_reducer(view.get("escalation"), escalation)
    for key, value in updates.items():
        if key not in ("task_state", "escalation"):
            view[key] = value
    return view


def snapshot_run(
    run_dir: RunDir, cfg: Config, state: dict, *, updates: dict | None = None
) -> None:
    """Refresh ``state.json`` and ``status.md`` for ``state`` folded with ``updates``.

    Called on every node's return path (pre-flight ruling R3) so the on-disk
    snapshot always matches the checkpointed graph state.
    """
    view = merge_updates(state, updates or {})
    snapshot_state(
        run_dir,
        run_id=view.get("run_id") or "",
        brief=brief_of(run_dir),
        integration_branch=cfg.integration_branch,
        tasks=view.get("tasks") or {},
        task_state=view.get("task_state") or {},
        escalation=view.get("escalation"),
        summary=view.get("summary"),
    )
