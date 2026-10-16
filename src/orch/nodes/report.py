"""The terminal node: write the final ``status.md`` and summarize the run."""

from orch.artifacts import RunDir
from orch.config import Config
from orch.nodes import snapshot_run
from orch.state import TStatus


def _status_value(status) -> str:
    """The serialized value of a status: ``integrated``, not ``TStatus.INTEGRATED``."""
    return status.value if isinstance(status, TStatus) else str(status)


def report(state: dict, *, cfg: Config, run_dir: RunDir) -> dict:
    """Refresh the run's artifacts and return the one-line summary.

    The graph's terminal node, so this snapshot is the final ``state.json`` and
    ``status.md`` a human reads.
    """
    tasks = state.get("tasks") or {}
    task_state = state.get("task_state") or {}
    escalation = state.get("escalation")

    integrated = [
        tid
        for tid, record in task_state.items()
        if _status_value(record.get("status")) == TStatus.INTEGRATED.value
    ]
    summary = f"run {state.get('run_id')}: {len(integrated)}/{len(tasks)} integrated"
    if escalation is not None:
        summary += f" — escalated: {escalation.get('kind')}"

    # Fold the summary into the final snapshot so `state.json` marks the run over
    # (the CLI's latest-run selection and the dashboard both read it).
    snapshot_run(run_dir, cfg, {**state, "summary": summary})
    return {"summary": summary}
