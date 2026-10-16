"""The live activity channel: who is running *right now* (``activity.json``).

``state.json`` is written only when a pipeline node *returns* —
``graph.pipeline_node`` snapshots on the way out — so during a multi-minute
``adapter.run(...)`` the run's snapshot still shows the *previous* status. That is
why a task being implemented never appeared in the dashboard's working column:
there was no mid-node signal at all.

This module is that signal: a small map of in-flight turns, keyed by task id,
written at dispatch and cleared in a ``finally``. It is deliberately a *separate*
file rather than an optimistic ``state.json`` write, so ``state.json`` stays the
authoritative, checkpointed snapshot and this stays advisory.

Advisory means nothing about a run's correctness depends on it: a lost, corrupt or
stale entry can only mislead a UI, so every read degrades to "nothing running"
rather than raising. Parallel branches (``concurrency`` up to 8) dispatch at once,
so a read-modify-write is serialized to stop one sibling's entry being dropped.
"""

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from orch.artifacts import RunDir, atomic_write

#: One file, many writer threads: parallel task branches dispatch concurrently.
_LOCK = threading.Lock()


def activity_path(run_dir: RunDir) -> Path:
    """``<run>/activity.json`` — the live map of in-flight turns."""
    return run_dir.root / "activity.json"


def read_activity(run_dir: RunDir) -> dict[str, dict]:
    """The in-flight turns, keyed by task id. Empty when absent or unreadable."""
    try:
        data = json.loads(activity_path(run_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _now() -> str:
    """An ISO-8601 UTC timestamp — what lets the UI show how long a turn has run."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def activity_begin(
    run_dir: RunDir, task_id: str, *, phase: str, role: str, model: str, title: str = ""
) -> None:
    """Record that ``task_id`` has a turn in flight; a re-dispatch overwrites."""
    with _LOCK:
        entries = read_activity(run_dir)
        entries[task_id] = {
            "task_id": task_id,
            "phase": phase,
            "role": role,
            "model": model,
            "title": title,
            "started_at": _now(),
        }
        _write(run_dir, entries)


def activity_end(run_dir: RunDir, task_id: str) -> None:
    """Clear ``task_id``'s in-flight entry; a no-op when it was never set."""
    with _LOCK:
        entries = read_activity(run_dir)
        if entries.pop(task_id, None) is None:
            return
        _write(run_dir, entries)


def _write(run_dir: RunDir, entries: dict[str, dict]) -> None:
    atomic_write(activity_path(run_dir), json.dumps(entries, indent=2) + "\n")


@contextmanager
def running(
    run_dir: RunDir, task_id: str, *, phase: str, role: str, model: str, title: str = ""
) -> Iterator[None]:
    """Mark a turn in flight for the duration of the block, then clear it.

    The ``finally`` is the whole point: a turn that raises must not leave a
    phantom spinner behind, so cleanup cannot depend on the happy path.
    """
    activity_begin(run_dir, task_id, phase=phase, role=role, model=model, title=title)
    try:
        yield
    finally:
        activity_end(run_dir, task_id)
