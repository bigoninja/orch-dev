"""Run artifacts: the run directory, atomic writes, ``status.md``, and the run lock.

One Run owns one directory under ``runs/<run-id>/`` (spec §11) holding its
``plan.yaml``, its machine-readable ``state.json``, its human-readable
``status.md``, the checkpoint database, and the ``evidence/``+``logs/`` dirs.
Every file that a crash might interrupt (``state.json``, ``status.md``) is
written **atomically** — temp file in the same directory, then ``os.replace`` —
so a reader never sees a half-written status.

The **run lock** (``state/orch.lock``) is what stops two Runs from driving the
same state directory at once; a second acquisition fails fast with
:class:`~orch.errors.LockHeldError` rather than interleaving two runs.
"""

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from orch.errors import LockHeldError
from orch.state import TStatus

#: The brief is truncated to this many characters in the status header.
BRIEF_MAX = 60


@dataclass(frozen=True)
class RunDir:
    """The paths of one Run's artifact directory, with ``evidence/``+``logs/`` present."""

    root: Path
    evidence: Path
    logs: Path

    @classmethod
    def create(cls, runs_dir: Path, run_id: str) -> "RunDir":
        """Create (or adopt) ``<runs_dir>/<run_id>/{evidence,logs}``.

        Idempotent: re-entering an existing run directory keeps whatever
        evidence and logs are already there.
        """
        root = Path(runs_dir) / run_id
        evidence = root / "evidence"
        logs = root / "logs"
        evidence.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        return cls(root=root, evidence=evidence, logs=logs)

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.yaml"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.md"

    @property
    def checkpoints_path(self) -> Path:
        return self.root / "checkpoints.db"


def atomic_write(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory.

    Same directory matters: ``os.replace`` is only atomic within one filesystem.
    The temp file is removed if anything fails, so a crash leaves either the old
    file or the new one — never a stray ``.tmp``. The file keeps ``mkstemp``'s
    ``0600`` mode: run artifacts are never meant to be world-readable (§13).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _status_value(status) -> str:
    """The serialized value of a status: ``integrated``, not ``TStatus.INTEGRATED``."""
    return status.value if isinstance(status, TStatus) else str(status)


def _note(state: dict) -> str:
    """One line of why a task is stuck: last error, else first violation, else ``-``."""
    raw = state.get("last_error")
    if not raw:
        violation = state.get("violation")
        if violation:
            raw = violation[0] if isinstance(violation, list) else violation
    if not raw:
        return "-"
    # A markdown cell: no newlines (would break the table) and pipes escaped.
    return " ".join(str(raw).splitlines()).replace("|", r"\|")


def _review(state: dict) -> str:
    """The verdict of the task's most recent review, or ``-`` if never reviewed."""
    verdicts = state.get("verdicts") or []
    if not verdicts:
        return "-"
    return str(verdicts[-1].get("verdict", "-"))


def _attempts(state: dict) -> str:
    attempts = state.get("attempts", 0)
    fix_attempts = state.get("fix_attempts", 0)
    return f"{attempts}/fix:{fix_attempts}" if fix_attempts else str(attempts)


def render_status_md(
    *,
    run_id: str,
    brief: str,
    integration_branch: str,
    tasks: dict[str, dict],
    task_state: dict[str, dict],
    escalation: dict | None,
) -> str:
    """Render ``status.md``: run state, integration progress, the board, escalations.

    The board has one row per task in ``tasks`` order (the plan's order), which
    keeps the file diffable across snapshots.
    """
    run_state = "ESCALATED" if escalation is not None else "RUNNING"
    integrated = sum(
        1
        for state in task_state.values()
        if _status_value(state.get("status")) == TStatus.INTEGRATED.value
    )

    lines = [
        f"Run {run_id} — {brief[:BRIEF_MAX]} — {run_state}",
        f"Integration: {integration_branch} @ {integrated} integrated",
        "| Task | State | Attempts | Review | Notes |",
        "|---|---|---|---|---|",
    ]
    for task_id in tasks:
        state = task_state.get(task_id, {})
        lines.append(
            f"| {task_id} | {_status_value(state.get('status', 'unknown'))} "
            f"| {_attempts(state)} | {_review(state)} | {_note(state)} |"
        )

    if escalation is not None:
        task_id = escalation.get("task_id")
        lines += [
            "## Escalation",
            f"kind: {escalation.get('kind', '-')}",
            f"message: {escalation.get('message', '-')}",
            f"task_id: {task_id if task_id is not None else '-'}",
            "fix the cause, then: orch resume",
        ]

    return "\n".join(lines) + "\n"


def write_status(run_dir: RunDir, **kwargs) -> None:
    """Regenerate ``status.md`` from the current state."""
    atomic_write(run_dir.status_path, render_status_md(**kwargs))


def _json_default(obj):
    """Serialize plan models and enums that reach ``state.json``."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")  # pydantic: TaskDef and friends
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__} to JSON")


def snapshot_state(
    run_dir: RunDir,
    *,
    run_id: str,
    brief: str,
    integration_branch: str,
    tasks: dict[str, dict],
    task_state: dict[str, dict],
    escalation: dict | None,
    summary: str | None = None,
) -> None:
    """Persist the live state: atomic ``state.json`` plus a regenerated ``status.md``.

    Called by every pipeline node (pre-flight ruling R3), so the on-disk
    snapshot always matches the checkpointed graph state. ``summary`` is written
    when present so a finished run is detectable from ``state.json`` alone.
    """
    payload = {
        "run_id": run_id,
        "brief": brief,
        "integration_branch": integration_branch,
        "tasks": tasks,
        "task_state": {
            task_id: {**state, "status": _status_value(state.get("status"))}
            for task_id, state in task_state.items()
        },
        "escalation": escalation,
        "summary": summary,
    }
    atomic_write(run_dir.state_path, json.dumps(payload, indent=2, default=_json_default) + "\n")
    write_status(
        run_dir,
        run_id=run_id,
        brief=brief,
        integration_branch=integration_branch,
        tasks=tasks,
        task_state=task_state,
        escalation=escalation,
    )


@contextmanager
def run_lock(state_dir: Path) -> Iterator[Path]:
    """Hold the run lock in ``state_dir`` for the duration of the block.

    Non-blocking: if another Run holds the lock, raise :class:`LockHeldError`
    immediately. The lock is released (and the descriptor closed) on exit, even
    when the block raises.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "orch.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise LockHeldError(f"another run holds the lock at {lock_path}") from exc
    try:
        yield lock_path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
