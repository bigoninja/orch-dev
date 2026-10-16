"""Deterministic pass/fail truth: run a task's checks and capture evidence.

Each check is a shell command from the plan's ``checks`` mapping, run in the
task worktree (optionally in a ``cwd`` subdirectory) with the run's environment.
Checks run **sequentially in plan-dict order** so a task's outcome is
reproducible; there is no parallelism here and no model in the loop.

Two properties are load-bearing:

* **bytes, not text.** Command output is captured as raw bytes and written to
  the evidence file verbatim. A compiler that emits non-UTF-8 garbage must not
  crash the runner (Review Focus #2); callers that want to *display* an evidence
  file decode it with ``errors="replace"``.
* **explicit failure codes.** A timeout kills the whole process group
  (``start_new_session=True`` + ``os.killpg``) and records ``exit_code == -9``
  rather than trusting whatever ``returncode`` the reaped child reports, so a
  killed check can never be mistaken for a green one.
"""

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from orch.plan import Check

#: Exit code recorded for a check killed on timeout (SIGKILL).
TIMEOUT_EXIT_CODE = -9


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome plus the path to its evidence file."""

    name: str
    cmd: str
    cwd: str | None
    exit_code: int
    duration_s: float
    output_path: Path


def checks_green(results: list[CheckResult]) -> bool:
    """True only when every check exited 0 (empty is vacuously green)."""
    return all(r.exit_code == 0 for r in results)


def _evidence_path(evidence_dir: Path, task_id: str, name: str) -> Path:
    return evidence_dir / f"{task_id}.check.{name}.log"


def run_checks(
    checks: dict[str, Check],
    *,
    worktree: Path,
    evidence_dir: Path,
    task_id: str,
    env: dict[str, str],
    timeout_s: int,
) -> list[CheckResult]:
    """Run every check in ``checks`` in plan-dict order and return the results.

    ``worktree`` is the task's working tree; a check's relative ``cwd`` is
    resolved against it. Each check gets one evidence file at
    ``<evidence_dir>/<task_id>.check.<name>.log`` holding a UTF-8 header (name,
    command, cwd, exit code, duration) followed by the raw stdout and stderr
    bytes. ``timeout_s`` is per check.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return [
        _run_check(
            name,
            check,
            worktree=worktree,
            evidence_dir=evidence_dir,
            task_id=task_id,
            env=env,
            timeout_s=timeout_s,
        )
        for name, check in checks.items()
    ]


def _run_check(
    name: str,
    check: Check,
    *,
    worktree: Path,
    evidence_dir: Path,
    task_id: str,
    env: dict[str, str],
    timeout_s: int,
) -> CheckResult:
    output_path = _evidence_path(evidence_dir, task_id, name)
    started = time.monotonic()

    proc = subprocess.Popen(
        check.cmd,
        shell=True,
        cwd=str(worktree / (check.cwd or "")),
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        # Kill the whole process group: the check may have spawned children.
        # The child can exit in the race window, hence the guard.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        exit_code = TIMEOUT_EXIT_CODE

    duration_s = time.monotonic() - started

    header = (
        f"CHECK {name}\n"
        f"CMD: {check.cmd}\n"
        f"CWD: {check.cwd or '.'}\n"
        f"EXIT: {exit_code}\n"
        f"DURATION_S: {duration_s:.3f}\n"
    ).encode("utf-8")
    output_path.write_bytes(header + b"STDOUT\n" + stdout + b"\nSTDERR\n" + stderr)

    return CheckResult(
        name=name,
        cmd=check.cmd,
        cwd=check.cwd,
        exit_code=exit_code,
        duration_s=duration_s,
        output_path=output_path,
    )
