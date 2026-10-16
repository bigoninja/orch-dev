"""Opt-in smoke test: one real ``OpenCodeAdapter`` run against the pinned model.

Unlike ``tests/unit/test_opencode_adapter.py`` (which shims the ``opencode``
binary), this exercises the real CLI, the real provider, and the real event
stream end to end. It therefore spends flash tokens and is **skipped by
default** — opt in with ``ORCH_SMOKE=1``:

    ORCH_SMOKE=1 uv run pytest tests/smoke -v -m smoke

What it proves, and why each assertion is here:

* ``session_id`` matches ``^ses_`` — the adapter decoded the ``--format json``
  stream and captured the resumable session id (fix loops depend on it).
* ``exit_code == 0`` and ``timed_out is False`` — the run finished on its own.
* ``hello.txt`` exists with stripped content ``hello`` — ``--auto`` let the model
  actually use its file tools in ``cwd``.
* ``gitops.commit_all`` returns a sha — the agent's write is uncommitted, so the
  integrator's stage-and-commit path has something to commit.

Environment note: ``opencode run`` resolves the project it reads/writes from the
``PWD`` environment variable, not ``getcwd()``. ``OpenCodeAdapter`` now exports
``PWD == cwd`` (see :meth:`OpenCodeAdapter.run`), so this test passes ``env={}``
and still requires the file to land in the worktree ``cwd``.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from orch import gitops
from orch.adapters.opencode import OpenCodeAdapter

#: The pinned model from ``orch.config.Config``; never substituted (no-substitution law).
MODEL = "opencode/deepseek-v4.1-flash-free"
PROMPT = "Create a file hello.txt containing exactly: hello"

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.environ.get("ORCH_SMOKE") != "1",
        reason="real model call; set ORCH_SMOKE=1 to opt in",
    ),
]


def _init_repo(path: Path) -> Path:
    """Create a fresh git repository at ``path`` with a ``main`` branch."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(path), "init", "-b", "main"],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def test_real_adapter_writes_file_and_commits(tmp_path: Path, record_property) -> None:
    repo = _init_repo(tmp_path / "fixture")

    adapter = OpenCodeAdapter(model=MODEL, timeout_s=120, extra_env={})
    result = adapter.run(
        role="implement",
        prompt=PROMPT,
        cwd=repo,
        session_id=None,
        title="smoke/SMOKE/implement",
        env={},
    )

    assert result.session_id is not None
    assert re.match(r"^ses_", result.session_id), result.session_id
    record_property("session_id", result.session_id)
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False

    hello = repo / "hello.txt"
    assert hello.exists(), f"agent produced no hello.txt; stdout={result.stdout[:500]!r}"
    assert hello.read_text().strip() == "hello"

    sha = gitops.commit_all(repo, "smoke")
    assert sha is not None, "hello.txt was already committed; expected it uncommitted"
    assert re.match(r"^[0-9a-f]{40}$", sha), sha
