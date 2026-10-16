"""Opt-in smoke test for risk R2: two concurrent real OpenCode sessions.

The scheduler runs several tasks at once through a *single* adapter instance,
each in its own worktree. This test reproduces that shape with the real CLI: two
agent turns in flight simultaneously, in two worktrees of the same repository,
asking for distinct files. It is **skipped by default** — opt in with
``ORCH_SMOKE=1``:

    ORCH_SMOKE=1 uv run pytest tests/smoke -v -m smoke

What it proves, and why each assertion is here:

* Both runs finish with ``exit_code == 0`` and neither ``timed_out`` — two
  headless sessions can coexist without one starving or clobbering the other.
* Each worktree holds only the file its own prompt asked for — the runs did not
  leak into each other's ``cwd``.
* The two ``session_id``s differ — OpenCode minted independent sessions, so
  per-task resume (fix loops) can never cross wires.

A single shared adapter instance is used on purpose: that mirrors the real
factory (one adapter per run, fanned out across worker threads), and the adapter
is stateless per call.

Environment note (verified during Task 18): ``opencode run`` resolves the project
it writes into from the ``PWD`` environment variable, not ``getcwd()``. The
adapter sets the child directory via ``Popen(cwd=...)``, which leaves the
inherited ``PWD`` stale; inside a nested OpenCode session the two worktrees would
therefore collapse onto the host project and the isolation assertion would be
meaningless. A real shell exports ``PWD == getcwd()``, so each worker passes
``env={"PWD": str(cwd)}`` to restore that invariant. The assertions are
unchanged: files must land in their own worktree.
"""

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orch import gitops
from orch.adapters.opencode import OpenCodeAdapter

MODEL = "opencode-go/deepseek-v4.1-flash"

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.environ.get("ORCH_SMOKE") != "1",
        reason="real model call; set ORCH_SMOKE=1 to opt in",
    ),
]


def _init_repo(path: Path) -> Path:
    """Create a fresh git repository at ``path`` with one commit on ``main``.

    A commit is required so sibling worktrees can be created off ``main``.
    """
    path.mkdir(parents=True, exist_ok=True)

    def g(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
        )

    g("init", "-b", "main")
    (path / "README.md").write_text("smoke fixture\n")
    g("add", "-A")
    g(
        "-c",
        "user.name=orch",
        "-c",
        "user.email=orch@local",
        "commit",
        "-m",
        "init",
    )
    return path


def test_two_concurrent_sessions_stay_isolated(tmp_path: Path, record_property) -> None:
    repo = _init_repo(tmp_path / "fixture")
    # Two worktrees of the same repo, exactly as the scheduler lays tasks out.
    gitops.create_branch(repo, "smoke-a", "main")
    gitops.create_branch(repo, "smoke-b", "main")
    wt_one = gitops.ensure_worktree(repo, tmp_path / "wt-one", "smoke-a")
    wt_two = gitops.ensure_worktree(repo, tmp_path / "wt-two", "smoke-b")

    adapter = OpenCodeAdapter(model=MODEL, timeout_s=120, extra_env={})

    def run_one(cwd: Path, name: str):
        return adapter.run(
            role="implement",
            prompt=f"Create a file {name}.txt containing exactly: {name}",
            cwd=cwd,
            session_id=None,
            title=f"smoke/{name.upper()}/implement",
            env={"PWD": str(cwd)},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_one = pool.submit(run_one, wt_one, "one")
        future_two = pool.submit(run_one, wt_two, "two")
        result_one = future_one.result()
        result_two = future_two.result()

    for result, name in ((result_one, "one"), (result_two, "two")):
        assert result.exit_code == 0, f"{name}: {result.stderr}"
        assert result.timed_out is False
        assert result.session_id is not None
        assert re.match(r"^ses_", result.session_id), result.session_id

    assert (wt_one / "one.txt").read_text().strip() == "one"
    assert (wt_two / "two.txt").read_text().strip() == "two"

    assert result_one.session_id != result_two.session_id, (
        "concurrent runs reused a session id: " f"{result_one.session_id}"
    )

    record_property("session_id_one", result_one.session_id)
    record_property("session_id_two", result_two.session_id)
