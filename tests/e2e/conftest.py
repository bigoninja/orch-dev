"""Shared fixtures and helpers for the end-to-end orchestrator tests.

The e2e tests drive the real compiled graph against a throwaway git repository:
they write only ``plan.yaml`` and a mock-adapter scenario, then invoke the graph
exactly the way the CLI will (first invoke stops at the approval gate, the
harness resumes with the human's decision).
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from orch.artifacts import RunDir
from orch.config import Config
from orch.plan import Plan, TaskDef, dump_plan

#: The demo repository copied into every e2e test (repo root ``fixtures/demo-repo``).
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``git -C <repo> <args>`` with a hermetic identity, raising on failure."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def make_demo_repo(path: Path) -> Path:
    """Copy ``fixtures/demo-repo`` to ``path`` and commit it on a ``main`` branch."""
    shutil.copytree(FIXTURES / "demo-repo", path)
    git(path, "init", "-b", "main")
    git(path, "add", "-A")
    git(
        path,
        "-c",
        "user.name=orch",
        "-c",
        "user.email=orch@local",
        "commit",
        "-m",
        "init",
    )
    return path


@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    """A committed copy of ``fixtures/demo-repo`` on a ``main`` branch."""
    return make_demo_repo(tmp_path / "demo")


def make_config(repo: Path, tmp_path: Path, **over) -> Config:
    """A ``Config`` pointed at ``repo`` with every artifact kept under ``tmp_path``."""
    params = dict(
        target_repo=repo,
        base_ref="main",
        integration_branch="orch/integration",
        worktree_root=".worktrees",
        runs_dir=str(tmp_path / "runs"),
        **over,
    )
    return Config(**params)


def write_plan(run_dir: RunDir, tasks: list[dict], *, rid: str = "R1") -> None:
    """Write ``plan.yaml`` into ``run_dir`` from raw task dicts."""
    plan = Plan(run_id=rid, brief="e2e", source=None, tasks=[TaskDef(**t) for t in tasks])
    dump_plan(plan, run_dir.plan_path)


def run_once(graph, run_id: str, thread: str = "t1"):
    """Invoke the graph once for ``run_id`` on ``thread``."""
    return graph.invoke({"run_id": run_id}, config={"configurable": {"thread_id": thread}})


def plan_sha256(run_dir: RunDir) -> str:
    """The approved-plan hash the approval gate re-checks on resume."""
    return hashlib.sha256(run_dir.plan_path.read_bytes()).hexdigest()
