"""End-to-end parallel-branch isolation for the task pipeline.

The graph's spine recovers the shared state through langgraph private internals
(``graph.pipeline_node`` → ``_shared_state`` → ``CONFIG_KEY_READ``) because a
``Send`` payload replaces the node's input. That read is load-bearing: a stale
view, or another branch's pending ``task_state`` patch, would make a node route
silently wrong instead of failing loudly.

Two independent tasks are dispatched into the *same* superstep (no deps, and the
default concurrency cap admits both), so their branches genuinely overlap. The
assertions pin the observable consequences of per-branch isolation: each task is
processed exactly once, each task branch commits only its own file, and each
reviewer's diff artifact names only its own file. A full break is already caught
by the happy-path E2E; this suite is what catches a *semantics* drift in the read.
"""

from pathlib import Path

import pytest
import yaml
from langgraph.types import Command

from orch.adapters import MockAdapter
from orch.artifacts import RunDir
from orch.graph import build_graph
from orch.state import TStatus

from conftest import git, make_config, make_demo_repo, plan_sha256, run_once, write_plan

#: ``src`` and ``dest`` per task, so a sibling bleed is visible in every artifact.
FILES = {"A": "a.txt", "B": "b.txt"}

#: The checkpointed thread the first invoke and the approval resume must share.
THREAD = "parallel"

TASKS = [
    {
        "id": tid,
        "title": f"write OK to {name}",
        "description": f"Write the marker OK to {name}.",
        "depends_on": [],
        "files_allowed": [name],
        "acceptance_criteria": [f"{name} contains OK"],
        "checks": {"test": {"cmd": f"grep -q OK {name}"}},
    }
    for tid, name in FILES.items()
]

#: Both implementers edit their own file uncommitted; both reviewers approve.
SCENARIO = {
    tid: {
        "implement": [{"edits": [{"file": name, "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    }
    for tid, name in FILES.items()
}


@pytest.fixture(scope="module")
def parallel_run(tmp_path_factory):
    """Run a two-task, no-dependency plan to quiescence once.

    Returns ``(cfg, run_dir, final_state)``. Both tasks are Ready at the first
    ``scheduler`` call, so both are dispatched in one superstep — the branches
    overlap for the whole pipeline, which is the condition this suite probes.
    """
    tmp_path = tmp_path_factory.mktemp("parallel_isolation")
    repo = make_demo_repo(tmp_path / "demo")
    cfg = make_config(repo, tmp_path)
    assert cfg.concurrency >= 2, "both tasks must be admitted at once"

    run_dir = RunDir.create(Path(cfg.runs_dir), "P1")
    write_plan(run_dir, TASKS, rid="P1")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(SCENARIO), encoding="utf-8")
    graph = build_graph(cfg=cfg, adapter=MockAdapter(scenario_path), run_dir=run_dir)

    interrupted = run_once(graph, "P1", thread=THREAD)
    assert interrupted.get("__interrupt__"), "the approval gate must interrupt the first invoke"

    final = graph.invoke(
        Command(
            resume={
                "approved": True,
                "pre_authorized": [],
                "plan_sha256": plan_sha256(run_dir),
            }
        ),
        config={"configurable": {"thread_id": THREAD}},
    )
    return cfg, run_dir, final


def test_both_parallel_tasks_integrate(parallel_run):
    """Neither branch is lost or escalated by the other's progress."""
    _cfg, _run_dir, final = parallel_run

    assert set(final["task_state"]) == {"A", "B"}
    for tid, state in final["task_state"].items():
        assert state["status"] == TStatus.INTEGRATED, f"{tid} ended {state['status']!r}"
    assert final.get("escalation") is None


def test_each_task_is_processed_exactly_once(parallel_run):
    """A sibling-read that replayed or skipped work would shift these counters."""
    _cfg, _run_dir, final = parallel_run

    for tid in FILES:
        state = final["task_state"][tid]
        assert state["attempts"] == 1, f"{tid} ran implement {state['attempts']} time(s)"
        assert len(state["verdicts"]) == 1, f"{tid} collected {len(state['verdicts'])} verdict(s)"
        assert state["session_id"], f"{tid} lost its implement session id"


def test_each_task_branch_commits_only_its_own_file(parallel_run):
    """A mis-routed commit — the wrong worktree, or a sibling's edits — shows here."""
    cfg, _run_dir, _final = parallel_run

    for tid, name in FILES.items():
        touched = git(
            cfg.target_repo, "diff", "--name-only", f"{cfg.base_ref}..agent/{tid}"
        ).stdout.splitlines()
        assert touched == [name], f"agent/{tid} touched {touched}"

        files = git(cfg.target_repo, "show", "--stat", "--format=%H", f"agent/{tid}").stdout
        assert sum(1 for line in files.splitlines() if "|" in line) == 1
        assert name in files


def test_each_reviewer_diff_names_only_its_own_file(parallel_run):
    """Each reviewer must see its own task's diff, never a sibling's changes."""
    _cfg, run_dir, _final = parallel_run

    for tid, name in FILES.items():
        text = (run_dir.evidence / f"{tid}.diff").read_text(encoding="utf-8")
        assert f"a/{name}" in text, f"{tid}.diff does not show {name}"
        for other in FILES.values():
            if other != name:
                assert other not in text, f"{tid}.diff leaked {other}"


def test_each_worktree_holds_only_its_own_edit(parallel_run):
    """No branch edited a sibling's file in its own private working tree."""
    cfg, _run_dir, _final = parallel_run

    for tid, name in FILES.items():
        worktree = cfg.target_repo / cfg.worktree_root / tid
        assert "OK" in (worktree / name).read_text(encoding="utf-8")
        for other in FILES.values():
            if other != name:
                assert (worktree / other).read_text(encoding="utf-8").strip() == "original"
