"""End-to-end happy path: a three-task plan runs from approval to integration.

Task A and Task B are independent and run concurrently; Task C depends on both.
Every implementer applies its edit uncommitted (the orchestrator owns the
commit), every reviewer approves, and every merge lands on the integration
branch. The test drives the compiled graph with a checkpointed thread exactly
as the CLI will: the first invoke stops at the approval gate, the resume carries
the human's approval plus the pinned plan hash.
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

#: Every implementer edits its own file to ``OK`` and leaves it uncommitted;
#: every reviewer approves.
SCENARIO = {
    "A": {
        "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    },
    "B": {
        "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    },
    "C": {
        "implement": [{"edits": [{"file": "c.txt", "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    },
}


def _task(tid: str, *, depends_on: list[str], check: str) -> dict:
    name = f"{tid.lower()}.txt"
    return {
        "id": tid,
        "title": f"write OK to {name}",
        "description": f"Write the marker OK to {name}.",
        "depends_on": depends_on,
        "files_allowed": [name],
        "acceptance_criteria": [f"{name} contains OK"],
        "checks": {"test": {"cmd": check}},
    }


TASKS = [
    _task("A", depends_on=[], check="grep -q OK a.txt"),
    _task("B", depends_on=[], check="grep -q OK b.txt"),
    _task("C", depends_on=["A", "B"], check="grep -q OK a.txt b.txt c.txt"),
]


@pytest.fixture(scope="module")
def happy_run(tmp_path_factory):
    """Run the plan to quiescence once and return ``(cfg, run_dir, final_state)``."""
    tmp_path = tmp_path_factory.mktemp("happy_path")
    repo = make_demo_repo(tmp_path / "demo")
    cfg = make_config(repo, tmp_path)
    run_dir = RunDir.create(Path(cfg.runs_dir), "R1")
    write_plan(run_dir, TASKS, rid="R1")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(SCENARIO), encoding="utf-8")
    graph = build_graph(cfg=cfg, adapter=MockAdapter(scenario_path), run_dir=run_dir)

    interrupted = run_once(graph, "R1")
    assert interrupted.get("__interrupt__"), "the approval gate must interrupt the first invoke"

    final = graph.invoke(
        Command(
            resume={
                "approved": True,
                "pre_authorized": [],
                "plan_sha256": plan_sha256(run_dir),
            }
        ),
        config={"configurable": {"thread_id": "t1"}},
    )
    return cfg, run_dir, final


def test_every_task_is_integrated(happy_run):
    _cfg, _run_dir, final = happy_run

    assert set(final["task_state"]) == {"A", "B", "C"}
    for tid, state in final["task_state"].items():
        assert state["status"] == TStatus.INTEGRATED, f"{tid} ended {state['status']!r}"
    assert final.get("escalation") is None


def test_the_integration_worktree_holds_every_edit(happy_run):
    cfg, _run_dir, _final = happy_run

    integration = cfg.target_repo / cfg.worktree_root / "orch-integration"
    assert integration.is_dir()
    assert "orch/integration" in git(cfg.target_repo, "branch", "--list", "orch/integration").stdout

    for name in ("a.txt", "b.txt", "c.txt"):
        assert "OK" in (integration / name).read_text(encoding="utf-8")


def test_each_task_lands_as_a_no_ff_merge(happy_run):
    cfg, _run_dir, _final = happy_run

    integration = cfg.target_repo / cfg.worktree_root / "orch-integration"
    merges = git(integration, "log", "--oneline", "--merges").stdout.splitlines()
    assert len(merges) >= 3


def test_the_main_branch_is_never_touched(happy_run):
    cfg, _run_dir, _final = happy_run

    assert "OK" not in (cfg.target_repo / "a.txt").read_text(encoding="utf-8")


def test_evidence_holds_checks_diffs_and_verdicts(happy_run):
    _cfg, run_dir, _final = happy_run

    for tid in ("A", "B", "C"):
        assert (run_dir.evidence / f"{tid}.diff").is_file()
        assert (run_dir.evidence / f"{tid}.verdict.json").is_file()
        assert (run_dir.evidence / f"{tid}.check.test.log").is_file()


def test_report_returns_the_final_summary(happy_run):
    """The pinned ``report`` contract: a one-line ``<n>/<total>`` summary, no escalation."""
    _cfg, _run_dir, final = happy_run

    assert final["summary"] == "run R1: 3/3 integrated"


def test_status_md_reports_three_integrated_tasks(happy_run):
    _cfg, run_dir, _final = happy_run

    status = run_dir.status_path.read_text(encoding="utf-8")
    assert "@ 3 integrated" in status
    for tid in ("A", "B", "C"):
        assert f"| {tid} | integrated |" in status


def test_the_run_is_checkpointed(happy_run):
    _cfg, run_dir, _final = happy_run

    assert run_dir.checkpoints_path.is_file()
