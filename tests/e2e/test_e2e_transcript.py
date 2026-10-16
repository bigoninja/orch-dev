"""Transcript persistence, driven through the real compiled graph.

A task's chat has to survive the turn: the prompt and the agent's own words are
what the dashboard's chat pane shows, and what a human reads when a reviewer
rejects. These tests prove the pipeline nodes actually write it.
"""

from pathlib import Path

import pytest
import yaml
from langgraph.types import Command

from orch.adapters import MockAdapter
from orch.artifacts import RunDir
from orch.graph import build_graph
from orch.state import TStatus
from orch.transcript import read_turns

from conftest import make_config, make_demo_repo, plan_sha256, run_once, write_plan

SCENARIO = {
    "A": {
        "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    },
}

TASK = {
    "id": "A",
    "title": "write OK to a.txt",
    "description": "Write the marker OK to a.txt.",
    "depends_on": [],
    "files_allowed": ["a.txt"],
    "acceptance_criteria": ["a.txt contains OK"],
    "checks": {"test": {"cmd": "grep -q OK a.txt"}},
}


@pytest.fixture
def transcript_run(tmp_path):
    repo = make_demo_repo(tmp_path / "demo")
    cfg = make_config(repo, tmp_path)
    run_dir = RunDir.create(Path(cfg.runs_dir), "R1")
    write_plan(run_dir, [TASK], rid="R1")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(SCENARIO), encoding="utf-8")
    graph = build_graph(cfg=cfg, adapter=MockAdapter(scenario_path), run_dir=run_dir)

    interrupted = run_once(graph, "R1")
    assert interrupted.get("__interrupt__"), "the gate must interrupt the first invoke"
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


def test_a_task_records_its_implement_and_review_turns(transcript_run):
    _cfg, run_dir, _final = transcript_run

    turns = read_turns(run_dir, "A")

    assert [t["phase"] for t in turns] == ["implement", "review"]
    assert turns[0]["prompt"], "the prompt is the first thing a chat pane shows"
    assert turns[0]["role"] == "implement"
    assert turns[1]["role"] == "review"


def test_the_recorded_chat_carries_the_agents_own_words(transcript_run):
    _cfg, run_dir, _final = transcript_run

    implement = read_turns(run_dir, "A")[0]

    assert implement["messages"] == [{"kind": "text", "text": "mock A: applied 1 edit(s)"}]


def test_a_repair_does_not_bleed_into_its_parents_chat(transcript_run):
    """``task-3`` and ``task-3-repair`` share a prefix but are different chats."""
    _cfg, run_dir, _final = transcript_run

    assert read_turns(run_dir, "A-repair") == []
