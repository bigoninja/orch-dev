"""The live activity channel, driven through the real compiled graph.

``state.json`` is written only when a node *returns*, so a board watching a run
cannot tell that a turn is in flight — the reason an implementing task never
appeared in the working column. These tests prove the pipeline nodes publish that
signal *while* an agent runs and clear it afterwards, which is exactly what the
board's "Implementing" column and its spinner depend on.
"""

from pathlib import Path

import pytest
import yaml
from langgraph.types import Command

from orch.activity import read_activity
from orch.adapters import MockAdapter
from orch.artifacts import RunDir
from orch.graph import build_graph
from orch.state import TStatus

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


class ObservingAdapter(MockAdapter):
    """A mock that snapshots the activity channel *during* each turn.

    The snapshot is taken before delegating, so it records what a dashboard
    polling at that instant would have seen.
    """

    def __init__(self, scenario_path: Path, run_dir: RunDir):
        super().__init__(scenario_path)
        self.run_dir = run_dir
        self.during: dict[tuple[str, str], dict] = {}

    def run(self, *, role, prompt, cwd, session_id, title, env):
        parts = title.split("/")
        if len(parts) == 3:
            self.during[(parts[1], parts[2])] = read_activity(self.run_dir)
        return super().run(
            role=role,
            prompt=prompt,
            cwd=cwd,
            session_id=session_id,
            title=title,
            env=env,
        )


@pytest.fixture
def activity_run(tmp_path):
    repo = make_demo_repo(tmp_path / "demo")
    cfg = make_config(repo, tmp_path)
    run_dir = RunDir.create(Path(cfg.runs_dir), "R1")
    write_plan(run_dir, [TASK], rid="R1")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(SCENARIO), encoding="utf-8")
    adapter = ObservingAdapter(scenario_path, run_dir)
    graph = build_graph(cfg=cfg, adapter=adapter, run_dir=run_dir)

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
    return cfg, run_dir, adapter, final


def test_a_turn_is_visible_as_running_while_it_runs(activity_run):
    _cfg, _run_dir, adapter, _final = activity_run

    seen = adapter.during[("A", "implement")]
    assert "A" in seen, "the implement turn must be visible while it is in flight"
    assert seen["A"]["phase"] == "implement"


def test_the_running_entry_is_cleared_when_the_turn_ends(activity_run):
    _cfg, run_dir, _adapter, final = activity_run

    assert final["task_state"]["A"]["status"] == TStatus.INTEGRATED
    assert read_activity(run_dir) == {}, "a finished run must leave no phantom spinners"
