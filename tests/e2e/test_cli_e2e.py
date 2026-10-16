"""End-to-end CLI tests: ``run`` / ``approve`` / ``resume`` / ``status``.

Every test drives ``orch.cli.main([...])`` **in-process** against a throwaway demo
repo and a throwaway config file, with the token-free mock adapter selected through
the two env vars the factory reads (``ORCH_ADAPTER=mock`` + ``ORCH_MOCK_SCENARIO``).
This is the same contract a real terminal gets: one process, one command, one exit
code (0 success/awaiting-approval, 1 error, 2 escalation).

The pause mechanics mirror ``tests/e2e/conftest.py`` and the failure suites exactly
— the first invoke stops at the approval gate, ``--yes`` resumes with the pinned
approval payload, and an escalation stops on a checkpointed thread that ``resume``
re-enters. The CLI owns the SQLite connection per invocation; each command is a
fresh process, so the mock's per-adapter attempt counters restart (the escalation
test therefore *replaces* the scenario rather than appending to it).
"""

import json
from pathlib import Path

import pytest
import yaml

from orch.artifacts import RunDir, run_lock
from orch.cli import main
from orch.plan import Plan, TaskDef, dump_plan

from conftest import write_plan

#: One task that writes ``OK`` to ``a.txt`` and greps for it.
TASK = {
    "id": "A",
    "title": "write OK to a.txt",
    "description": "Write the marker OK to a.txt.",
    "depends_on": [],
    "files_allowed": ["a.txt"],
    "acceptance_criteria": ["a.txt contains OK"],
    "checks": {"test": {"cmd": "grep -q OK a.txt"}},
}

#: A second, independent task — proves "all tasks" integrated, not just one.
TASK_B = {
    "id": "B",
    "title": "write OK to b.txt",
    "description": "Write the marker OK to b.txt.",
    "depends_on": [],
    "files_allowed": ["b.txt"],
    "acceptance_criteria": ["b.txt contains OK"],
    "checks": {"test": {"cmd": "grep -q OK b.txt"}},
}

#: A clean scenario: one passing implement attempt, one approving review.
SCENARIO = {
    "A": {
        "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    }
}

#: The same, for two independent tasks.
SCENARIO_AB = {
    **SCENARIO,
    "B": {
        "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
        "review": [{"verdict": "approve"}],
    },
}


def _task(tid: str, *, dep: str, file: str) -> TaskDef:
    """A minimal TaskDef for a hand-written (e.g. cyclic) plan."""
    return TaskDef(
        id=tid,
        title=f"write OK to {file}",
        description=f"Write the marker OK to {file}.",
        depends_on=[dep],
        files_allowed=[file],
        acceptance_criteria=[f"{file} contains OK"],
        checks={"test": {"cmd": f"grep -q OK {file}"}},
    )


def _scenario(tmp_path: Path, scenario: dict) -> Path:
    """Write a mock-adapter scenario and return its path."""
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    return path


def _use_mock(monkeypatch, scenario_path: Path) -> None:
    """Select the mock adapter through the exact env the factory reads."""
    monkeypatch.setenv("ORCH_ADAPTER", "mock")
    monkeypatch.setenv("ORCH_MOCK_SCENARIO", str(scenario_path))


def _config(tmp_path: Path, repo: Path, **over) -> Path:
    """Write an ``orch.config.yaml`` keeping every artifact under ``tmp_path``."""
    data = {
        "target_repo": str(repo),
        "runs_dir": str(tmp_path / "runs"),
        "state_dir": str(tmp_path / "state"),
        "concurrency": 2,
        "retries": 2,
    }
    data.update(over)
    path = tmp_path / "orch.config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _boot(tmp_path: Path, repo: Path, monkeypatch, scenario: dict, *, rid="R1", tasks=None):
    """A demo repo + config + scenario + a run dir holding ``plan.yaml``."""
    scenario_path = _scenario(tmp_path, scenario)
    _use_mock(monkeypatch, scenario_path)
    cfg_path = _config(tmp_path, repo)
    run_dir = RunDir.create(Path(tmp_path / "runs"), rid)
    write_plan(run_dir, tasks or [TASK], rid=rid)
    return cfg_path, run_dir


def _state(run_dir: RunDir) -> dict:
    return json.loads(run_dir.state_path.read_text())


# --- happy path ----------------------------------------------------------------


def test_run_yes_completes_and_integrates(tmp_path, demo_repo, monkeypatch, capsys):
    cfg_path, run_dir = _boot(
        tmp_path, demo_repo, monkeypatch, SCENARIO_AB, tasks=[TASK, TASK_B]
    )

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 2/2 integrated" in out
    state = _state(run_dir)
    assert state["task_state"]["A"]["status"] == "integrated"
    assert state["task_state"]["B"]["status"] == "integrated"
    assert run_dir.checkpoints_path.is_file()

    integration = demo_repo / ".worktrees" / "orch-integration"
    for name in ("a.txt", "b.txt"):
        assert "OK" in (integration / name).read_text(encoding="utf-8")


def test_run_stops_at_the_gate_then_approve_completes(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO)

    rc = main(["run", "--config", str(cfg_path)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "awaiting approval" in out
    assert "orch approve --run R1" in out
    assert not _state(run_dir).get("summary"), "the run must not have finished yet"

    rc = main(["approve", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    assert _state(run_dir)["task_state"]["A"]["status"] == "integrated"


# --- approve --edit ------------------------------------------------------------


def test_approve_edit_refuses_an_invalid_plan_and_stays_resumable(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO)
    assert main(["run", "--config", str(cfg_path)]) == 0
    capsys.readouterr()

    # The editor rewrites plan.yaml into a dependency cycle.
    cycle = Plan(
        run_id="R1",
        brief="e2e",
        source=None,
        tasks=[_task("A", dep="B", file="a.txt"), _task("B", dep="A", file="b.txt")],
    )
    cycle_path = tmp_path / "cycle.yaml"
    dump_plan(cycle, cycle_path)
    editor = tmp_path / "editor.sh"
    editor.write_text(f'#!/bin/sh\ncp {cycle_path} "$1"\n', encoding="utf-8")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    rc = main(["approve", "--edit", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "cycle" in out
    assert not _state(run_dir).get("summary"), "a refused approve must not resume the run"

    # Repair the plan and approve again: the run resumes and completes.
    write_plan(run_dir, [TASK], rid="R1")
    rc = main(["approve", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out


# --- status --------------------------------------------------------------------


def test_status_prints_the_board_and_json(tmp_path, demo_repo, monkeypatch, capsys):
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO)
    assert main(["run", "--config", str(cfg_path)]) == 0
    capsys.readouterr()

    rc = main(["status", "--run", "R1", "--config", str(cfg_path)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "| Task | State | Attempts | Review | Notes |" in out
    assert "| A | pending |" in out

    rc = main(["status", "--json", "--run", "R1", "--config", str(cfg_path)])
    out = capsys.readouterr().out
    assert rc == 0, out
    data = json.loads(out)
    assert data["run_id"] == "R1"
    assert "A" in data["tasks"]
    assert data["task_state"]["A"]["status"] == "pending"


def test_status_without_a_run_reports_an_error(tmp_path, demo_repo, monkeypatch, capsys):
    _use_mock(monkeypatch, _scenario(tmp_path, SCENARIO))
    cfg_path = _config(tmp_path, demo_repo)

    rc = main(["status", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "error:" in out


# --- escalation → resume -------------------------------------------------------


def test_escalation_exits_two_then_resume_completes(
    tmp_path, demo_repo, monkeypatch, capsys
):
    exhausted = {
        "A": {
            "implement": [
                {"edits": [{"file": "a.txt", "content": "nope 1"}]},
                {"edits": [{"file": "a.txt", "content": "nope 2"}]},
                {"edits": [{"file": "a.txt", "content": "nope 3"}]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, exhausted)

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 2, out
    assert "task_retries_exhausted" in out
    assert "fixed the cause? then: orch resume" in out
    paused = _state(run_dir)
    assert paused["escalation"]["kind"] == "task_retries_exhausted"
    assert paused["escalation"]["task_id"] == "A"
    assert paused["task_state"]["A"]["status"] == "escalated"

    # The human fixes the environment: replace the scenario so the first attempt
    # passes, then resume — the retry resets the task's budget.
    _scenario(tmp_path, SCENARIO)
    rc = main(["resume", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    assert _state(run_dir)["task_state"]["A"]["status"] == "integrated"


# --- run lock ------------------------------------------------------------------


def test_run_refuses_when_the_lock_is_held(tmp_path, demo_repo, monkeypatch, capsys):
    cfg_path, _run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO)

    with run_lock(Path(tmp_path / "state")):
        rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "another run holds the lock" in out


def test_approve_refuses_when_the_lock_is_held(tmp_path, demo_repo, monkeypatch, capsys):
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO)
    assert main(["run", "--config", str(cfg_path)]) == 0
    capsys.readouterr()

    # A second terminal holds the lock: approve must not touch the run at all.
    with run_lock(Path(tmp_path / "state")):
        rc = main(["approve", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "another run holds the lock" in out
    state = _state(run_dir)
    assert not state.get("summary"), "a locked-out approve must not resume the run"
    assert state["task_state"]["A"]["status"] == "pending"


def test_resume_refuses_when_the_lock_is_held(tmp_path, demo_repo, monkeypatch, capsys):
    exhausted = {
        "A": {
            "implement": [
                {"edits": [{"file": "a.txt", "content": "nope 1"}]},
                {"edits": [{"file": "a.txt", "content": "nope 2"}]},
                {"edits": [{"file": "a.txt", "content": "nope 3"}]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, exhausted)
    assert main(["run", "--yes", "--config", str(cfg_path)]) == 2
    capsys.readouterr()

    # A second terminal holds the lock: resume must not touch the run at all.
    with run_lock(Path(tmp_path / "state")):
        rc = main(["resume", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "another run holds the lock" in out
    state = _state(run_dir)
    assert state["escalation"]["kind"] == "task_retries_exhausted"
    assert state["task_state"]["A"]["status"] == "escalated"


# --- error paths ---------------------------------------------------------------


def test_run_without_a_plan_reports_an_error(tmp_path, demo_repo, monkeypatch, capsys):
    _use_mock(monkeypatch, _scenario(tmp_path, SCENARIO))
    cfg_path = _config(tmp_path, demo_repo)

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "error:" in out


def test_run_reports_a_config_error(tmp_path, capsys):
    rc = main(["run", "--config", str(tmp_path / "missing.yaml")])

    out = capsys.readouterr().out
    assert rc == 1
    assert "error:" in out


def test_plan_missing_from_file_reports_an_error(
    tmp_path, demo_repo, monkeypatch, capsys
):
    _use_mock(monkeypatch, _scenario(tmp_path, SCENARIO))
    cfg_path = _config(tmp_path, demo_repo)

    rc = main(
        [
            "plan",
            "a brief",
            "--from",
            "/nonexistent/from.md",
            "--config",
            str(cfg_path),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "/nonexistent/from.md" in out
    assert "Traceback" not in out
