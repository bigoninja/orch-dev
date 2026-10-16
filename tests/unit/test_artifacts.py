"""Run artifacts: the run directory, atomic writes, status.md, and the run lock."""

import json
import os

import pytest

from orch.artifacts import (
    RunDir,
    atomic_write,
    render_status_md,
    run_lock,
    snapshot_state,
    write_status,
)
from orch.errors import LockHeldError
from orch.state import TStatus, new_task_state

TASKS = {
    "B1B-01": {"id": "B1B-01", "title": "harden headers"},
    "B1B-02": {"id": "B1B-02", "title": "tighten cookies"},
    "B1B-06": {"id": "B1B-06", "title": "rate limit login"},
}

ESCALATION = {
    "kind": "retries_exhausted",
    "message": "B1B-06 exhausted fix attempts",
    "task_id": "B1B-06",
}

#: The pinned contract for `render_status_md` — a run mid-flight, one task
#: Integrated, one Implementing with a violation, one Escalated with an
#: escalation block. Exact bytes, not a regex.
GOLDEN = (
    "Run 2026-10-16-1 — security phase 1b — ESCALATED\n"
    "Integration: orch/integration @ 1 integrated\n"
    "| Task | State | Attempts | Review | Notes |\n"
    "|---|---|---|---|---|\n"
    "| B1B-01 | integrated | 1 | approve | - |\n"
    "| B1B-02 | implementing | 2/fix:1 | - | b.py: read-only file |\n"
    "| B1B-06 | escalated | 3 | reject | retries exhausted |\n"
    "## Escalation\n"
    "kind: retries_exhausted\n"
    "message: B1B-06 exhausted fix attempts\n"
    "task_id: B1B-06\n"
    "fix the cause, then: orch resume\n"
)


def _task_state() -> dict[str, dict]:
    return {
        "B1B-01": {
            **new_task_state("B1B-01"),
            "status": TStatus.INTEGRATED,
            "attempts": 1,
            "verdicts": [{"verdict": "approve"}],
        },
        "B1B-02": {
            **new_task_state("B1B-02"),
            "status": TStatus.IMPLEMENTING,
            "attempts": 2,
            "fix_attempts": 1,
            "violation": ["b.py: read-only file"],
        },
        "B1B-06": {
            **new_task_state("B1B-06"),
            "status": TStatus.ESCALATED,
            "attempts": 3,
            "verdicts": [{"verdict": "reject", "findings": []}],
            "last_error": "retries exhausted",
        },
    }


def _render(**overrides) -> str:
    kwargs = {
        "run_id": "2026-10-16-1",
        "brief": "security phase 1b",
        "integration_branch": "orch/integration",
        "tasks": TASKS,
        "task_state": _task_state(),
        "escalation": ESCALATION,
    }
    kwargs.update(overrides)
    return render_status_md(**kwargs)


# --- RunDir ---------------------------------------------------------------


def test_run_dir_create_makes_evidence_and_logs(tmp_path):
    run_dir = RunDir.create(tmp_path / "runs", "2026-10-16-1")

    assert run_dir.root == tmp_path / "runs" / "2026-10-16-1"
    assert run_dir.evidence.is_dir()
    assert run_dir.logs.is_dir()
    assert run_dir.evidence == run_dir.root / "evidence"
    assert run_dir.logs == run_dir.root / "logs"


def test_run_dir_artifact_paths(tmp_path):
    run_dir = RunDir.create(tmp_path / "runs", "r1")

    assert run_dir.plan_path == run_dir.root / "plan.yaml"
    assert run_dir.state_path == run_dir.root / "state.json"
    assert run_dir.status_path == run_dir.root / "status.md"
    assert run_dir.checkpoints_path == run_dir.root / "checkpoints.db"


def test_run_dir_create_is_idempotent(tmp_path):
    first = RunDir.create(tmp_path / "runs", "r1")
    (first.logs / "B1B-01.implement.log").write_text("kept", encoding="utf-8")

    second = RunDir.create(tmp_path / "runs", "r1")

    assert second == first
    assert (second.logs / "B1B-01.implement.log").read_text(encoding="utf-8") == "kept"


# --- atomic_write ---------------------------------------------------------


def test_atomic_write_round_trips_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "state.json"

    atomic_write(target, "hello\n")

    assert target.read_text(encoding="utf-8") == "hello\n"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_atomic_write_overwrites_existing_content(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")

    atomic_write(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_atomic_write_creates_missing_parent_directory(tmp_path):
    target = tmp_path / "runs" / "r1" / "state.json"

    atomic_write(target, "{}")

    assert target.read_text(encoding="utf-8") == "{}"


# --- render_status_md -----------------------------------------------------


def test_render_status_md_golden():
    assert _render() == GOLDEN


def test_render_status_md_running_without_escalation():
    rendered = _render(escalation=None)

    assert rendered.splitlines()[0] == "Run 2026-10-16-1 — security phase 1b — RUNNING"
    assert rendered.splitlines()[1] == "Integration: orch/integration @ 1 integrated"
    assert "## Escalation" not in rendered


def test_render_status_md_truncates_the_brief_to_sixty_chars():
    brief = "b" * 70

    header = _render(brief=brief).splitlines()[0]

    assert header == f"Run 2026-10-16-1 — {'b' * 60} — ESCALATED"


def test_render_status_md_counts_only_integrated_tasks():
    task_state = _task_state()
    task_state["B1B-02"]["status"] = TStatus.APPROVED

    rendered = _render(task_state=task_state)

    assert rendered.splitlines()[1] == "Integration: orch/integration @ 1 integrated"


def test_render_status_md_rows_follow_tasks_dict_order():
    rendered = _render()

    rows = [line for line in rendered.splitlines() if line.startswith("| B1B-")]
    assert [row.split(" | ")[0] for row in rows] == ["| B1B-01", "| B1B-02", "| B1B-06"]


def test_render_status_md_attempts_omit_the_fix_suffix_at_zero():
    task_state = _task_state()

    row = next(line for line in _render(task_state=task_state).splitlines() if line.startswith("| B1B-06"))
    assert row.split(" | ")[2] == "3"


def test_render_status_md_review_is_a_dash_without_verdicts():
    task_state = _task_state()

    row = next(line for line in _render(task_state=task_state).splitlines() if line.startswith("| B1B-02"))
    assert row.split(" | ")[3] == "-"


def test_render_status_md_prefers_last_error_over_violation():
    task_state = _task_state()
    task_state["B1B-02"]["last_error"] = "worker blocked"

    row = next(line for line in _render(task_state=task_state).splitlines() if line.startswith("| B1B-02"))
    assert row.split(" | ")[-1] == "worker blocked |"


def test_render_status_md_falls_through_an_empty_last_error_to_the_violation():
    task_state = _task_state()
    task_state["B1B-02"]["last_error"] = ""

    row = next(line for line in _render(task_state=task_state).splitlines() if line.startswith("| B1B-02"))
    assert row.split(" | ")[-1] == "b.py: read-only file |"


def test_render_status_md_notes_flatten_newlines_and_escape_pipes():
    task_state = _task_state()
    task_state["B1B-02"]["violation"] = ["a | b\nc"]

    row = next(line for line in _render(task_state=task_state).splitlines() if line.startswith("| B1B-02"))

    assert row == "| B1B-02 | implementing | 2/fix:1 | - | a \\| b c |"


def test_render_status_md_marks_a_missing_escalation_task_id():
    rendered = _render(escalation={"kind": "deadlock", "message": "nothing ready", "task_id": None})

    assert "task_id: -" in rendered


# --- write_status / snapshot_state ---------------------------------------


def test_write_status_writes_the_rendered_markdown(tmp_path):
    run_dir = RunDir.create(tmp_path / "runs", "r1")

    write_status(
        run_dir,
        run_id="2026-10-16-1",
        brief="security phase 1b",
        integration_branch="orch/integration",
        tasks=TASKS,
        task_state=_task_state(),
        escalation=ESCALATION,
    )

    assert run_dir.status_path.read_text(encoding="utf-8") == GOLDEN


def test_snapshot_state_writes_state_json_and_status_md(tmp_path):
    run_dir = RunDir.create(tmp_path / "runs", "r1")

    snapshot_state(
        run_dir,
        run_id="2026-10-16-1",
        brief="security phase 1b",
        integration_branch="orch/integration",
        tasks=TASKS,
        task_state=_task_state(),
        escalation=ESCALATION,
    )

    payload = json.loads(run_dir.state_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "2026-10-16-1"
    assert payload["brief"] == "security phase 1b"
    assert payload["integration_branch"] == "orch/integration"
    assert payload["escalation"] == ESCALATION
    assert list(payload["tasks"]) == ["B1B-01", "B1B-02", "B1B-06"]
    assert payload["task_state"]["B1B-01"]["status"] == "integrated"
    assert payload["task_state"]["B1B-02"]["status"] == "implementing"
    assert payload["task_state"]["B1B-06"]["status"] == "escalated"
    assert run_dir.status_path.read_text(encoding="utf-8") == GOLDEN


def test_snapshot_state_overwrites_and_leaves_no_temp_files(tmp_path):
    runs = tmp_path / "runs"
    run_dir = RunDir.create(runs, "r1")
    snapshot_state(
        run_dir,
        run_id="r1",
        brief="first",
        integration_branch="orch/integration",
        tasks=TASKS,
        task_state=_task_state(),
        escalation=ESCALATION,
    )

    task_state = _task_state()
    task_state["B1B-01"]["status"] = TStatus.APPROVED
    snapshot_state(
        run_dir,
        run_id="r1",
        brief="second",
        integration_branch="orch/integration",
        tasks=TASKS,
        task_state=task_state,
        escalation=None,
    )

    payload = json.loads(run_dir.state_path.read_text(encoding="utf-8"))
    assert payload["brief"] == "second"
    assert payload["escalation"] is None
    assert payload["task_state"]["B1B-01"]["status"] == "approved"
    assert sorted(p.name for p in run_dir.root.iterdir() if p.is_file()) == [
        "state.json",
        "status.md",
    ]


def test_snapshot_state_serializes_plan_task_models(tmp_path):
    from orch.plan import TaskDef

    run_dir = RunDir.create(tmp_path / "runs", "r1")
    plan_tasks = {
        "B1B-01": TaskDef(
            id="B1B-01",
            title="harden headers",
            description="set the headers",
            files_allowed=["app.py"],
            acceptance_criteria=["headers set"],
            checks={"test": {"cmd": "true"}},
        )
    }

    snapshot_state(
        run_dir,
        run_id="r1",
        brief="b",
        integration_branch="orch/integration",
        tasks=plan_tasks,
        task_state={"B1B-01": new_task_state("B1B-01")},
        escalation=None,
    )

    payload = json.loads(run_dir.state_path.read_text(encoding="utf-8"))
    assert payload["tasks"]["B1B-01"]["id"] == "B1B-01"
    assert payload["task_state"]["B1B-01"]["status"] == "pending"


# --- run_lock -------------------------------------------------------------


def test_run_lock_creates_the_lock_file(tmp_path):
    with run_lock(tmp_path):
        assert (tmp_path / "orch.lock").is_file()


def test_run_lock_second_acquire_raises(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(LockHeldError):
            with run_lock(tmp_path):
                pass


def test_run_lock_is_released_after_exit(tmp_path):
    with run_lock(tmp_path):
        pass

    with run_lock(tmp_path):  # would raise if the first lock leaked
        pass


def test_run_lock_is_released_after_an_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with run_lock(tmp_path):
            raise RuntimeError("boom")

    with run_lock(tmp_path):
        pass


def test_run_lock_creates_the_state_directory(tmp_path):
    state_dir = tmp_path / "state"

    with run_lock(state_dir):
        assert (state_dir / "orch.lock").is_file()

    assert os.path.isdir(state_dir)
