"""Unit tests for the live activity channel (``activity.json``).

``state.json`` only changes when a node *returns*, so a dashboard watching a run
cannot tell that a turn is in flight — which is why an implementing task never
appeared in the board's working column. ``orch.activity`` is that missing signal:
a small map of running turns, keyed by task id, written at dispatch and cleared
on completion. It is advisory only: nothing about a run's correctness depends on
it, so a lost or corrupt entry must never raise.
"""

import threading
from pathlib import Path

import pytest

from orch.activity import (
    activity_begin,
    activity_end,
    activity_path,
    read_activity,
    running,
)
from orch.artifacts import RunDir


def _run_dir(tmp_path: Path) -> RunDir:
    return RunDir.create(tmp_path / "runs", "R1")


def test_begin_records_the_running_turn(tmp_path):
    run_dir = _run_dir(tmp_path)

    activity_begin(
        run_dir,
        "task-3",
        phase="implement",
        role="implement",
        model="opencode/space-bunny-free",
        title="R1/task-3/implement",
    )

    entry = read_activity(run_dir)["task-3"]
    assert entry["phase"] == "implement"
    assert entry["role"] == "implement"
    assert entry["model"] == "opencode/space-bunny-free"
    assert entry["title"] == "R1/task-3/implement"
    assert entry["started_at"], "a start time is what lets the UI show elapsed"


def test_end_clears_only_that_task(tmp_path):
    run_dir = _run_dir(tmp_path)
    activity_begin(run_dir, "A", phase="implement", role="implement", model="m")
    activity_begin(run_dir, "B", phase="review", role="review", model="m")

    activity_end(run_dir, "A")

    assert set(read_activity(run_dir)) == {"B"}


def test_end_on_a_task_that_never_started_is_a_no_op(tmp_path):
    run_dir = _run_dir(tmp_path)

    activity_end(run_dir, "never-started")

    assert read_activity(run_dir) == {}


def test_read_is_empty_when_nothing_has_run(tmp_path):
    assert read_activity(_run_dir(tmp_path)) == {}


def test_read_tolerates_a_corrupt_file(tmp_path):
    """A half-written or hand-mangled file must degrade to 'nothing running'."""
    run_dir = _run_dir(tmp_path)
    activity_path(run_dir).write_text("{not json", encoding="utf-8")

    assert read_activity(run_dir) == {}


def test_concurrent_begins_lose_no_entries(tmp_path):
    """Parallel branches dispatch at once; a read-modify-write must not drop one."""
    run_dir = _run_dir(tmp_path)
    ids = [f"task-{i}" for i in range(8)]

    threads = [
        threading.Thread(
            target=activity_begin,
            args=(run_dir, tid),
            kwargs={"phase": "implement", "role": "implement", "model": "m"},
        )
        for tid in ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert set(read_activity(run_dir)) == set(ids)


def test_running_marks_the_turn_only_for_the_duration_of_the_block(tmp_path):
    run_dir = _run_dir(tmp_path)

    with running(
        run_dir, "A", phase="implement", role="implement", model="m", title="R1/A/implement"
    ):
        assert "A" in read_activity(run_dir), "visible while the block is open"

    assert read_activity(run_dir) == {}, "cleared when the block closes"


def test_running_clears_the_entry_when_the_block_raises(tmp_path):
    """A crashing turn must not leave a phantom spinner behind."""
    run_dir = _run_dir(tmp_path)

    with pytest.raises(RuntimeError):
        with running(run_dir, "A", phase="implement", role="implement", model="m"):
            raise RuntimeError("boom")

    assert read_activity(run_dir) == {}
