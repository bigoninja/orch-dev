"""Unit tests for :mod:`orch.dashboard` — the read-only progress page.

Two contracts:

* :func:`build_dashboard_state` renders ``state.json`` into the page payload:
  per-task rows in plan order, the integrated count, and a run state derived
  from the escalation/summary/task statuses (it never writes anything).
* The embedded HTTP server serves that payload and the page, and 404s anything
  else — safe to leave open beside a live run.
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from orch.artifacts import RunDir
from orch.activity import activity_begin
from orch.adapters.base import AgentRunResult
from orch.config import Config
from orch.transcript import write_transcript

from orch.dashboard import (
    _PAGE,
    _handler_for,
    build_chat_payload,
    build_dashboard_state,
)


def _run_dir_with_state(tmp_path: Path, payload: dict) -> RunDir:
    run_dir = RunDir.create(tmp_path / "runs", "R1")
    run_dir.state_path.write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def _payload(**over) -> dict:
    base = {
        "run_id": "R1",
        "brief": "the brief",
        "integration_branch": "orch/integration",
        "tasks": {
            "A": {"title": "Task A", "risky": True},
            "B": {"title": "Task B", "risky": False},
        },
        "task_state": {
            "A": {
                "status": "integrated",
                "attempts": 1,
                "fix_attempts": 1,
                "verdicts": [{"verdict": "reject"}, {"verdict": "approve"}],
            },
            "B": {
                "status": "implementing",
                "attempts": 1,
                "fix_attempts": 0,
                "branch": "agent/B",
                "commits": ["abc1234"],
            },
        },
        "escalation": None,
        "summary": None,
    }
    base.update(over)
    return base


def test_missing_state_json_is_a_clean_error(tmp_path: Path):
    run_dir = RunDir.create(tmp_path / "runs", "R1")
    out = build_dashboard_state(run_dir)
    assert "error" in out


def test_rows_in_plan_order_with_review_and_counts(tmp_path: Path):
    run_dir = _run_dir_with_state(tmp_path, _payload())
    out = build_dashboard_state(run_dir)

    assert out["run_state"] == "RUNNING"
    assert (out["integrated"], out["total"]) == (1, 2)
    assert [r["id"] for r in out["tasks"]] == ["A", "B"]
    a, b = out["tasks"]
    assert a["status"] == "integrated" and a["risky"] is True
    assert a["fix_attempts"] == 1 and a["review"] == "approve"  # LAST verdict wins
    assert b["status"] == "implementing" and b["review"] == ""
    assert a["branch"] == "agent/A" and a["commits"] == 0  # defaults when absent
    assert b["branch"] == "agent/B" and b["commits"] == 1


def test_notes_prefer_last_error_then_violation(tmp_path: Path):
    payload = _payload(
        task_state={
            "A": {"status": "escalated", "last_error": "BLOCKED: cannot"},
            "B": {"status": "implementing", "violation": ["stray.txt"]},
        }
    )
    run_dir = _run_dir_with_state(tmp_path, payload)
    out = build_dashboard_state(run_dir)
    a, b = out["tasks"]
    assert a["notes"] == "BLOCKED: cannot"
    assert "stray.txt" in b["notes"]


def test_escalation_wins_over_everything(tmp_path: Path):
    payload = _payload(escalation={"kind": "model_unavailable", "task_id": "A", "message": "x"})
    run_dir = _run_dir_with_state(tmp_path, payload)
    out = build_dashboard_state(run_dir)
    assert out["run_state"] == "ESCALATED"
    assert out["escalation"]["kind"] == "model_unavailable"


def test_all_integrated_reads_complete_even_without_summary(tmp_path: Path):
    """The pre-summary fix-up heuristic: a live run's final snapshot may land
    before the summary key existed; all-integrated is terminal regardless."""
    payload = _payload(
        task_state={
            "A": {"status": "integrated", "attempts": 1, "fix_attempts": 0},
            "B": {"status": "integrated", "attempts": 1, "fix_attempts": 0},
        }
    )
    run_dir = _run_dir_with_state(tmp_path, payload)
    out = build_dashboard_state(run_dir)
    assert out["run_state"] == "COMPLETE"


def test_summary_in_state_json_marks_complete(tmp_path: Path):
    payload = _payload(summary="run R1: 1/2 integrated")
    run_dir = _run_dir_with_state(tmp_path, payload)
    out = build_dashboard_state(run_dir)
    assert out["run_state"] == "COMPLETE"


def test_http_serves_page_payload_and_404(tmp_path: Path):
    run_dir = _run_dir_with_state(tmp_path, _payload())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(run_dir))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        with urllib.request.urlopen(base + "/") as resp:
            page = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "<title>orch dashboard</title>" in page and "/api/state" in page
            assert 'id="board"' in page
            assert "Ready to merge" in page and "Needs you" in page
            assert 'id="view-status"' in page and 'id="view-cards"' in page
            assert 'class="nav"' in page and 'id="brief"' not in page
            assert 'class="logo"' in page and "F69800" in page

        with urllib.request.urlopen(base + "/api/state") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["run_state"] == "RUNNING"
            assert data["tasks"][0]["id"] == "A"

        try:
            urllib.request.urlopen(base + "/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_snapshot_run_writes_summary_so_finished_runs_are_detectable(tmp_path: Path):
    """The plumbing fix this feature surfaced: `report`'s summary must reach
    state.json, which the CLI's latest-run pick and this dashboard both read."""
    from orch.nodes import snapshot_run

    run_dir = RunDir.create(tmp_path / "runs", "R1")
    cfg = Config(target_repo=tmp_path)
    snapshot_run(
        run_dir,
        cfg,
        {"run_id": "R1", "tasks": {}, "task_state": {}, "summary": "run R1: 0/0 integrated"},
    )
    written = json.loads(run_dir.state_path.read_text(encoding="utf-8"))
    assert written["summary"] == "run R1: 0/0 integrated"


# --- page structure: the readability/a11y polish (asserted on the embedded page) ---


def test_page_marks_escalations_as_a_live_region():
    # A new escalation must be announced, not just painted.
    assert 'role="alert"' in _PAGE


def test_page_gives_review_and_approved_their_own_pill_colors():
    # Review and approved no longer collapse into the shared "working" blue.
    assert ".pill.review" in _PAGE and ".pill.approved" in _PAGE
    assert "#e3b341" in _PAGE  # review amber
    assert "#d2a8ff" in _PAGE  # approved purple


def test_page_faint_tier_passes_contrast():
    # #484f58 on #0d1117 is ~2.6:1 (fails AA); the faint tier is lightened.
    assert "#7d8590" in _PAGE
    assert "#484f58" not in _PAGE


def test_page_has_sticky_header_progress_bar_and_scrollable_table():
    assert "position: sticky" in _PAGE
    assert 'class="progress"' in _PAGE
    assert 'class="tablewrap"' in _PAGE


def test_page_pauses_when_hidden_and_reconciles_rows_by_id():
    assert "document.hidden" in _PAGE
    assert "visibilitychange" in _PAGE
    assert "dataset.id" in _PAGE


def test_page_honors_reduced_motion_and_marks_the_active_tab():
    assert "prefers-reduced-motion" in _PAGE
    assert "aria-current" in _PAGE


def test_page_keeps_its_contract_markers_and_the_resume_hint():
    assert "<title>orch dashboard</title>" in _PAGE
    assert 'id="brief"' not in _PAGE
    assert "orch resume" in _PAGE


# --- live activity: an in-flight turn must be visible ----------------------------


def test_a_live_turn_shows_the_task_as_implementing(tmp_path: Path):
    """``state.json`` only changes when a node returns, so the live channel is
    what makes an in-flight implement turn visible on the board at all."""
    run_dir = _run_dir_with_state(
        tmp_path,
        _payload(
            task_state={
                "A": {"status": "pending", "attempts": 0, "fix_attempts": 0},
                "B": {"status": "pending", "attempts": 0, "fix_attempts": 0},
            }
        ),
    )
    activity_begin(
        run_dir,
        "B",
        phase="implement",
        role="implement",
        model="opencode/space-bunny-free",
        title="R1/B/implement",
    )

    out = build_dashboard_state(run_dir)

    row = next(r for r in out["tasks"] if r["id"] == "B")
    assert row["status"] == "implementing"
    assert row["live"]["model"] == "opencode/space-bunny-free"


def test_a_live_fix_turn_shows_as_fixing(tmp_path: Path):
    run_dir = _run_dir_with_state(
        tmp_path,
        _payload(
            task_state={
                "A": {"status": "pending", "attempts": 1, "fix_attempts": 0},
                "B": {"status": "pending", "attempts": 0, "fix_attempts": 0},
            }
        ),
    )
    activity_begin(run_dir, "A", phase="fix", role="fix", model="m", title="R1/A/fix1")

    row = next(r for r in build_dashboard_state(run_dir)["tasks"] if r["id"] == "A")

    assert row["status"] == "fixing"


def test_a_stale_live_entry_never_resurrects_a_finished_task(tmp_path: Path):
    """A killed run leaves entries behind; a terminal task must stay terminal."""
    run_dir = _run_dir_with_state(tmp_path, _payload())  # A is integrated
    activity_begin(run_dir, "A", phase="implement", role="implement", model="m")

    row = next(r for r in build_dashboard_state(run_dir)["tasks"] if r["id"] == "A")

    assert row["status"] == "integrated"
    assert build_dashboard_state(run_dir)["integrated"] == 1


def test_no_live_turn_leaves_the_committed_status_alone(tmp_path: Path):
    run_dir = _run_dir_with_state(tmp_path, _payload())

    row = next(r for r in build_dashboard_state(run_dir)["tasks"] if r["id"] == "B")

    assert row["status"] == "implementing"  # the committed status, untouched
    assert row["live"] is None


def test_a_repair_task_is_marked_as_such(tmp_path: Path):
    """The board routes repairs to their own lane, so origin must reach the page."""
    run_dir = _run_dir_with_state(
        tmp_path,
        _payload(
            task_state={
                "A": {"status": "pending", "origin": "repair", "parent": "X"},
                "B": {"status": "pending", "origin": "plan"},
            }
        ),
    )

    rows = {r["id"]: r for r in build_dashboard_state(run_dir)["tasks"]}

    assert rows["A"]["is_repair"] is True
    assert rows["B"]["is_repair"] is False


def test_page_has_implementing_repairing_and_fixing_lanes():
    for lane in ("Implementing", "Repairing", "Fixing"):
        assert lane in _PAGE, f"the board is missing its {lane} lane"


def test_page_has_a_spinner_and_marks_live_cards():
    assert "spinner" in _PAGE
    assert "is-live" in _PAGE


# --- the chat pane: a task's transcript, served read-only ------------------------


def _record_a_turn(run_dir: RunDir, task_id: str = "A") -> None:
    write_transcript(
        run_dir,
        task_id,
        phase="implement",
        role="implement",
        model="opencode/space-bunny-free",
        prompt="do the thing",
        result=AgentRunResult(
            session_id="ses_1",
            exit_code=0,
            stdout="ok",
            stderr="",
            timed_out=False,
            text="ok",
            raw_json=[{"type": "text", "part": {"type": "text", "text": "ok"}}],
        ),
    )


def test_chat_payload_returns_a_tasks_recorded_turns(tmp_path: Path):
    run_dir = _run_dir_with_state(tmp_path, _payload())
    _record_a_turn(run_dir)

    data = build_chat_payload(run_dir, "A")

    assert data["task_id"] == "A"
    assert data["turns"][0]["prompt"] == "do the thing"
    assert data["turns"][0]["messages"] == [{"kind": "text", "text": "ok"}]


def test_chat_payload_is_empty_for_a_task_that_has_not_run(tmp_path: Path):
    run_dir = _run_dir_with_state(tmp_path, _payload())

    assert build_chat_payload(run_dir, "A")["turns"] == []


def test_chat_payload_carries_the_worktree_so_the_session_can_be_opened(tmp_path: Path):
    """``opencode --session <id>`` only resolves in the project the turn ran in."""
    run_dir = _run_dir_with_state(
        tmp_path,
        _payload(task_state={"A": {"status": "integrated", "worktree": "/tmp/wt/A"}}),
    )

    assert build_chat_payload(run_dir, "A")["worktree"] == "/tmp/wt/A"


def test_chat_endpoint_serves_the_turns(tmp_path: Path):
    run_dir = _run_dir_with_state(tmp_path, _payload())
    _record_a_turn(run_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(run_dir))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urllib.request.urlopen(base + "/api/chat/A") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
        assert data["turns"][0]["prompt"] == "do the thing"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_chat_endpoint_refuses_a_task_id_that_escapes_the_run(tmp_path: Path):
    """The task id becomes a glob under ``evidence/`` — it must not walk out."""
    run_dir = _run_dir_with_state(tmp_path, _payload())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.meta.json").write_text(
        json.dumps({"phase": "leak", "prompt": "SECRET"}), encoding="utf-8"
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(run_dir))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        for attempt in ("..%2f..%2foutside%2fsecret", "..", "../outside/secret"):
            try:
                urllib.request.urlopen(base + "/api/chat/" + attempt)
                raise AssertionError(f"expected a refusal for {attempt!r}")
            except urllib.error.HTTPError as e:
                assert e.code == 400, f"{attempt!r} returned {e.code}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_page_has_the_chat_pane_and_its_controls():
    assert 'id="chat"' in _PAGE
    assert "maximize" in _PAGE.lower()
    assert "Escape" in _PAGE


# --- layout: cards left, chat right, and a board that fits ------------------------


def test_page_puts_the_board_before_the_chat_so_the_chat_sits_on_the_right():
    assert _PAGE.index('id="board-wrap"') < _PAGE.index('id="chat"')


def test_page_fixes_the_chat_width_and_toggles_it_with_an_arrow():
    assert "flex: 0 0 460px" in _PAGE
    assert 'id="chat-max"' in _PAGE
    # The arrow reflects the state; a glyph alone is not an accessible name, so the
    # button also carries a label.
    assert "«" in _PAGE and "»" in _PAGE
    assert 'aria-label' in _PAGE


def test_page_fits_the_board_with_a_wrapping_grid():
    """Seven fixed 272px columns overflowed every screen; the grid wraps instead."""
    assert "grid-template-columns: repeat(auto-fit, minmax(180px, 1fr))" in _PAGE
    assert "min-width: 250px" not in _PAGE


def test_page_caps_the_board_height_and_scrolls_inside_each_lane():
    assert "max-height: calc(100vh" in _PAGE
    assert "overflow-y: auto" in _PAGE

