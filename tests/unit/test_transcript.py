"""Persisted agent transcripts — the raw material for the chat pane.

The adapter already returns the full ``--format json`` event stream and the
session id; until now nothing kept them, so there was no chat to open. These
tests pin what a turn's record must contain, how the stream renders into a
conversation, and that a pathological stream cannot fill the disk.
"""

import json
from pathlib import Path

from orch.adapters.base import AgentRunResult
from orch.artifacts import RunDir
from orch.transcript import (
    MAX_EVENTS,
    meta_path,
    read_turns,
    render_messages,
    transcript_path,
    write_transcript,
)


def _run_dir(tmp_path: Path) -> RunDir:
    return RunDir.create(tmp_path / "runs", "R1")


def _result(raw, *, text="hello", session="ses_1") -> AgentRunResult:
    return AgentRunResult(
        session_id=session,
        exit_code=0,
        stdout=text,
        stderr="",
        timed_out=False,
        text=text,
        raw_json=raw,
    )


def test_write_transcript_persists_the_stream_and_the_prompt(tmp_path):
    run_dir = _run_dir(tmp_path)
    raw = [{"type": "text", "part": {"type": "text", "text": "done"}}]

    write_transcript(
        run_dir,
        "A",
        phase="implement",
        role="implement",
        model="vendor/m",
        prompt="do the thing",
        result=_result(raw),
    )

    turns = read_turns(run_dir, "A")
    assert len(turns) == 1
    assert turns[0]["phase"] == "implement"
    assert turns[0]["role"] == "implement"
    assert turns[0]["model"] == "vendor/m"
    assert turns[0]["prompt"] == "do the thing"
    assert turns[0]["session_id"] == "ses_1"
    assert turns[0]["messages"] == [{"kind": "text", "text": "done"}]


def test_read_turns_is_empty_when_nothing_was_recorded(tmp_path):
    assert read_turns(_run_dir(tmp_path), "A") == []


def test_turns_come_back_in_recorded_order(tmp_path):
    run_dir = _run_dir(tmp_path)
    write_transcript(
        run_dir, "A", phase="implement", role="implement", model="m",
        prompt="p1", result=_result([]),
    )
    write_transcript(
        run_dir, "A", phase="fix1", role="fix", model="m", prompt="p2", result=_result([])
    )

    assert [t["phase"] for t in read_turns(run_dir, "A")] == ["implement", "fix1"]


def test_render_messages_reads_both_event_shapes():
    """The real CLI nests text under ``part``; the mock emits it at top level."""
    real = [{"type": "text", "part": {"type": "text", "text": "real"}}]
    mock = [{"type": "text", "text": "mock"}]

    assert render_messages(real) == [{"kind": "text", "text": "real"}]
    assert render_messages(mock) == [{"kind": "text", "text": "mock"}]


def test_render_messages_extracts_a_tool_call():
    raw = [
        {
            "type": "tool",
            "part": {
                "type": "tool",
                "name": "shell",
                "state": {"status": "completed", "input": {"command": "npm test"}},
            },
        }
    ]

    assert render_messages(raw) == [
        {"kind": "tool", "name": "shell", "status": "completed", "command": "npm test"}
    ]


def test_render_messages_ignores_anything_it_does_not_understand():
    raw = [{"type": "step_start"}, {"type": "text"}, {"type": "text", "part": {}}, 42]

    assert render_messages(raw) == []


def test_a_huge_stream_is_capped_and_flagged(tmp_path):
    """A chat is worth keeping; an unbounded stream is not worth a full disk."""
    run_dir = _run_dir(tmp_path)
    raw = [{"type": "text", "part": {"type": "text", "text": "x"}} for _ in range(MAX_EVENTS + 10)]

    write_transcript(
        run_dir, "A", phase="implement", role="implement", model="m",
        prompt="p", result=_result(raw),
    )

    meta = json.loads(meta_path(run_dir, "A", "implement").read_text())
    assert meta["truncated"] is True
    lines = transcript_path(run_dir, "A", "implement").read_text().splitlines()
    assert len(lines) == MAX_EVENTS
