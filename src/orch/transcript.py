"""Persisted agent transcripts: what an agent actually said and did.

The adapter returns the full ``opencode run --format json`` event stream, but the
orchestrator discarded it when a node returned — so a dashboard had nothing to
show for a task, and a human reviewing a run had only the diff and the verdict.
Each turn is now written under ``evidence/`` as two files:

* ``<task>.<phase>.transcript.jsonl`` — the raw events, one JSON object per line;
* ``<task>.<phase>.meta.json`` — the prompt, role, model and session id.

The stream is capped at :data:`MAX_EVENTS`: a chat is worth keeping, an unbounded
stream is not worth a full disk. Truncation is recorded in the meta so a reader
never mistakes a capped transcript for a complete one.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from orch.adapters.base import AgentRunResult
from orch.artifacts import RunDir, atomic_write

#: Cap on stored events per turn. Whole lines only, so the file stays parseable.
MAX_EVENTS = 5000


def transcript_path(run_dir: RunDir, task_id: str, phase: str) -> Path:
    """``evidence/<task>.<phase>.transcript.jsonl`` — the raw event stream."""
    return run_dir.evidence / f"{task_id}.{phase}.transcript.jsonl"


def meta_path(run_dir: RunDir, task_id: str, phase: str) -> Path:
    """``evidence/<task>.<phase>.meta.json`` — prompt, role, model, session."""
    return run_dir.evidence / f"{task_id}.{phase}.meta.json"


def _now() -> str:
    """Microsecond precision, so turns recorded in the same second still order."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def render_messages(raw_json: list[dict] | None) -> list[dict]:
    """The event stream as conversation messages: assistant text and tool calls.

    Tolerant by design: the real CLI nests text under ``part``
    (``{"type":"text","part":{"type":"text","text":…}}``) while the mock adapter
    emits it at the top level, and a stream carries many event kinds a chat does
    not show. Anything unrecognised is skipped rather than guessed at.
    """
    messages: list[dict] = []
    for event in raw_json or []:
        if not isinstance(event, dict):
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        kind = event.get("type")
        if kind == "text":
            text = part.get("text") if part else event.get("text")
            if isinstance(text, str) and text:
                messages.append({"kind": "text", "text": text})
        elif kind == "tool":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            message: dict = {
                "kind": "tool",
                "name": str(part.get("name") or part.get("tool") or "tool"),
                "status": str(state.get("status") or ""),
            }
            inp = state.get("input")
            if isinstance(inp, dict) and isinstance(inp.get("command"), str):
                message["command"] = inp["command"]
            messages.append(message)
    return messages


def write_transcript(
    run_dir: RunDir,
    task_id: str,
    *,
    phase: str,
    role: str,
    model: str,
    prompt: str,
    result: AgentRunResult,
) -> None:
    """Persist one turn: the raw stream (capped) plus its prompt and metadata."""
    events = list(result.raw_json or [])
    truncated = len(events) > MAX_EVENTS
    body = "\n".join(json.dumps(event) for event in events[:MAX_EVENTS])
    atomic_write(transcript_path(run_dir, task_id, phase), body + "\n" if body else "")
    atomic_write(
        meta_path(run_dir, task_id, phase),
        json.dumps(
            {
                "task_id": task_id,
                "phase": phase,
                "role": role,
                "model": model,
                "session_id": result.session_id,
                "prompt": prompt,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "truncated": truncated,
                "recorded_at": _now(),
            },
            indent=2,
        )
        + "\n",
    )


def _read_meta(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_events(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def read_turns(run_dir: RunDir, task_id: str) -> list[dict]:
    """Every recorded turn for a task, oldest first, as a rendered conversation.

    The ``.`` after the task id keeps a repair (``task-3-repair``) out of its
    parent's turns — the two are different tasks that merely share a prefix.
    """
    turns: list[dict] = []
    for path in sorted(run_dir.evidence.glob(f"{task_id}.*.meta.json")):
        meta = _read_meta(path)
        if meta is None:
            continue
        phase = str(meta.get("phase") or "")
        if not phase:
            continue
        turns.append(
            {
                "phase": phase,
                "role": str(meta.get("role") or ""),
                "model": str(meta.get("model") or ""),
                "session_id": meta.get("session_id"),
                "prompt": str(meta.get("prompt") or ""),
                "truncated": bool(meta.get("truncated")),
                "recorded_at": str(meta.get("recorded_at") or ""),
                "messages": render_messages(
                    _read_events(transcript_path(run_dir, task_id, phase))
                ),
            }
        )
    turns.sort(key=lambda turn: turn["recorded_at"])
    return turns
