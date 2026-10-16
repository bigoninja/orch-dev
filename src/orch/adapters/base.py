"""Adapter boundary: the protocol every LLM role runs behind.

The real adapter (Task 6) shells out to ``opencode run --format json``; the mock
adapter (``orch.adapters.mock``) replays a YAML scenario for token-free tests.
Both satisfy :class:`AgentAdapter`, which is ``runtime_checkable`` so tests can
assert conformance with ``isinstance``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentRunResult:
    """The outcome of one agent invocation, adapter-agnostic.

    ``text`` is the concatenated assistant text (the payload callers parse);
    ``raw_json`` is the ordered list of decoded ``--format json`` events.
    """

    session_id: str | None
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    text: str
    raw_json: list[dict]


@runtime_checkable
class AgentAdapter(Protocol):
    """Runs one agent role in ``cwd`` and returns its result.

    ``role`` is the orchestrator role (``planner``/``implement``/``review``/…);
    ``title`` is the OpenCode session title (``<run-id>/<TASK-ID>/<phase>``);
    ``session_id`` resumes an existing session when set (fix loops); ``env`` is
    the per-run environment exported to the subprocess.
    """

    def run(
        self,
        *,
        role: str,
        prompt: str,
        cwd: Path,
        session_id: str | None,
        title: str,
        env: dict[str, str],
    ) -> AgentRunResult: ...
