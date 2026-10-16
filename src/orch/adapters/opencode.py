"""The real adapter: one headless OpenCode run per agent role.

This is the only place in orch where a live model is invoked. Each call spawns
``opencode run --format json --auto --model <pinned> --title <title> [--session
<sid>] <prompt>`` in the task worktree and decodes its JSON-lines output.

Verified against a real probe on 2026-10-16: every emitted line has the shape
``{"type": ..., "timestamp": ..., "sessionID": "ses_…", "part": {...}}``, assistant
text arrives as ``{"type": "text", "part": {"type": "text", "text": "…"}}``, and
the process exits 0 on success.

``PWD`` is exported as the run's ``cwd``: ``opencode run`` resolves its project
(where agent tools read and write) from ``$PWD``, not the process working
directory, so a stale inherited ``PWD`` would leak a worktree run into the
orchestrator's launch directory.

Failure policy (mirrors the no-substitution law): a provider/auth signature on
stdout+stderr raises :class:`ModelUnavailableError` — orch never retries or
swaps the model. Bare numeric provider codes (``401``/``402``) are matched on
**stderr only**, because an agent's own transcript (stdout) legitimately contains
numbers like "HTTP 401 handling" and matching those would fail a healthy run. A
wall-clock timeout kills the whole process group and returns a ``timed_out=True``
result; the adapter deliberately does **not** raise
:class:`AdapterTimeout` itself because escalating is the caller's decision. A
run that produces no session id when one is required also fails closed.
"""

import json
import os
import re
import signal
import subprocess
from pathlib import Path

from orch.adapters.base import AgentRunResult
from orch.errors import ModelUnavailableError, OrchError

#: Word signatures that mean the pinned model is unusable; checked over
#: stdout+stderr because a provider/auth failure may be printed on either stream.
MODEL_UNAVAILABLE_RE = re.compile(
    r"(?i)no such model|model .*not found|unauthorized|invalid api key"
)

#: Bare numeric provider codes, matched on **stderr only**. An agent's own
#: transcript (stdout) can contain a number like 401 in prose ("HTTP 401
#: handling"); treating that as a provider failure would fail a healthy run.
#: This stderr/stdout split is pinned here on purpose.
MODEL_UNAVAILABLE_CODE_RE = re.compile(r"\b401\b|\b402\b")


def parse_jsonl(output: str) -> tuple[list[dict], str | None, str]:
    """Decode OpenCode's ``--format json`` stream.

    Returns ``(lines, session_id, text)`` where ``lines`` are the successfully
    decoded event dicts in order, ``session_id`` is the first ``sessionID`` seen
    (``None`` if the stream carries none), and ``text`` is the verbatim
    concatenation of ``part.text`` across ``type == "text"`` events.

    Lines that are not JSON are skipped rather than failing the run: the CLI
    interleaves non-JSON noise (warnings, progress) with the event stream.
    (``opencode session list`` is intentionally not parsed in v1; a missing id
    is handled fail-closed by :meth:`OpenCodeAdapter.run`.)
    """
    lines: list[dict] = []
    session_id: str | None = None
    texts: list[str] = []
    for raw in output.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        lines.append(event)
        if session_id is None and event.get("sessionID"):
            session_id = str(event["sessionID"])
        part = event.get("part")
        if event.get("type") == "text" and isinstance(part, dict):
            chunk = part.get("text")
            if isinstance(chunk, str):
                texts.append(chunk)
    return lines, session_id, "".join(texts)


class OpenCodeAdapter:
    """Drives ``opencode run`` as a subprocess; satisfies :class:`AgentAdapter`."""

    def __init__(self, model: str, timeout_s: int, extra_env: dict[str, str]):
        self.model = model
        self.timeout_s = timeout_s
        self.extra_env = dict(extra_env)

    def run(
        self,
        *,
        role: str,
        prompt: str,
        cwd: Path,
        session_id: str | None,
        title: str,
        env: dict[str, str],
    ) -> AgentRunResult:
        """Run one agent turn and return its :class:`AgentRunResult`.

        ``role`` is unused for dispatch (the prompt carries the role's contract)
        but is part of the adapter protocol. The prompt is passed as the final
        positional argument so the model sees it verbatim.
        """
        cmd = [
            "opencode", "run", "--format", "json", "--auto",
            "--model", self.model, "--title", title,
        ]
        if session_id:
            cmd += ["--session", session_id]
        cmd.append(prompt)

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            # PWD must track the child's real cwd: ``opencode run`` resolves the
            # project it reads/writes from ``$PWD``, not ``getcwd()``, and
            # ``Popen(cwd=...)`` does not update the inherited ``PWD``. Without
            # this, a run launched for a task worktree would resolve into the
            # orchestrator's launch directory instead of the worktree — an
            # isolation break. Precedence is pinned low-to-high as
            # ``os.environ`` < derived PWD < ``extra_env`` < caller ``env`` <
            # ``NO_COLOR``, so a caller that deliberately supplies ``PWD``
            # (e.g. tests) still wins.
            env={
                **os.environ,
                "PWD": str(cwd),
                **self.extra_env,
                **env,
                "NO_COLOR": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=self.timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            # Kill the whole group: opencode may have spawned tool children.
            # The child may have exited in the race window, hence the guard.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            timed_out = True

        lines, captured_session, text = parse_jsonl(out)

        if MODEL_UNAVAILABLE_RE.search(err + out) or MODEL_UNAVAILABLE_CODE_RE.search(
            err
        ):
            raise ModelUnavailableError(
                f"pinned model {self.model!r} is unavailable: "
                f"{(err or out).strip()[:500]}"
            )

        if captured_session is None and not timed_out:
            raise OrchError(
                f"opencode run produced no session id (title={title!r}, "
                f"exit={proc.returncode}); cannot resume this session:\n"
                f"{(err or out).strip()[:500]}"
            )

        return AgentRunResult(
            session_id=captured_session,
            exit_code=proc.returncode,
            stdout=out,
            stderr=err,
            timed_out=timed_out,
            text=text,
            raw_json=lines,
        )
