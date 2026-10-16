"""Scenario-driven mock adapter for token-free tests.

A scenario is a YAML file keyed by task id; each value holds the scripted
attempts for the ``implement`` and ``review`` phases::

    T1:
      implement:
        - {edits: [{file: app.txt, content: "OK one"}]}          # uncommitted
        - {edits: [{file: app.txt, content: "OK two"}], commit: true}
        - {blocked: "cannot do it"}
      review:
        - {verdict: reject, findings: ["line 2 wrong"]}
        - {verdict: approve}

An attempt may also ``delete`` paths, which models a fix that cleans up a file it
should never have touched::

    - {edits: [{file: app.txt, content: OK}], delete: [stray.txt]}

Two failure/observation hooks make the adapter a faithful stand-in for the real
one:

* ``raise_model_error: true`` — the pinned model is unusable on this attempt; the
  adapter raises :class:`~orch.errors.ModelUnavailableError` (never substitutes).
  This applies to any ``implement``, ``fix`` or ``review`` entry, so a test can
  drive a model failure out of the reviewer too.
* ``timed_out: true`` — the attempt hits the adapter's wall-clock timeout; the
  result carries ``timed_out=True`` with no edits and no verdict, so the caller
  escalates rather than reading it as a pass (any entry, any phase).
* ``text: "..."`` — overrides the reply text and ``stdout`` (prose only).
* ``tool_commands: ["...", ...]`` — shell commands the attempt "ran": each becomes
  a completed ``shell`` tool part in ``raw_json``, which is what the destructive
  scan reads (prose and tool outputs are deliberately not scanned).

The task id is read from the session ``title`` in ``<run-id>/<TASK-ID>/<phase>``
form. Attempt counters advance per ``(task_id, phase)``; ``fix`` attempts share
the implementer's ``implement`` bucket because a fix is just another attempt on the
same task. Failures are loud: a missing scenario entry or an exhausted
attempt list raises rather than fabricating a result.

The ``planner`` role has its **own** top-level scenario bucket (a list of
``{text: "..."}`` entries), not a per-task one, because a plan is produced once
per run before any task exists::

    planner:
      - {text: "```yaml\\nrun_id: r\\n...\\n```"}

Its title is ``<run-id>/planner``; the reply text is returned verbatim for the
planner node to parse.
"""

import json
from pathlib import Path

import yaml

from orch.adapters.base import AgentRunResult
from orch.errors import ModelUnavailableError, OrchError
from orch.gitops import commit_all


class MockScenarioError(OrchError):
    """The scenario has no entry for the requested task or phase."""


class MockScenarioExhausted(OrchError):
    """The scenario's attempt list for a ``(task, phase)`` is used up."""


def _parse_title(title: str) -> tuple[str | None, str | None]:
    """Split ``<run-id>/<TASK-ID>/<phase>`` into ``(task_id, phase)``.

    Returns ``(None, None)`` when the title carries no ``/``-separated task id,
    so callers can fall back to the scenario's sole task.
    """
    parts = title.split("/")
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, None


class MockAdapter:
    """Replays a scripted scenario; deterministic and token-free."""

    def __init__(self, scenario_path: Path):
        self.scenario: dict = yaml.safe_load(Path(scenario_path).read_text()) or {}
        # Attempt counters, keyed by (task_id, phase-bucket).
        self._phase_calls: dict[tuple[str, str], int] = {}
        # Total calls per task id, for mock session numbering.
        self._task_calls: dict[str, int] = {}

    # -- resolution -----------------------------------------------------------

    def _task_id(self, title: str) -> str:
        task_id, _ = _parse_title(title)
        if task_id is None:
            if len(self.scenario) == 1:
                return next(iter(self.scenario))
            raise MockScenarioError(
                f"cannot derive task id from title {title!r}; expected "
                f"'<run-id>/<TASK-ID>/<phase>'; scenario has {sorted(self.scenario)}"
            )
        if task_id not in self.scenario:
            raise MockScenarioError(
                f"no scenario entry for task {task_id!r} (have "
                f"{sorted(self.scenario)})"
            )
        return task_id

    @staticmethod
    def _bucket(role: str, phase: str | None) -> str:
        """Map a role/phase pair onto its scenario attempt list.

        Either signal naming a review is enough; a fix phase (``fix1``, ``fix2``, …)
        is another attempt on the same task, so it shares the implementer's list.
        The planner has its own top-level bucket: it is not a task attempt.
        """
        if role == "review" or phase == "review":
            return "review"
        if role == "planner" or phase == "planner":
            return "planner"
        if phase is not None and phase.startswith("fix"):
            return "implement"
        return "implement"

    def _next_entry(self, task_id: str, bucket: str) -> dict:
        entries = self.scenario.get(task_id, {}).get(bucket)
        if entries is None:
            raise MockScenarioError(
                f"scenario for task {task_id!r} has no {bucket!r} phase"
            )
        key = (task_id, bucket)
        n = self._phase_calls.get(key, 0)
        if n >= len(entries):
            raise MockScenarioExhausted(
                f"scenario for {task_id}/{bucket} exhausted after {n} call(s)"
            )
        self._phase_calls[key] = n + 1
        return entries[n]

    def _next_planner_entry(self) -> dict:
        """Consume the next ``planner`` entry from the scenario's top-level bucket."""
        entries = self.scenario.get("planner")
        if entries is None:
            raise MockScenarioError("scenario has no 'planner' bucket")
        key = ("planner", "planner")
        n = self._phase_calls.get(key, 0)
        if n >= len(entries):
            raise MockScenarioExhausted(
                f"scenario for planner exhausted after {n} call(s)"
            )
        self._phase_calls[key] = n + 1
        return entries[n]

    # -- adapter protocol -----------------------------------------------------

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
        _, phase = _parse_title(title)
        bucket = self._bucket(role, phase)

        if bucket == "planner":
            session = session_id or "ses_mock_planner"
            text = self._next_planner_entry()["text"]
            return AgentRunResult(
                session_id=session,
                exit_code=0,
                stdout=text,
                stderr="",
                timed_out=False,
                text=text,
                raw_json=[{"type": "text", "text": text}],
            )

        task_id = self._task_id(title)
        self._task_calls[task_id] = self._task_calls.get(task_id, 0) + 1
        session = session_id or f"ses_mock_{task_id}_{self._task_calls[task_id]}"
        entry = self._next_entry(task_id, bucket)

        if entry.get("raise_model_error"):
            # The pinned model is unusable on this attempt: fail loudly, never
            # substitute (the same law the real adapter enforces).
            raise ModelUnavailableError(
                f"mock: pinned model unavailable for {task_id}/{bucket}"
            )

        if entry.get("timed_out"):
            # The adapter's wall-clock timeout: no edits, no verdict — the caller
            # decides to escalate (``timed_out=True`` is never a pass).
            return AgentRunResult(
                session_id=session,
                exit_code=-1,
                stdout="",
                stderr="",
                timed_out=True,
                text="",
                raw_json=[],
            )

        if bucket == "review":
            return self._review_result(entry, session)
        return self._implement_result(entry, task_id, title, Path(cwd), session)

    # -- phase builders -------------------------------------------------------

    @staticmethod
    def _review_result(entry: dict, session: str) -> AgentRunResult:
        """Return ``text`` already shaped for the reviewer verdict contract (§8)."""
        verdict = {
            "verdict": entry["verdict"],
            "risky": bool(entry.get("risky", False)),
            # Scenario findings are plain strings; wrap each in the strict shape.
            "findings": [
                {"severity": "major", "file": "", "note": str(f)}
                for f in entry.get("findings") or []
            ],
            "summary": "mock review",
        }
        text = json.dumps(verdict)
        return AgentRunResult(
            session_id=session,
            exit_code=0,
            stdout=text,
            stderr="",
            timed_out=False,
            text=text,
            raw_json=[{"type": "text", "text": text}],
        )

    @staticmethod
    def _implement_result(
        entry: dict, task_id: str, title: str, cwd: Path, session: str
    ) -> AgentRunResult:
        blocked = entry.get("blocked")
        if blocked is not None:
            text = f"BLOCKED: {blocked}"
            return AgentRunResult(
                session_id=session,
                exit_code=0,
                stdout=text,
                stderr="",
                timed_out=False,
                text=text,
                raw_json=[{"type": "text", "text": text}],
            )

        edits = entry.get("edits", [])
        for e in edits:
            dest = cwd / e["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(e["content"])
        for rel in entry.get("delete", []):
            (cwd / rel).unlink(missing_ok=True)
        if entry.get("commit"):
            commit_all(cwd, f"mock: {title}")

        text = entry.get("text") or f"mock {task_id}: applied {len(edits)} edit(s)"
        raw: list[dict] = [{"type": "text", "text": text}]
        for command in entry.get("tool_commands", []):
            raw.append(
                {
                    "type": "tool",
                    "part": {
                        "type": "tool",
                        "name": "shell",
                        "state": {"status": "completed", "input": {"command": command}},
                    },
                }
            )
        return AgentRunResult(
            session_id=session,
            exit_code=0,
            stdout=text,
            stderr="",
            timed_out=False,
            text=text,
            raw_json=raw,
        )
