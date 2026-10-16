"""Governance guards: the destructive scan, pre-authorization, and plan pinning.

Three small, pure-ish policies the graph and the CLI both lean on, kept in one
place so the rules are pinned in code rather than prose:

* :func:`scan_destructive` — the deny-list of shell commands a worker must never
  run. The orchestrator does not sandbox the agent's shell, so the honest control
  is a loud stop: the matched patterns are reported and the run escalates.
* :func:`is_pre_authorized` — a reviewer's runtime ``risky`` verdict stops the run
  *unless* the human pre-authorized that exact task at the approval gate. The
  approval payload is the single source of that authority.
* :func:`verify_plan_hash` — ``plan.yaml`` is immutable after approval. A resume
  whose checkpointed ``plan_sha256`` no longer matches the file has had its plan
  swapped underneath it, so it escalates instead of executing a plan nobody saw.
"""

import hashlib
import re

from orch.artifacts import RunDir
from orch.errors import Escalation

#: The destructive-command deny-list, in the pinned order the report preserves.
DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Temp-directory cleanup is not the hazard; `rm -rf /`, /etc, /usr, /home are.
    re.compile(r"rm\s+-rf\s+/(?!tmp\b|var/tmp\b|dev/shm\b)"),
    re.compile(r"DROP\s+TABLE"),
    re.compile(r"git\s+push\b"),
    re.compile(r"git\s+reset\s+--hard"),
    # `\b` keeps the pipeline check from matching `| shasum`.
    re.compile(r"curl[^|]*\|\s*sh\b"),
    re.compile(r":\(\)\{:\|:&\};:"),
)


def scan_destructive(text: str) -> list[str]:
    """The deny-list patterns matched by ``text``, in list order and deduped.

    Returns the *pattern strings* (not the matched text) so the escalation message
    names the rule that tripped, reproducibly.
    """
    hits = [pattern.pattern for pattern in DESTRUCTIVE_PATTERNS if pattern.search(text)]
    return list(dict.fromkeys(hits))


def extract_tool_commands(raw_json: list[dict] | None) -> list[str]:
    """Shell commands the worker ACTUALLY ran, from the JSONL tool events.

    Only executed tool calls count: assistant prose and tool *outputs* are
    deliberately excluded — a file that merely mentions ``git reset --hard``
    (e.g. a deploy runbook the worker read) is not a destructive action. The
    first live pilot escalation was exactly that false positive.
    """
    commands: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "tool":
                state = node.get("state")
                if isinstance(state, dict):
                    inp = state.get("input")
                    if isinstance(inp, dict) and isinstance(inp.get("command"), str):
                        commands.append(inp["command"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for line in raw_json or []:
        walk(line)
    return commands


def is_pre_authorized(approval: dict | None, task_id: str) -> bool:
    """True iff ``approval`` is a dict whose ``pre_authorized`` list names ``task_id``."""
    return isinstance(approval, dict) and task_id in (approval.get("pre_authorized") or [])


def verify_plan_hash(state: dict, run_dir: RunDir) -> None:
    """Raise :class:`Escalation` when the pinned plan hash no longer matches.

    An unapproved run has no pinned hash, so there is nothing to verify. Once an
    ``approval`` exists, ``plan.yaml`` must hash to exactly the ``plan_sha256`` it
    carries — any other value means the file changed after approval.
    """
    approval = state.get("approval")
    if approval is None:
        return None
    expected = hashlib.sha256(run_dir.plan_path.read_bytes()).hexdigest()
    if approval.get("plan_sha256") != expected:
        raise Escalation(
            "plan_changed", "plan.yaml changed after approval (sha256 mismatch)"
        )
    return None
