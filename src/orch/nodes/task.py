"""Per-task pipeline nodes: implement → checks → commit → review.

Every node here is a ``(state, payload, *, cfg, adapter, run_dir)`` function. The
graph binds the last three through ``functools.partial`` and hands the node the
Send payload's ``task_id`` (``tid = payload["task_id"]``); the shared graph state
is recovered by ``graph.pipeline_node``. Each node makes its routing decision with
``Command(goto=...)`` — a ``Send`` for the next step, or the parent-level
``"t_escalate"`` string — because a conditional-edge router would see only the
shared state and could not tell which parallel branch it belongs to.

The orchestrator, not the worker, owns the commit: ``commit_all`` runs only after
the authoritative checks are green and the ``files_allowed`` boundary is clean.
"""

import json
from pathlib import Path

from langgraph.types import Command, Send

from orch.adapters.base import AgentAdapter
from orch.artifacts import RunDir, atomic_write
from orch.checks import checks_green, run_checks
from orch.config import Config
from orch.errors import ModelUnavailableError
from orch.gitops import (
    branch_exists,
    changed_files,
    commit_all,
    create_branch,
    diff_three_dot,
    ensure_worktree,
)
from orch.nodes import escalation_record, merge_updates, passthrough_env
from orch.plan import TaskDef
from orch.prompts import build_review_prompt, build_task_prompt
from orch.review import parse_verdict
from orch.state import TStatus

#: Appended to the reviewer prompt on the single retry after an unparseable reply.
VERDICT_RETRY_SUFFIX = (
    "\n\nYour previous reply was not parseable. "
    "Return ONLY the JSON object specified above."
)


def task_branch(task_id: str) -> str:
    """``agent/<TASK-ID>`` — the task's branch, cut from the integration branch."""
    return f"agent/{task_id}"


def task_worktree(cfg: Config, task_id: str) -> Path:
    """``<target>/<worktree_root>/<TASK-ID>`` — the task's private working tree."""
    return cfg.target_repo / cfg.worktree_root / task_id


def allowed_violation(worktree: Path, files_allowed: list[str]) -> list[str]:
    """Changed paths that fall outside ``files_allowed``.

    Literal-path comparison only: the plan's ownership is a hard boundary, and at
    commit time the worker's literal edits are what matter (glob-aware ownership
    lives in the scheduler). Sorted so a violation is reproducible.
    """
    return sorted(set(changed_files(worktree)) - set(files_allowed))


def pre_authorized(task_id: str) -> bool:
    """Whether the human pre-authorized this task's risky change.

    A stub until Task 17 wires the approval gate's ``pre_authorized`` list through;
    until then a reviewer's ``risky`` flag always stops the run for a human.
    """
    return False


def route_after_implement(state: dict, payload: dict):
    """``Send("t_checks", payload)``, or ``"t_escalate"`` when the run is failing."""
    if state.get("escalation") is not None:
        return "t_escalate"
    return Send("t_checks", payload)


def route_after_checks(state: dict, payload: dict):
    """Happy path: green checks → ``Send("t_commit", payload)``; red → escalate."""
    if state.get("escalation") is not None:
        return "t_escalate"
    return Send("t_commit", payload)


def route_after_review(state: dict, payload: dict):
    """Happy path: approve → ``Send("integrate", payload)``; reject → escalate."""
    verdicts = (state.get("task_state", {}).get(payload["task_id"]) or {}).get("verdicts") or []
    last = verdicts[-1] if verdicts else {}
    if last.get("verdict") == "approve":
        return Send("integrate", payload)
    return "t_escalate"


def t_implement(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Run the worker in a fresh session on its own branch/worktree.

    The worker's edits are left uncommitted on purpose — ``t_commit`` owns the
    commit. ``BLOCKED:`` in the reply is recorded as ``last_error``; edits outside
    ``files_allowed`` are recorded as a ``violation`` for ``t_commit`` to refuse.
    """
    tid = payload["task_id"]
    task = TaskDef(**state["tasks"][tid])
    record = state["task_state"].get(tid) or {}
    worktree = task_worktree(cfg, tid)

    if not branch_exists(cfg.target_repo, task_branch(tid)):
        create_branch(cfg.target_repo, task_branch(tid), cfg.integration_branch)
    ensure_worktree(cfg.target_repo, worktree, task_branch(tid))

    update = {
        "status": TStatus.IMPLEMENTING,
        "branch": task_branch(tid),
        "worktree": str(worktree),
        "attempts": record.get("attempts", 0) + 1,
    }
    prompt = build_task_prompt(task, run_id=state["run_id"], repo_note=f"Worktree: {worktree}")

    escalation = None
    try:
        result = adapter.run(
            role="implement",
            prompt=prompt,
            cwd=worktree,
            session_id=None,
            title=f"{state['run_id']}/{tid}/implement",
            env=passthrough_env(cfg),
        )
    except ModelUnavailableError as e:
        escalation = escalation_record("model_unavailable", str(e), tid)
    else:
        if result.timed_out:
            escalation = escalation_record(
                "adapter_timeout",
                f"implement timed out after {cfg.timeout_minutes} minute(s)",
                tid,
            )
        else:
            update["session_id"] = result.session_id
            if "BLOCKED:" in result.text:
                update["last_error"] = result.text
            else:
                violation = allowed_violation(worktree, task.files_allowed)
                if violation:
                    update["violation"] = violation

    updates: dict = {"task_state": {tid: update}}
    if escalation is not None:
        updates["escalation"] = escalation
    goto = route_after_implement(merge_updates(state, updates), payload)
    return Command(update=updates, goto=goto)


def t_checks(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Run the task's checks in its worktree — the authoritative pass/fail truth.

    A check command that cannot be spawned raises ``OSError`` from the runner
    (deferred finding from Task 7); that is an escalation, never a pass.
    """
    tid = payload["task_id"]
    task = TaskDef(**state["tasks"][tid])
    worktree = task_worktree(cfg, tid)

    try:
        results = run_checks(
            task.checks,
            worktree=worktree,
            evidence_dir=run_dir.evidence,
            task_id=tid,
            env=passthrough_env(cfg),
            timeout_s=cfg.timeout_minutes * 60,
        )
    except OSError as e:
        updates = {"escalation": escalation_record("check_spawn_error", str(e), tid)}
        return Command(update=updates, goto="t_escalate")

    update: dict = {"evidence": [str(r.output_path) for r in results]}
    updates: dict = {"task_state": {tid: update}}
    if checks_green(results):
        update["status"] = TStatus.REVIEW
    else:
        failed = [r.name for r in results if r.exit_code != 0]
        updates["escalation"] = escalation_record(
            "checks_failed", f"failed checks: {', '.join(failed)}", tid
        )
    return Command(update=updates, goto=route_after_checks(merge_updates(state, updates), payload))


def t_commit(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Commit the worker's changes — unless they crossed the ``files_allowed`` line.

    Review Focus #1: a violation must never be committed, so it escalates before
    ``git add``; a worker that changed nothing is flagged in ``notes`` and left to
    the reviewer to judge.
    """
    tid = payload["task_id"]
    task = TaskDef(**state["tasks"][tid])
    record = state["task_state"].get(tid) or {}
    worktree = task_worktree(cfg, tid)

    violation = record.get("violation")
    if violation:
        escalation = escalation_record(
            "file_violation", f"changes outside files_allowed: {', '.join(violation)}", tid
        )
        return Command(update={"escalation": escalation}, goto="t_escalate")

    update: dict = {}
    sha = commit_all(worktree, f"task({tid}): {task.title} — {state['run_id']}")
    if sha is None:
        update["notes"] = [*(record.get("notes") or []), "no changes to commit"]
    else:
        update["commits"] = [*(record.get("commits") or []), sha]
    return Command(update={"task_state": {tid: update}}, goto=Send("t_review", payload))


def t_review(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Review the committed task in a fresh session and route on the verdict.

    The diff and the verdict are both written under ``evidence/``. An unparseable
    reply gets exactly one retry with the same prompt plus a one-line reminder;
    a second failure escalates rather than guessing.
    """
    tid = payload["task_id"]
    task = TaskDef(**state["tasks"][tid])
    record = state["task_state"].get(tid) or {}
    worktree = task_worktree(cfg, tid)

    diff_path = run_dir.evidence / f"{tid}.diff"
    atomic_write(diff_path, diff_three_dot(worktree, cfg.integration_branch))
    prompt = build_review_prompt(
        task,
        run_id=state["run_id"],
        diff_path=diff_path,
        evidence_paths=[Path(p) for p in record.get("evidence") or []],
        integration_branch=cfg.integration_branch,
    )
    title = f"{state['run_id']}/{tid}/review"

    verdict = None
    error = "reviewer reply was never parsed"
    for attempt in range(2):
        result = adapter.run(
            role="review",
            prompt=prompt if attempt == 0 else prompt + VERDICT_RETRY_SUFFIX,
            cwd=worktree,
            session_id=None,
            title=title,
            env=passthrough_env(cfg),
        )
        try:
            verdict = parse_verdict(result.text)
        except ValueError as e:
            error = str(e)
            continue
        break

    if verdict is None:
        updates = {"escalation": escalation_record("verdict_unparseable", error, tid)}
        return Command(update=updates, goto="t_escalate")

    verdicts = [*(record.get("verdicts") or []), verdict.model_dump(mode="json")]
    atomic_write(
        run_dir.evidence / f"{tid}.verdict.json",
        json.dumps(verdict.model_dump(mode="json"), indent=2) + "\n",
    )

    update: dict = {"verdicts": verdicts}
    updates: dict = {"task_state": {tid: update}}
    if verdict.risky and not pre_authorized(tid):
        updates["escalation"] = escalation_record(
            "risk_reviewer", "the reviewer flagged this change as risky", tid
        )
        return Command(update=updates, goto="t_escalate")

    goto = route_after_review(merge_updates(state, updates), payload)
    if goto == "t_escalate":
        findings = [f"{f.severity}: {f.file or '-'}: {f.note}" for f in verdict.findings]
        updates["escalation"] = escalation_record(
            "review_rejected", "; ".join(findings) or "review rejected", tid
        )
    else:
        update["status"] = TStatus.APPROVED
    return Command(update=updates, goto=goto)
