"""Per-task pipeline nodes: implement → checks → commit → review, with a fix loop.

Every node here is a ``(state, payload, *, cfg, adapter, run_dir)`` function. The
graph binds the last three through ``functools.partial`` and hands the node the
Send payload's ``task_id`` (``tid = payload["task_id"]``); the shared graph state
is recovered by ``graph.pipeline_node``. Each node makes its routing decision with
``Command(goto=...)`` — a ``Send`` for the next step, or the parent-level
``"t_escalate"`` string — because a conditional-edge router would see only the
shared state and could not tell which parallel branch it belongs to.

Any *fixable* failure — BLOCKED, a ``files_allowed`` violation, red checks, a review
reject — routes through :func:`fix_or_escalate` (the one budget rule): back to the
implementer's own session via ``t_fix`` while attempts remain, otherwise to
``t_escalate`` for a human. Model errors, timeouts and a destructive command in the
worker's transcript are not fixable and escalate at once, without spending a fix.

The orchestrator, not the worker, owns the commit: ``commit_all`` runs only after
the authoritative checks are green and the ``files_allowed`` boundary is clean.
"""

import json
from pathlib import Path, PurePosixPath

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
from orch.guard import extract_tool_commands, is_pre_authorized, scan_destructive
from orch.nodes import escalation_record, merge_updates, passthrough_env
from orch.plan import TaskDef
from orch.prompts import build_fix_prompt, build_review_prompt, build_task_prompt
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
    """Changed paths that no ``files_allowed`` pattern full-matches.

    Ownership is declared as globs (``src/**``), so a changed path is allowed iff
    ANY pattern full-matches it via :meth:`PurePosixPath.full_match`; a literal
    pattern still matches only itself. At commit time the worker's literal edits
    are what matter; the boundary fails closed, so an unmatched path is a
    violation. Sorted so a violation is reproducible.
    """
    changed = set(changed_files(worktree))
    allowed = {
        path
        for path in changed
        if any(PurePosixPath(path).full_match(pat) for pat in files_allowed)
    }
    return sorted(changed - allowed)


def attempt_failures(task_state: dict, tid: str) -> list[str]:
    """The findings a fix should address, in the pinned precedence order.

    A single task can carry several failure signals at once; the most specific — and
    the one the *last* attempt actually produced — wins:

    1. a ``files_allowed`` violation (stray edits),
    2. ``last_error`` (a worker's ``BLOCKED:`` reply),
    3. the failed check names from the last check run,
    4. otherwise the last reviewer verdict's finding notes.
    """
    record = task_state.get(tid) or {}

    violation = record.get("violation")
    if violation:
        return [f"edited disallowed path: {path}" for path in violation]

    last_error = record.get("last_error")
    if last_error:
        return [str(last_error)]

    failed = record.get("failed_checks")
    if failed:
        return [f"check failed: {name}" for name in failed]

    verdicts = record.get("verdicts") or []
    if verdicts:
        notes = [
            finding.get("note")
            for finding in (verdicts[-1].get("findings") or [])
            if finding.get("note")
        ]
        return notes or ["review rejected"]

    return []


def fix_or_escalate(state: dict, payload: dict, *, cfg: Config):
    """The pinned budget rule, in one place. Returns ``(goto, escalation)``.

    Any *fixable* failure — BLOCKED, a ``files_allowed`` violation, red checks, a
    review reject — goes back to ``t_fix`` while the task has fix attempts left
    (``fix_attempts < cfg.retries``). When the budget is spent the escalation is
    returned instead and the node routes to the parent-level ``t_escalate``.
    """
    tid = payload["task_id"]
    record = (state.get("task_state") or {}).get(tid) or {}
    if record.get("fix_attempts", 0) < cfg.retries:
        return Send("t_fix", payload), None
    findings = attempt_failures(state.get("task_state") or {}, tid)
    return "t_escalate", escalation_record(
        "task_retries_exhausted", "; ".join(findings), tid
    )


def route_after_checks(state: dict, payload: dict, *, cfg: Config):
    """Green checks → ``Send("t_commit")``; a red run enters the fix budget.

    Returns ``(goto, escalation)``; a non-``None`` escalation means the fix budget is
    spent and the node must route the failing task to ``t_escalate``.
    """
    record = (state.get("task_state") or {}).get(payload["task_id"]) or {}
    if record.get("failed_checks"):
        return fix_or_escalate(state, payload, cfg=cfg)
    return Send("t_commit", payload), None


def route_after_review(state: dict, payload: dict, *, cfg: Config):
    """An approve → ``Send("integrate")``; a reject enters the fix budget."""
    verdicts = ((state.get("task_state") or {}).get(payload["task_id"]) or {}).get(
        "verdicts"
    ) or []
    if verdicts and verdicts[-1].get("verdict") == "approve":
        return Send("integrate", payload), None
    return fix_or_escalate(state, payload, cfg=cfg)


def _fail(updates: dict, update: dict, escalation: dict | None) -> None:
    """Record a spent-budget escalation and mark the failing task ``ESCALATED``."""
    if escalation is None:
        return
    updates["escalation"] = escalation
    update["status"] = TStatus.ESCALATED


def t_implement(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Run the worker in a fresh session on its own branch/worktree.

    The worker's edits are left uncommitted on purpose — ``t_commit`` owns the
    commit. ``BLOCKED:`` in the reply is recorded as ``last_error`` and edits outside
    ``files_allowed`` as a ``violation``; both are fixable failures routed through
    :func:`fix_or_escalate`.
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
        # A fresh attempt starts from a clean slate: the previous attempt's boundary
        # violation and BLOCKED text are recomputed below rather than shadowing this
        # attempt's outcome.
        "violation": None,
        "last_error": None,
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
        # Governance: the orchestrator does not sandbox the worker's shell, so an
        # EXECUTED destructive command is a hard stop — immediate and unfixable
        # (never routed through the fix budget). Only tool-call commands are
        # scanned: prose and tool output (e.g. a deploy runbook the worker read)
        # are not actions.
        hits = scan_destructive("\n".join(extract_tool_commands(result.raw_json)))
        if hits:
            escalation = escalation_record(
                "destructive_command", "; ".join(hits), tid
            )
        elif result.timed_out:
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
                update["violation"] = allowed_violation(worktree, task.files_allowed) or None

    updates: dict = {"task_state": {tid: update}}
    if escalation is not None:
        # Model errors, timeouts and destructive commands are not fixable by the
        # worker: escalate at once.
        updates["escalation"] = escalation
        return Command(update=updates, goto="t_escalate")

    if update["last_error"] or update["violation"]:
        goto, exhausted = fix_or_escalate(merge_updates(state, updates), payload, cfg=cfg)
        _fail(updates, update, exhausted)
    else:
        goto = Send("t_checks", payload)
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

    update: dict = {
        "evidence": [str(r.output_path) for r in results],
        "failed_checks": [r.name for r in results if r.exit_code != 0],
    }
    updates: dict = {"task_state": {tid: update}}
    goto, exhausted = route_after_checks(merge_updates(state, updates), payload, cfg=cfg)
    if checks_green(results):
        update["status"] = TStatus.REVIEW
    _fail(updates, update, exhausted)
    return Command(update=updates, goto=goto)


def t_commit(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Commit the worker's changes — unless they crossed the ``files_allowed`` line.

    Review Focus #1: a violation must never be committed, so it is routed through the
    fix budget before ``git add``. The marker from the implement attempt is not
    trusted alone: the boundary is RE-derived from the worktree at commit time, so a
    fix round cannot slip a stray edit past it (the live pilot's t1 fix completed
    ``web/package-lock.json`` off-list — necessary and approved, but the commit gate
    must re-check). A worker that changed nothing is flagged in ``notes`` and left to
    the reviewer to judge.
    """
    tid = payload["task_id"]
    task = TaskDef(**state["tasks"][tid])
    record = state["task_state"].get(tid) or {}
    worktree = task_worktree(cfg, tid)

    violation = list(record.get("violation") or []) or allowed_violation(
        worktree, task.files_allowed
    )
    if violation:
        # Never commit a stray edit: route through the same budget as any other
        # fixable failure so a clean fix can restore the boundary.
        update: dict = {"violation": list(violation)}
        updates: dict = {"task_state": {tid: update}}
        goto, exhausted = fix_or_escalate(state, payload, cfg=cfg)
        _fail(updates, update, exhausted)
        return Command(update=updates, goto=goto)

    update: dict = {}
    sha = commit_all(worktree, f"task({tid}): {task.title} — {state['run_id']}")
    if sha is None:
        update["notes"] = [*(record.get("notes") or []), "no changes to commit"]
    else:
        update["commits"] = [*(record.get("commits") or []), sha]
    return Command(update={"task_state": {tid: update}}, goto=Send("t_review", payload))


def t_fix(
    state: dict, payload: dict, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir
) -> Command:
    """Repair a failed task, then re-run its checks.

    The fix runs in the *implementer's own session* (design §5.4 step 6) with the
    classified findings, and works on the same worktree — so the ``files_allowed``
    boundary is recomputed here: a fix that cleans up a stray edit restores a
    committable state.

    ``last_error``/``violation`` are one-shot markers describing the attempt this fix
    answers: building ``findings`` above consumes them, so a later failure on the same
    task is classified from its own fresh signals, never from text this fix already saw.
    """
    tid = payload["task_id"]
    task = TaskDef(**state["tasks"][tid])
    record = state["task_state"].get(tid) or {}
    worktree = task_worktree(cfg, tid)
    fix_attempts = record.get("fix_attempts", 0) + 1
    findings = attempt_failures(state["task_state"], tid)

    update: dict = {
        "status": TStatus.FIXING,
        "fix_attempts": fix_attempts,
        # Consumed into `findings`; recomputed below for this attempt's own outcome.
        "last_error": None,
        "violation": None,
    }
    escalation = None
    try:
        result = adapter.run(
            role="fix",
            prompt=build_fix_prompt(task, findings),
            cwd=worktree,
            session_id=record.get("session_id"),
            title=f"{state['run_id']}/{tid}/fix{fix_attempts}",
            env=passthrough_env(cfg),
        )
    except ModelUnavailableError as e:
        escalation = escalation_record("model_unavailable", str(e), tid)
    else:
        if result.timed_out:
            escalation = escalation_record(
                "adapter_timeout",
                f"fix timed out after {cfg.timeout_minutes} minute(s)",
                tid,
            )
        else:
            update["violation"] = allowed_violation(worktree, task.files_allowed) or None
            if "BLOCKED:" in result.text:
                update["last_error"] = result.text

    updates: dict = {"task_state": {tid: update}}
    if escalation is not None:
        updates["escalation"] = escalation
        return Command(update=updates, goto="t_escalate")
    return Command(update=updates, goto=Send("t_checks", payload))


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
    escalation = None
    for attempt in range(2):
        try:
            result = adapter.run(
                role="review",
                prompt=prompt if attempt == 0 else prompt + VERDICT_RETRY_SUFFIX,
                cwd=worktree,
                session_id=None,
                title=title,
                env=passthrough_env(cfg),
            )
        except ModelUnavailableError as e:
            # A model error is not a verdict problem: escalate at once (mirroring
            # ``t_implement``) rather than retrying the same unusable model.
            escalation = escalation_record("model_unavailable", str(e), tid)
            break
        if result.timed_out:
            escalation = escalation_record(
                "adapter_timeout",
                f"review timed out after {cfg.timeout_minutes} minute(s)",
                tid,
            )
            break
        try:
            verdict = parse_verdict(result.text)
        except ValueError as e:
            error = str(e)
            continue
        break

    if escalation is not None:
        return Command(update={"escalation": escalation}, goto="t_escalate")

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
    if verdict.risky and not is_pre_authorized(state.get("approval"), tid):
        # A runtime risk surprise stops the run unless the human pre-authorized this
        # exact task at the gate (``approve --accept-risky``); the verdict is stored
        # either way, and a pre-authorized risk proceeds down the normal approve path.
        updates["escalation"] = escalation_record(
            "risk_reviewer", "the reviewer flagged this change as risky", tid
        )
        return Command(update=updates, goto="t_escalate")

    if verdict.verdict == "approve":
        update["status"] = TStatus.APPROVED
    goto, exhausted = route_after_review(merge_updates(state, updates), payload, cfg=cfg)
    _fail(updates, update, exhausted)
    return Command(update=updates, goto=goto)
