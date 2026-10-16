"""The orchestrator graph.

Topology (design §5): ``START → prepare_run → approval_gate → scheduler`` and, per
dispatched task, ``t_implement → t_checks → t_commit → t_review → integrate``, with
a fix loop (``t_fix``) on any fixable failure, ``integrate`` re-entering
``scheduler`` until the graph is quiescent, then ``report``. ``t_escalate`` is the
parent-level sink every failure route converges on; it interrupts for a human and
applies the resume decision (retry, or mark the task escalated and report).

Why ``Command`` rather than conditional edges inside the task pipeline
-----------------------------------------------------------------------
A ``Send`` payload becomes the dispatched node's *first argument*, replacing the
shared state: LangGraph never hands a ``Send`` target the graph state. The task
pipeline needs both — the payload's ``task_id`` to disambiguate parallel branches,
and the shared state (``tasks``, ``task_state``, ``escalation``) to act — so every
pipeline node is wrapped by :func:`pipeline_node`, which reads the shared state
through the node-local read function LangGraph injects (``CONFIG_KEY_READ``, the
same mechanism its own ``ToolNode`` uses) and calls ``fn(state, payload, ...)``.
A conditional-edge router sees only the shared state and therefore cannot tell
which parallel branch it belongs to; the per-branch branch decision is expressed
instead as the node's own ``Command(goto=Send(<next>, payload))`` — the pinned
Send topology, with branch identity preserved.
"""

import hashlib
import sqlite3
from functools import partial, wraps
from typing import Annotated, TypedDict

from langgraph._internal._constants import CONF, CONFIG_KEY_READ
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from orch.adapters.base import AgentAdapter
from orch.artifacts import RunDir
from orch.config import Config
from orch.errors import PlanValidationError
from orch.gitops import branch_exists, create_branch, ensure_worktree, tracked_dirty
from orch.nodes import escalation_record, snapshot_run
from orch.nodes import integrate as integrate_nodes
from orch.nodes import report as report_nodes
from orch.nodes import task as task_nodes
from orch.plan import flag_risky, load_plan, validate_plan
from orch.scheduler import compute_dispatch
from orch.state import (
    TStatus,
    escalation_reducer,
    keep_first,
    merge_task_state,
    new_task_state,
)

INTEGRATION_WORKTREE = "orch-integration"

#: The channels a pipeline node reads; exactly the keys of :class:`GState`.
_STATE_CHANNELS = ["run_id", "tasks", "task_state", "approval", "escalation", "summary"]


class GState(TypedDict):
    """The graph state, locked for Tasks 13–17."""

    run_id: str
    tasks: dict[str, dict]
    task_state: Annotated[dict[str, dict], merge_task_state]
    approval: Annotated[dict | None, keep_first]
    escalation: Annotated[dict | None, escalation_reducer]
    summary: Annotated[str | None, keep_first]


def _shared_state() -> dict:
    """The current committed graph state, read from inside a node.

    LangGraph injects a read function under ``CONFIG_KEY_READ`` for every task; a
    channel with no value yet is simply absent from the result.
    """
    read = get_config().get(CONF, {}).get(CONFIG_KEY_READ)
    if read is None:  # not running inside a graph task (e.g. a direct call)
        return {}
    return read(list(_STATE_CHANNELS), True)


def _updates(result) -> dict:
    """The state patch a node returned, for the pre-return snapshot."""
    if isinstance(result, Command):
        return result.update if isinstance(result.update, dict) else {}
    if isinstance(result, dict):
        return result
    return {}


def pipeline_node(fn, *, cfg: Config, adapter: AgentAdapter, run_dir: RunDir):
    """Wrap a ``(state, payload)`` pipeline node into a LangGraph node callable.

    ``fn`` is bound to ``cfg``/``adapter``/``run_dir`` with ``functools.partial``;
    the wrapper supplies the shared state (which a ``Send`` payload replaced) and
    refreshes ``state.json``/``status.md`` on every return path (ruling R3).
    """
    bound = partial(fn, cfg=cfg, adapter=adapter, run_dir=run_dir)

    @wraps(fn)
    def node(payload: dict):
        state = _shared_state()
        result = bound(state, payload)
        snapshot_run(run_dir, cfg, state, updates=_updates(result))
        return result

    return node


# --- parent-level nodes ---------------------------------------------------


def prepare_run(state: GState, *, cfg: Config, run_dir: RunDir) -> dict:
    """Load and validate the plan, then set up the integration branch and worktree.

    Refuses (via escalation) a dirty target tree or an integration branch that
    already exists; on success initializes ``tasks``/``task_state`` from the plan.
    An invalid plan raises ``PlanValidationError`` for the CLI to report — it is a
    pre-run failure, not a graph path.
    """
    plan = load_plan(run_dir.plan_path)
    issues = validate_plan(plan, cfg.risky_patterns)
    if issues:
        raise PlanValidationError(
            [f"{tid or '-'}: {code}: {msg}" for tid, code, msg in issues]
        )
    plan = flag_risky(plan, cfg.risky_patterns)

    target = cfg.target_repo
    dirty = tracked_dirty(target)
    if dirty:
        updates = {
            "escalation": escalation_record(
                "dirty_tree", f"target repo has uncommitted tracked changes: {dirty}", None
            )
        }
        snapshot_run(run_dir, cfg, state, updates=updates)
        return updates

    if branch_exists(target, cfg.integration_branch):
        updates = {
            "escalation": escalation_record(
                "integration_branch_exists",
                f"integration branch already exists: {cfg.integration_branch}",
                None,
            )
        }
        snapshot_run(run_dir, cfg, state, updates=updates)
        return updates

    create_branch(target, cfg.integration_branch, cfg.base_ref)
    ensure_worktree(
        target, target / cfg.worktree_root / INTEGRATION_WORKTREE, cfg.integration_branch
    )

    updates = {
        "tasks": {t.id: t.model_dump(mode="json") for t in plan.tasks},
        "task_state": {t.id: new_task_state(t.id) for t in plan.tasks},
    }
    snapshot_run(run_dir, cfg, state, updates=updates)
    return updates


def approval_gate(state: GState, *, cfg: Config, run_dir: RunDir) -> dict:
    """Pause for the human to approve the plan; verify the plan hash on resume.

    The interrupt payload carries the plan path and the risky task ids. The
    resumed value is the approval; a ``plan_sha256`` that no longer matches
    ``plan.yaml`` escalates (the plan is immutable after approval).
    """
    if state.get("approval") is not None:
        return {}

    risky = [tid for tid, task in (state.get("tasks") or {}).items() if task.get("risky")]
    decision = interrupt({"plan_path": str(run_dir.plan_path), "risky": risky})

    expected = hashlib.sha256(run_dir.plan_path.read_bytes()).hexdigest()
    if not isinstance(decision, dict) or decision.get("plan_sha256") != expected:
        updates = {
            "escalation": escalation_record(
                "plan_changed", "plan.yaml changed after approval (sha256 mismatch)", None
            )
        }
        snapshot_run(run_dir, cfg, state, updates=updates)
        return updates

    updates = {"approval": decision}
    snapshot_run(run_dir, cfg, state, updates=updates)
    return updates


def _decide(state: GState, cfg: Config):
    """The scheduler's pure decision for the current state."""
    return compute_dispatch(
        tasks=state.get("tasks") or {},
        task_state=state.get("task_state") or {},
        concurrency=cfg.concurrency,
    )


def scheduler(state: GState, *, cfg: Config, run_dir: RunDir) -> dict:
    """Write the deadlock escalation, if any; the routing decision is separate.

    Also the merge point: an ``integrate`` completion re-enters ``scheduler`` to
    re-evaluate the Ready set.
    """
    updates: dict = {}
    if _decide(state, cfg).state == "deadlock":
        updates["escalation"] = escalation_record(
            "deadlock", "no task is ready and none is in flight", None
        )
    snapshot_run(run_dir, cfg, state, updates=updates)
    return updates


def route_scheduler(state: GState, *, cfg: Config):
    """``Send`` one packet per dispatched task; ``[]`` to wait; a string otherwise."""
    decision = _decide(state, cfg)
    if decision.state == "ok":
        return [Send("t_implement", {"task_id": tid}) for tid in decision.dispatch]
    if decision.state == "deadlock":
        return "t_escalate"
    return "report"


def t_escalate(state: GState, *, cfg: Config, run_dir: RunDir) -> Command:
    """Pause the run for a human and apply the resume decision.

    The run stops on a checkpointed ``interrupt(state["escalation"])``. On resume with
    ``{"action": "retry", "task_id": X}`` the task is reset to ``PENDING`` with a
    fresh budget (``human_retry`` is recorded) and re-dispatched; a deadlock retry
    (no task) clears the escalation and re-enters the scheduler; anything else marks
    the failing task ``ESCALATED`` and ends the run at ``report``.
    """
    escalation = state.get("escalation")
    if escalation is None:
        # Defensive: nothing to escalate is not a reason to pause.
        return Command(goto="report")

    decision = interrupt(escalation)

    if isinstance(decision, dict) and decision.get("action") == "retry":
        tid = decision.get("task_id") or escalation.get("task_id")
        if tid:
            updates = {
                "escalation": {"clear": True},
                "task_state": {
                    tid: {
                        "status": TStatus.PENDING,
                        "attempts": 0,
                        "fix_attempts": 0,
                        "human_retry": True,
                        "violation": None,
                        "last_error": None,
                    }
                },
            }
            snapshot_run(run_dir, cfg, state, updates=updates)
            return Command(update=updates, goto=Send("t_implement", {"task_id": tid}))
        if escalation.get("kind") == "deadlock":
            updates = {"escalation": {"clear": True}}
            snapshot_run(run_dir, cfg, state, updates=updates)
            return Command(update=updates, goto="scheduler")
        updates = {"escalation": {"clear": True}}
        snapshot_run(run_dir, cfg, state, updates=updates)
        return Command(update=updates, goto="report")

    tid = escalation.get("task_id")
    if tid:
        updates = {"task_state": {tid: {"status": TStatus.ESCALATED}}}
        snapshot_run(run_dir, cfg, state, updates=updates)
        return Command(update=updates, goto="report")
    return Command(goto="report")


def _route_after_prepare(state: GState):
    return "report" if state.get("escalation") is not None else "approval_gate"


def _route_after_approval(state: GState):
    return "report" if state.get("escalation") is not None else "scheduler"


def _route_after_integrate(state: GState):
    return "t_escalate" if state.get("escalation") is not None else "scheduler"


def build_graph(*, cfg: Config, adapter: AgentAdapter, run_dir: RunDir):
    """Compile the orchestrator graph with a SQLite checkpointer at ``run_dir``.

    The caller invokes with ``config={"configurable": {"thread_id": run_id}}``.
    The connection is owned by the returned ``SqliteSaver``, which the compiled
    graph holds.
    """
    graph = StateGraph(GState)

    graph.add_node("prepare_run", partial(prepare_run, cfg=cfg, run_dir=run_dir))
    graph.add_node("approval_gate", partial(approval_gate, cfg=cfg, run_dir=run_dir))
    graph.add_node("scheduler", partial(scheduler, cfg=cfg, run_dir=run_dir))
    graph.add_node(
        "t_implement",
        pipeline_node(task_nodes.t_implement, cfg=cfg, adapter=adapter, run_dir=run_dir),
    )
    graph.add_node(
        "t_checks", pipeline_node(task_nodes.t_checks, cfg=cfg, adapter=adapter, run_dir=run_dir)
    )
    graph.add_node(
        "t_commit", pipeline_node(task_nodes.t_commit, cfg=cfg, adapter=adapter, run_dir=run_dir)
    )
    graph.add_node(
        "t_review", pipeline_node(task_nodes.t_review, cfg=cfg, adapter=adapter, run_dir=run_dir)
    )
    graph.add_node(
        "t_fix", pipeline_node(task_nodes.t_fix, cfg=cfg, adapter=adapter, run_dir=run_dir)
    )
    graph.add_node(
        "integrate",
        pipeline_node(integrate_nodes.integrate, cfg=cfg, adapter=adapter, run_dir=run_dir),
    )
    graph.add_node("t_escalate", partial(t_escalate, cfg=cfg, run_dir=run_dir))
    graph.add_node("report", partial(report_nodes.report, cfg=cfg, run_dir=run_dir))

    graph.add_edge(START, "prepare_run")
    graph.add_conditional_edges(
        "prepare_run", _route_after_prepare, ["approval_gate", "report"]
    )
    graph.add_conditional_edges(
        "approval_gate", _route_after_approval, ["scheduler", "report"]
    )
    graph.add_conditional_edges(
        "scheduler", partial(route_scheduler, cfg=cfg), ["t_implement", "report", "t_escalate"]
    )
    graph.add_conditional_edges(
        "integrate", _route_after_integrate, ["scheduler", "t_escalate"]
    )
    graph.add_edge("report", END)

    # The state carries `TStatus` members, which msgpack stores as a tagged ext;
    # allowlist our own enum so deserialization stays explicit and does not depend
    # on the permissive (and eventually strict) default.
    conn = sqlite3.connect(str(run_dir.checkpoints_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("orch.state", "TStatus")])
    return graph.compile(checkpointer=SqliteSaver(conn, serde=serde))
