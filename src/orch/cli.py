"""The ``orch`` command line: one process, one command, one exit code.

Exit codes (the plan's pinned contract): ``0`` success or awaiting-approval,
``2`` escalation, ``1`` error. Every command that drives the graph opens its own
SQLite checkpointer connection and hands it to :func:`orch.graph.build_graph`
(``conn=...``), closing it in a ``finally`` — a run's checkpoints live on disk, so
each CLI invocation is a fresh, self-contained reader/writer of them.

The commands are thin: load the config, resolve the run directory, build the real
adapter (``orch.adapters.factory`` — the mock only under the two test env vars),
invoke or resume the compiled graph, and translate the outcome into output plus an
exit code. The pause detection mirrors the E2E harness exactly: a paused invoke
returns ``__interrupt__``, and an escalation additionally sets the ``escalation``
channel.
"""

import argparse
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from orch import __version__


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"orch {__version__}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from orch.config import init_config
    from orch.errors import ConfigError

    try:
        cfg = init_config(Path(args.target_repo), Path(args.config))
    except ConfigError as e:
        print(f"error: {e}")
        return 1
    Path(cfg.state_dir).mkdir(exist_ok=True)
    Path(cfg.runs_dir).mkdir(exist_ok=True)
    print(f"wrote {args.config}\nnext: orch plan \"<brief>\"")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from orch.adapters.factory import factory
    from orch.artifacts import RunDir
    from orch.config import load_config
    from orch.errors import ConfigError, PlanValidationError
    from orch.nodes.planner import run_planner

    try:
        cfg = load_config(Path(args.config))
    except ConfigError as e:
        print(f"error: {e}")
        return 1

    run_id = args.run or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = RunDir.create(Path(cfg.runs_dir), run_id)
    adapter = factory(cfg)

    try:
        plan = run_planner(
            cfg=cfg,
            adapter=adapter,
            brief=args.brief,
            source=args.source,
            run_dir=run_dir,
        )
    except OSError as e:
        # A missing ``--from`` file (or any unreadable path) is an input error, not
        # a crash: report the path and exit.
        print(f"error: {e.filename or e}")
        return 1
    except PlanValidationError as e:
        invalid = run_dir.root / "plan.invalid.yaml"
        if invalid.exists():
            print(f"invalid plan written to {invalid}")
        for issue in e.issues:
            print(f"issue: {issue}")
        return 1

    risky = [task.id for task in plan.tasks if task.risky]
    print(f"plan written to {run_dir.plan_path}")
    if risky:
        print("risky tasks: " + ", ".join(risky))
    else:
        print("no risky tasks")
    return 0


# --- shared helpers for the run-driving commands -------------------------------


def _config_or_error(config_path: str):
    """Load the config; on :class:`ConfigError` print and return ``(None, 1)``."""
    from orch.config import load_config
    from orch.errors import ConfigError

    try:
        return load_config(Path(config_path)), 0
    except ConfigError as e:
        print(f"error: {e}")
        return None, 1


def _finished(root: Path) -> bool:
    """True when the run's ``state.json`` carries a summary (the run is over)."""
    state = root / "state.json"
    if not state.is_file():
        return False
    try:
        return bool(json.loads(state.read_text(encoding="utf-8")).get("summary"))
    except (OSError, json.JSONDecodeError):
        return False


def _run_dir(cfg, run_id: str | None, *, awaiting: bool):
    """Resolve the run directory for ``run_id``, or the latest eligible run.

    ``awaiting`` restricts the automatic pick to runs whose ``state.json`` has no
    ``summary`` (one still to be driven); ``status`` passes ``False`` so it can
    report a finished run too. A run qualifies when it holds a ``plan.yaml``.
    """
    from orch.artifacts import RunDir

    runs_dir = Path(cfg.runs_dir)
    if run_id:
        if not (runs_dir / run_id / "plan.yaml").is_file():
            print(f"error: no plan.yaml for run {run_id!r} under {runs_dir}")
            return None
        return RunDir.create(runs_dir, run_id)

    if not runs_dir.is_dir():
        print(f"error: no runs under {runs_dir}")
        return None

    candidates = [
        root
        for root in runs_dir.iterdir()
        if root.is_dir()
        and (root / "plan.yaml").is_file()
        and not (awaiting and _finished(root))
    ]
    if not candidates:
        print(f"error: no run awaiting execution under {runs_dir}")
        return None
    return RunDir.create(runs_dir, max(candidates, key=lambda r: r.name).name)


@contextmanager
def _graph(cfg, run_dir):
    """A compiled graph on a CLI-owned checkpointer connection, closed on exit.

    The connection is opened here and handed to ``build_graph(conn=...)`` so it is
    closed on every path — normal return, escalation, or an exception mid-invoke
    (the ``ResourceWarning`` leak the Task 12 review flagged).
    """
    from orch.adapters.factory import factory
    from orch.graph import build_graph

    conn = sqlite3.connect(str(run_dir.checkpoints_path), check_same_thread=False)
    try:
        yield build_graph(cfg=cfg, adapter=factory(cfg), run_dir=run_dir, conn=conn)
    finally:
        conn.close()


def _thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _invoke(graph, run_id: str) -> dict:
    """Start the run: the initial state, stopping at the first pause."""
    return graph.invoke({"run_id": run_id}, config=_thread(run_id))


def _resume(graph, run_id: str, value) -> dict:
    from langgraph.types import Command

    return graph.invoke(Command(resume=value), config=_thread(run_id))


def _continue(graph, run_id: str) -> dict:
    """Re-enter a crashed (not paused) run from its last checkpoint."""
    return graph.invoke(None, config=_thread(run_id))


def _pending_interrupt(graph, run_id: str):
    """The run's pending :class:`~langgraph.types.Interrupt`, or ``None``."""
    snapshot = graph.get_state(_thread(run_id))
    interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
    if not interrupts:
        for task in getattr(snapshot, "tasks", ()) or ():
            if task.interrupts:
                interrupts = tuple(task.interrupts)
                break
    return interrupts[0] if interrupts else None


def _interrupt_value(interrupt) -> dict:
    """The interrupt payload as a dict (an interrupt may carry any JSON value)."""
    return interrupt.value if isinstance(interrupt.value, dict) else {}


def _gate_payload(run_dir, *, pre_authorized=()) -> dict:
    """The pinned approval decision: the plan's hash is re-checked on resume."""
    return {
        "approved": True,
        "pre_authorized": list(pre_authorized),
        "plan_sha256": hashlib.sha256(run_dir.plan_path.read_bytes()).hexdigest(),
    }


def _outcome(result: dict, *, run_dir, run_id: str) -> int:
    """Print the invoke outcome and return its exit code.

    An escalation is checked first and is always non-zero: a task-failure pause
    carries the escalation alongside ``__interrupt__``, while a pre-run refusal
    (e.g. a dirty target tree) reaches ``report`` with the escalation still set and
    no interrupt. An interrupt without an escalation is the approval gate.
    """
    escalation = result.get("escalation")
    if escalation is not None:
        task_id = escalation.get("task_id")
        print(f"escalation: {escalation.get('kind')}")
        print(f"task: {task_id if task_id is not None else '-'}")
        print(f"message: {escalation.get('message')}")
        print("fixed the cause? then: orch resume")
        return 2
    if result.get("__interrupt__"):
        print(
            f"awaiting approval: edit {run_dir.plan_path}, "
            f"then: orch approve --run {run_id}"
        )
        return 0
    if result.get("summary"):
        print(result["summary"])
    return 0


# --- run / approve / resume / status -------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from orch.artifacts import run_lock
    from orch.errors import LockHeldError, PlanValidationError

    cfg, rc = _config_or_error(args.config)
    if cfg is None:
        return rc
    run_dir = _run_dir(cfg, args.run, awaiting=True)
    if run_dir is None:
        return 1
    run_id = run_dir.root.name

    try:
        with run_lock(Path(cfg.state_dir)):
            with _graph(cfg, run_dir) as graph:
                result = _invoke(graph, run_id)
                if (
                    args.yes
                    and result.get("__interrupt__")
                    and result.get("escalation") is None
                ):
                    result = _resume(graph, run_id, _gate_payload(run_dir))
    except LockHeldError as e:
        print(f"error: {e}")
        return 1
    except PlanValidationError as e:
        for issue in e.issues:
            print(f"issue: {issue}")
        return 1

    return _outcome(result, run_dir=run_dir, run_id=run_id)


def _edit_plan(run_dir) -> bool:
    """Open ``$EDITOR`` (default ``vi``) on ``plan.yaml``; False if it cannot run."""
    editor = os.environ.get("EDITOR") or "vi"
    try:
        subprocess.run([*shlex.split(editor), str(run_dir.plan_path)], check=False)
    except OSError as e:
        print(f"error: cannot run editor {editor!r}: {e}")
        return False
    return True


def cmd_approve(args: argparse.Namespace) -> int:
    from orch.artifacts import run_lock
    from orch.errors import LockHeldError, PlanValidationError
    from orch.plan import flag_risky, load_plan, validate_plan

    cfg, rc = _config_or_error(args.config)
    if cfg is None:
        return rc
    run_dir = _run_dir(cfg, args.run, awaiting=True)
    if run_dir is None:
        return 1
    run_id = run_dir.root.name

    try:
        with run_lock(Path(cfg.state_dir)):
            with _graph(cfg, run_dir) as graph:
                at_gate = _pending_interrupt(graph, run_id)
                if at_gate is None or "plan_path" not in _interrupt_value(at_gate):
                    print(
                        "error: the run is not paused at the approval gate; "
                        "nothing to approve"
                    )
                    print(f"next: orch status --run {run_id}")
                    return 1

                if args.edit and not _edit_plan(run_dir):
                    return 1

                try:
                    plan = load_plan(run_dir.plan_path)
                except PlanValidationError as e:
                    for issue in e.issues:
                        print(f"issue: {issue}")
                    return 1

                # A human may edit the plan at the gate, so re-validate before
                # resuming — an invalid plan is refused here, never executed.
                issues = validate_plan(plan, cfg.risky_patterns)
                if issues:
                    for task_id, code, message in issues:
                        print(f"issue: {task_id or '-'}: {code}: {message}")
                    return 1

                plan = flag_risky(plan, cfg.risky_patterns)
                risky = [task.id for task in plan.tasks if task.risky]
                pre_authorized = risky if args.accept_risky else []
                result = _resume(
                    graph, run_id, _gate_payload(run_dir, pre_authorized=pre_authorized)
                )
    except LockHeldError as e:
        print(f"error: {e}")
        return 1

    return _outcome(result, run_dir=run_dir, run_id=run_id)


def cmd_resume(args: argparse.Namespace) -> int:
    from orch.artifacts import run_lock
    from orch.errors import LockHeldError

    cfg, rc = _config_or_error(args.config)
    if cfg is None:
        return rc
    run_dir = _run_dir(cfg, args.run, awaiting=True)
    if run_dir is None:
        return 1
    run_id = run_dir.root.name

    try:
        with run_lock(Path(cfg.state_dir)):
            with _graph(cfg, run_dir) as graph:
                pending = _pending_interrupt(graph, run_id)
                value = _interrupt_value(pending) if pending is not None else {}
                if "kind" in value:
                    # A paused escalation: apply the human's decision (retry this task).
                    decision: dict = {"action": "retry"}
                    if args.task:
                        decision["task_id"] = args.task
                    result = _resume(graph, run_id, decision)
                else:
                    # A crash (no interrupt) — or a gate pause, which just re-pauses —
                    # is a plain re-entry from the last checkpoint.
                    result = _continue(graph, run_id)
    except LockHeldError as e:
        print(f"error: {e}")
        return 1

    return _outcome(result, run_dir=run_dir, run_id=run_id)


def cmd_status(args: argparse.Namespace) -> int:
    cfg, rc = _config_or_error(args.config)
    if cfg is None:
        return rc
    run_dir = _run_dir(cfg, args.run, awaiting=False)
    if run_dir is None:
        return 1

    path = run_dir.state_path if args.as_json else run_dir.status_path
    if not path.is_file():
        print(f"error: no {path.name} for run {run_dir.root.name}")
        return 1
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orch")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("version", help="print version")
    v.set_defaults(fn=cmd_version)
    i = sub.add_parser("init", help="write orch.config.yaml for a target repo")
    i.add_argument("target_repo", help="path to the target git repository")
    i.add_argument("--config", default="orch.config.yaml", help="path to write the config")
    i.set_defaults(fn=cmd_init)
    pl = sub.add_parser("plan", help="run the planner and write plan.yaml")
    pl.add_argument("brief", help="the brief to plan against")
    pl.add_argument("--from", dest="source", default=None, help="source document to plan from")
    pl.add_argument("--config", default="orch.config.yaml", help="path to the config")
    pl.add_argument("--run", default=None, help="run id (defaults to a UTC timestamp)")
    pl.set_defaults(fn=cmd_plan)
    r = sub.add_parser("run", help="validate and execute; stop at the approval gate")
    r.add_argument(
        "--yes", action="store_true", help="approve the plan immediately and run to the end"
    )
    r.add_argument("--run", default=None, help="run id (defaults to the latest awaiting run)")
    r.add_argument("--config", default="orch.config.yaml", help="path to the config")
    r.set_defaults(fn=cmd_run)
    a = sub.add_parser("approve", help="resume past the approval gate")
    a.add_argument("--edit", action="store_true", help="open $EDITOR on plan.yaml first")
    a.add_argument(
        "--accept-risky",
        action="store_true",
        help="pre-authorize every currently flagged risky task",
    )
    a.add_argument("--run", default=None, help="run id (defaults to the latest awaiting run)")
    a.add_argument("--config", default="orch.config.yaml", help="path to the config")
    a.set_defaults(fn=cmd_approve)
    rs = sub.add_parser("resume", help="continue after an escalation or a crash")
    rs.add_argument("--task", default=None, help="task id to retry (defaults to the escalated one)")
    rs.add_argument("--run", default=None, help="run id (defaults to the latest awaiting run)")
    rs.add_argument("--config", default="orch.config.yaml", help="path to the config")
    rs.set_defaults(fn=cmd_resume)
    st = sub.add_parser("status", help="print the run board")
    st.add_argument("--json", dest="as_json", action="store_true", help="print state.json")
    st.add_argument("--run", default=None, help="run id (defaults to the latest run)")
    st.add_argument("--config", default="orch.config.yaml", help="path to the config")
    st.set_defaults(fn=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
