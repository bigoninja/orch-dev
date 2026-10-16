"""The scheduler's dispatch decision — the whole policy as one pure function.

Design §5.2: Ready means every dependency is Integrated; at most `concurrency`
tasks are in flight; two tasks whose `files_allowed` overlap never run together
(the later one stays Ready but is deferred, deterministically by plan order).
`compute_dispatch` reads no files and calls no LLM, so the table below *is* the
specification.
"""

import pytest

from orch.scheduler import DispatchDecision, compute_dispatch
from orch.state import TStatus


def task(
    tid: str,
    *,
    depends_on: tuple[str, ...] = (),
    files: tuple[str, ...] = ("src/x.py",),
) -> dict:
    """A `TaskDef.model_dump()`-shaped task record — the keys the scheduler reads."""
    return {"id": tid, "depends_on": list(depends_on), "files_allowed": list(files)}


def tstate(*pairs: tuple[str, TStatus | str]) -> dict[str, dict]:
    """A `task_state`-shaped mapping built from ``(task_id, status)`` pairs."""
    return {tid: {"task_id": tid, "status": status} for tid, status in pairs}


# --- Ready set & plan order -------------------------------------------------


def test_independent_tasks_dispatch_together_in_plan_order():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["A", "B"], "ok"
    )


def test_plan_order_comes_from_the_tasks_dict_not_from_task_state():
    tasks = {"B": task("B", files=("src/b.py",)), "A": task("A", files=("src/a.py",))}
    # task_state happened to be built in the other order; it must not matter.
    state = {"A": {"status": TStatus.PENDING}, "B": {"status": TStatus.PENDING}}

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["B", "A"], "ok"
    )


def test_concurrency_cap_truncates_the_dispatch_in_plan_order():
    tasks = {tid: task(tid, files=(f"src/{tid}.py",)) for tid in ("A", "B", "C", "D")}
    state = tstate(*((tid, TStatus.PENDING) for tid in ("A", "B", "C", "D")))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=2) == DispatchDecision(
        ["A", "B"], "ok"
    )


# --- Dependency gating ------------------------------------------------------


def test_task_with_an_unintegrated_dependency_is_held_back():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", depends_on=("A",), files=("src/b.py",))}
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["A"], "ok"
    )


def test_task_is_ready_once_its_dependency_is_integrated():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", depends_on=("A",), files=("src/b.py",))}
    state = tstate(("A", TStatus.INTEGRATED), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["B"], "ok"
    )


def test_every_dependency_must_be_integrated():
    tasks = {
        "A": task("A", files=("src/a.py",)),
        "B": task("B", files=("src/b.py",)),
        "C": task("C", depends_on=("A", "B"), files=("src/c.py",)),
        "D": task("D", files=("src/d.py",)),
    }
    state = tstate(
        ("A", TStatus.INTEGRATED),
        ("B", TStatus.PENDING),
        ("C", TStatus.PENDING),
        ("D", TStatus.PENDING),
    )

    # C's deps are not *all* Integrated, so it waits; its unrelated peer D does not.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["B", "D"], "ok"
    )


# --- In-flight accounting ---------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [TStatus.IMPLEMENTING, TStatus.REVIEW, TStatus.FIXING, TStatus.APPROVED],
)
def test_each_in_flight_status_consumes_a_slot(status):
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", status), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=1) == DispatchDecision([], "ok")
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=2) == DispatchDecision(
        ["B"], "ok"
    )


def test_empty_dispatch_with_work_in_flight_is_ok_not_deadlock():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", depends_on=("A",), files=("src/b.py",))}
    state = tstate(("A", TStatus.IMPLEMENTING), ("B", TStatus.PENDING))

    # The graph waits: A's integrate completion re-enters the scheduler.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "ok")


def test_an_approved_task_is_in_flight_not_a_deadlock():
    tasks = {"A": task("A", files=("src/a.py",))}
    state = tstate(("A", TStatus.APPROVED))

    # Reviewed but not yet integrated: its `integrate` branch is still pending,
    # so the scheduler waits (`"ok"` with an empty dispatch), never `"deadlock"`.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "ok")


def test_a_task_blocked_by_an_approved_task_waits_without_deadlock():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", depends_on=("A",), files=("src/b.py",))}
    state = tstate(("A", TStatus.APPROVED), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "ok")


def test_an_approved_task_consumes_a_concurrency_slot():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", TStatus.APPROVED), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=1) == DispatchDecision([], "ok")


def test_concurrency_already_reached_dispatches_nothing():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",)), "C": task("C", files=("src/c.py",))}
    state = tstate(("A", TStatus.IMPLEMENTING), ("B", TStatus.REVIEW), ("C", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=2) == DispatchDecision([], "ok")


# --- File-ownership exclusion -----------------------------------------------


def test_ownership_defers_a_ready_task_that_overlaps_an_in_flight_task():
    tasks = {"A": task("A", files=("src/**",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", TStatus.IMPLEMENTING), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "ok")


def test_ownership_defers_a_later_task_in_the_same_wave_without_stalling_the_rest():
    tasks = {
        "A": task("A", files=("src/**",)),
        "B": task("B", files=("src/b.py",)),
        "C": task("C", files=("docs/readme.md",)),
    }
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING), ("C", TStatus.PENDING))

    # A claims src/**; B is skipped (not failed) and the plan carries on to C.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["A", "C"], "ok"
    )


@pytest.mark.parametrize(
    "first, second",
    [
        (["src/orch/scheduler.py"], ["src/orch/scheduler.py"]),  # the same file
        (["src/**"], ["src/orch/scheduler.py"]),  # a glob covering the file
        (["src/orch/*.py"], ["src/orch/scheduler.py"]),  # a sibling glob form
        (["src/orch"], ["src/orch/scheduler.py"]),  # a directory claims its subtree
    ],
)
def test_overlapping_ownership_defers_the_later_task(first, second):
    tasks = {"A": task("A", files=tuple(first)), "B": task("B", files=tuple(second))}
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["A"], "ok"
    )


@pytest.mark.parametrize(
    "first, second",
    [
        (["src/**"], ["tests/**"]),
        (["src/orch/scheduler.py"], ["src/orch/state.py"]),
        (["docs/**"], ["src/orch/*.py"]),
    ],
)
def test_disjoint_ownership_runs_concurrently(first, second):
    tasks = {"A": task("A", files=tuple(first)), "B": task("B", files=tuple(second))}
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["A", "B"], "ok"
    )


# --- Repair-task ownership exemption ----------------------------------------


def test_a_repair_task_is_not_deferred_by_its_own_approved_parent():
    """A repair works on its parent's files by design: the parent must not block it.

    The parent is APPROVED — in flight, reviewed but not yet Integrated — so it
    claims its files. Without the exemption it would defer its own repair forever.
    """
    tasks = {
        "B": task("B", files=("src/b.py",)),
        "B-repair": task("B-repair", files=("src/b.py",)),
    }
    state = {
        "B": {"task_id": "B", "status": TStatus.APPROVED},
        "B-repair": {
            "task_id": "B-repair",
            "status": TStatus.PENDING,
            "origin": "repair",
            "parent": "B",
        },
    }

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["B-repair"], "ok"
    )


def test_a_repair_task_is_still_deferred_by_a_non_parent_overlap():
    """Only the parent is exempt: any other in-flight claim still defers the repair."""
    tasks = {
        "B": task("B", files=("src/b.py",)),
        "C": task("C", files=("src/**",)),
        "B-repair": task("B-repair", files=("src/b.py",)),
    }
    state = {
        "B": {"status": TStatus.APPROVED},
        "C": {"status": TStatus.IMPLEMENTING},
        "B-repair": {"status": TStatus.PENDING, "origin": "repair", "parent": "B"},
    }

    # C's ``src/**`` overlaps the repair's file and is not its parent → deferred.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "ok")


def test_a_plan_task_still_defers_to_an_overlapping_in_flight_task():
    """The exemption is repair-only: a plain task's ownership behavior is unchanged."""
    tasks = {
        "A": task("A", files=("src/a.py",)),
        "B": task("B", files=("src/a.py",)),
    }
    state = tstate(("A", TStatus.APPROVED), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "ok")


# --- Terminal states --------------------------------------------------------


def test_done_when_every_task_is_integrated():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", TStatus.INTEGRATED), ("B", TStatus.INTEGRATED))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "done")


def test_escalation_wins_over_ready_work():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", TStatus.ESCALATED), ("B", TStatus.PENDING))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        [], "escalated"
    )


def test_escalation_wins_over_done():
    tasks = {"A": task("A", files=("src/a.py",)), "B": task("B", files=("src/b.py",))}
    state = tstate(("A", TStatus.ESCALATED), ("B", TStatus.INTEGRATED))

    # Escalation is checked first: the run stopped for a reason, even if the
    # remaining tasks happen to be Integrated.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        [], "escalated"
    )


def test_dependency_cycle_with_nothing_in_flight_is_a_deadlock():
    tasks = {
        "A": task("A", depends_on=("B",), files=("src/a.py",)),
        "B": task("B", depends_on=("A",), files=("src/b.py",)),
    }
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING))

    # Nothing ready, nothing in flight, unfinished tasks remain: only a human
    # editing the plan can unstick this.
    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        [], "deadlock"
    )


def test_a_dependency_cycle_hides_behind_a_third_integrated_task():
    tasks = {
        "A": task("A", depends_on=("B",), files=("src/a.py",)),
        "B": task("B", depends_on=("A",), files=("src/b.py",)),
        "C": task("C", files=("src/c.py",)),
    }
    state = tstate(("A", TStatus.PENDING), ("B", TStatus.PENDING), ("C", TStatus.INTEGRATED))

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        [], "deadlock"
    )


def test_an_empty_plan_is_done():
    # Not reachable via the plan schema (min_length=1), but the vacuous truth
    # ("every task is Integrated") keeps the function total.
    assert compute_dispatch(tasks={}, task_state={}, concurrency=3) == DispatchDecision([], "done")


# --- Status normalization & interface ---------------------------------------


def test_statuses_may_arrive_serialized_as_strings():
    tasks = {
        "A": task("A", files=("src/a.py",)),
        "B": task("B", depends_on=("A",), files=("src/b.py",)),
        "C": task("C", files=("src/c.py",)),
    }
    state = {"A": {"status": "integrated"}, "B": {"status": "pending"}, "C": {"status": "fixing"}}

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision(
        ["B"], "ok"
    )


def test_all_string_statuses_can_report_done():
    tasks = {"A": task("A", files=("src/a.py",))}
    state = {"A": {"status": "integrated"}}

    assert compute_dispatch(tasks=tasks, task_state=state, concurrency=3) == DispatchDecision([], "done")


def test_missing_task_state_record_reads_as_pending():
    tasks = {"A": task("A", files=("src/a.py",))}

    assert compute_dispatch(tasks=tasks, task_state={}, concurrency=1) == DispatchDecision(["A"], "ok")


def test_unknown_status_is_rejected_loudly():
    # A status the scheduler does not know must not be waved through as PENDING:
    # dispatching a task that is already mid-flight would double it up.
    with pytest.raises(ValueError, match="not-a-status"):
        compute_dispatch(
            tasks={"A": task("A")}, task_state={"A": {"status": "not-a-status"}}, concurrency=1
        )


def test_compute_dispatch_is_keyword_only():
    with pytest.raises(TypeError):
        compute_dispatch({"A": task("A")}, {"A": {"status": TStatus.PENDING}}, 1)
