"""The task-state vocabulary and the reducers LangGraph uses to merge updates.

`task_state` is a mapping of ``task_id -> record``. A LangGraph node never
mutates it in place: it *returns a patch* under the ``task_state`` channel, and
the channel's reducer decides how that patch folds into the accumulator. The
three reducers here are that policy, kept pure so they can be tested without a
graph:

* :func:`merge_task_state` — per-task shallow merge (a node patching
  ``{"status": REVIEW}`` must not wipe the task's ``attempts``).
* :func:`keep_first` — for decision channels (``escalation``, ``approval``,
  ``summary``) where the first committed value is authoritative and later
  writes must not clobber it.
* :func:`escalation_reducer` — :func:`keep_first` plus an explicit
  ``{"clear": True}`` sentinel so a resumed run can clear the escalation.
"""

from enum import Enum


class TStatus(str, Enum):
    """A task's position in the lifecycle (see GLOSSARY: Approved/Integrated/Ready)."""

    PENDING = "pending"
    IMPLEMENTING = "implementing"
    REVIEW = "review"
    FIXING = "fixing"
    APPROVED = "approved"
    INTEGRATED = "integrated"
    ESCALATED = "escalated"


def new_task_state(task_id: str, *, origin: str = "plan", parent: str | None = None) -> dict:
    """A fresh task record: ``PENDING``, nothing attempted, nothing recorded.

    ``origin`` distinguishes plan Tasks from Repair tasks; ``parent`` is the
    task a repair was spawned from.
    """
    return {
        "task_id": task_id,
        "status": TStatus.PENDING,
        "attempts": 0,
        "fix_attempts": 0,
        "session_id": None,
        "branch": None,
        "worktree": None,
        "commits": [],
        "verdicts": [],
        "origin": origin,
        "parent": parent,
        "last_error": None,
        "violation": None,
    }


def merge_task_state(a: dict, b: dict) -> dict:
    """Merge ``b`` into ``a`` task by task; on a key collision the later value wins.

    Only the keys present in ``b`` change — a patch that moves a task to
    ``REVIEW`` leaves its ``attempts``/``commits`` untouched. Tasks absent from
    ``b`` are carried over unchanged, and neither input is mutated.
    """
    out = dict(a)
    for task_id, patch in b.items():
        merged = dict(out.get(task_id, {}))
        merged.update(patch)
        out[task_id] = merged
    return out


def keep_first(a, b):
    """``a`` when already set, else ``b`` — first writer wins."""
    return a if a is not None else b


def escalation_reducer(a, b):
    """First escalation wins; ``{"clear": True}`` clears it (a resume)."""
    if b == {"clear": True}:
        return None
    return a if a is not None else b
