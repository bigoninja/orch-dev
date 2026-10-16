"""The state vocabulary and the LangGraph reducers that carry it.

`task_state` is a per-task-id mapping of mutable task records; LangGraph
channels only know how to *merge* what a node returns, so the merge policy has
to be a pure function here, tested on its own.
"""

from orch.state import (
    TStatus,
    escalation_reducer,
    keep_first,
    merge_task_state,
    new_task_state,
)


def test_status_vocabulary_is_the_lifecycle_vocabulary():
    assert [s.name for s in TStatus] == [
        "PENDING",
        "IMPLEMENTING",
        "REVIEW",
        "FIXING",
        "APPROVED",
        "INTEGRATED",
        "ESCALATED",
    ]
    assert [s.value for s in TStatus] == [
        "pending",
        "implementing",
        "review",
        "fixing",
        "approved",
        "integrated",
        "escalated",
    ]


def test_status_is_a_str_enum():
    # `state.json` and `status.md` both want the string value; `TStatus(...)`
    # round-trips from a serialized state.
    assert isinstance(TStatus.PENDING, str)
    assert TStatus("integrated") is TStatus.INTEGRATED


def test_new_task_state_baseline():
    state = new_task_state("B1B-01")

    assert state == {
        "task_id": "B1B-01",
        "status": TStatus.PENDING,
        "attempts": 0,
        "fix_attempts": 0,
        "session_id": None,
        "branch": None,
        "worktree": None,
        "commits": [],
        "verdicts": [],
        "origin": "plan",
        "parent": None,
        "last_error": None,
        "violation": None,
    }


def test_new_task_state_accepts_origin_and_parent():
    state = new_task_state("REPAIR-1", origin="repair", parent="B1B-06")

    assert state["origin"] == "repair"
    assert state["parent"] == "B1B-06"


def test_new_task_state_mutable_defaults_are_not_shared():
    a = new_task_state("A")
    b = new_task_state("B")
    a["commits"].append("deadbeef")
    a["verdicts"].append({"verdict": "approve"})

    assert b["commits"] == []
    assert b["verdicts"] == []


def test_merge_task_state_merges_per_task():
    a = {"T1": {"status": TStatus.IMPLEMENTING, "attempts": 1}}
    b = {"T1": {"status": TStatus.REVIEW}, "T2": {"status": TStatus.PENDING}}

    merged = merge_task_state(a, b)

    assert merged["T1"] == {"status": TStatus.REVIEW, "attempts": 1}
    assert merged["T2"] == {"status": TStatus.PENDING}


def test_merge_task_state_later_writes_win_per_key_and_leaves_inputs_alone():
    a = {"T1": {"attempts": 1, "session_id": None}}
    b = {"T1": {"attempts": 2, "session_id": "ses_1"}}

    merged = merge_task_state(a, b)

    assert merged == {"T1": {"attempts": 2, "session_id": "ses_1"}}
    # Reducers must not mutate the accumulator they were handed.
    assert a == {"T1": {"attempts": 1, "session_id": None}}


def test_merge_task_state_keeps_tasks_absent_from_the_patch():
    a = {"T1": {"attempts": 1}, "T2": {"attempts": 5}}

    assert merge_task_state(a, {}) == a


def test_merge_task_state_preserves_caller_order():
    a = {"T1": {}, "T2": {}}
    b = {"T2": {"attempts": 1}, "T3": {}}

    assert list(merge_task_state(a, b)) == ["T1", "T2", "T3"]


def test_keep_first_prefers_an_existing_value():
    assert keep_first("first", "second") == "first"
    assert keep_first({"kind": "x"}, None) == {"kind": "x"}


def test_keep_first_falls_back_when_unset():
    assert keep_first(None, "second") == "second"
    assert keep_first(None, None) is None


def test_escalation_reducer_lets_the_first_escalation_win():
    escal = {"kind": "retries_exhausted", "message": "m", "task_id": "T1"}

    assert escalation_reducer(escal, None) is escal
    # A second escalation must not clobber the one that stopped the run.
    assert escalation_reducer(escal, {"kind": "other"}) is escal


def test_escalation_reducer_records_the_first_escalation():
    escal = {"kind": "retries_exhausted", "message": "m", "task_id": "T1"}

    assert escalation_reducer(None, escal) == escal


def test_escalation_reducer_clears_on_explicit_clear():
    escal = {"kind": "retries_exhausted", "message": "m", "task_id": "T1"}

    assert escalation_reducer(escal, {"clear": True}) is None
    assert escalation_reducer(None, {"clear": True}) is None
