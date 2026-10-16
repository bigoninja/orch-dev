"""The ``tasks`` channel reducer: two parallel Repair spawns must merge.

``integrate`` runs in parallel for independent tasks; when two of them each spawn
a Repair task in the same super-step, both writes land on the ``tasks`` channel.
Without :func:`merge_task_defs` the second write is a concurrent update and
LangGraph raises ``InvalidUpdateError`` before either repair can run.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from orch.graph import merge_task_defs


def test_merge_task_defs_keeps_both_spawns():
    a = {"A-repair": {"id": "A-repair"}}
    b = {"B-repair": {"id": "B-repair"}}

    assert merge_task_defs(a, b) == {
        "A-repair": {"id": "A-repair"},
        "B-repair": {"id": "B-repair"},
    }


def test_merge_task_defs_lets_a_later_write_win_on_collision():
    a = {"A-repair": {"id": "A-repair", "title": "old"}}
    b = {"A-repair": {"id": "A-repair", "title": "new"}}

    assert merge_task_defs(a, b)["A-repair"]["title"] == "new"


class _SpawnState(TypedDict):
    tasks: Annotated[dict, merge_task_defs]


def test_two_parallel_spawns_merge_in_one_superstep():
    def left(_state):
        return {"tasks": {"A-repair": {"id": "A-repair"}}}

    def right(_state):
        return {"tasks": {"B-repair": {"id": "B-repair"}}}

    graph = StateGraph(_SpawnState)
    graph.add_node("left", left)
    graph.add_node("right", right)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    graph.add_edge("left", END)
    graph.add_edge("right", END)

    out = graph.compile().invoke({"tasks": {}})

    assert set(out["tasks"]) == {"A-repair", "B-repair"}
