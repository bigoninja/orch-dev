"""End-to-end integration hardening: conflicts, bounded repair, parallel deferral.

Three integration behaviours are pinned here, driven through the real compiled
graph against a throwaway git repository:

* a merge conflict or a red post-merge check spawns **exactly one** Repair task
  (``agent/<TASK-ID>-repair``), which runs the ordinary pipeline in its own
  worktree and, when it merges, resolves its parent to Integrated;
* one repair per failure event, and **no nested repairs**: a repair that fails
  integration escalates ``repair_exhausted`` instead of spawning a second one;
* re-integrating an already-merged task is a no-op (Review Focus #5), and a wave
  that owns an overlapping file defers the later task until its owner integrates,
  so the deferred branch is cut from an integration tip that already holds the
  owner's merge.

The failure injections are surgical: ``merge_no_ff``/``run_checks`` are stubbed
*once* so the graph's spawn/cap/parent-resolution logic is under test; the real
git mechanics are pinned by the ``gitops`` unit tests.
"""

from pathlib import Path

import pytest
import yaml
from langgraph.types import Command

from orch.adapters import MockAdapter
from orch.artifacts import RunDir
from orch.checks import CheckResult
from orch.gitops import head_sha, is_ancestor
from orch.graph import build_graph
from orch.nodes import integrate as integrate_nodes
from orch.state import TStatus

from conftest import git, make_config, make_demo_repo, plan_sha256, run_once, write_plan

#: The checkpointed thread each scenario's first invoke and resume share.
THREAD = "integration"


class RecordingMock(MockAdapter):
    """A ``MockAdapter`` that records every ``run`` call (role/session/title)."""

    def __init__(self, scenario_path: Path):
        super().__init__(scenario_path)
        self.calls: list[dict] = []

    def run(self, **kwargs):  # type: ignore[override]
        result = super().run(**kwargs)
        self.calls.append({**kwargs, "result_session": result.session_id})
        return result


def _task(tid: str, *, file: str, depends_on: tuple[str, ...] = ()) -> dict:
    """A one-file task: write ``OK`` to ``file``, checked with ``grep``."""
    return {
        "id": tid,
        "title": f"write OK to {file} for {tid}",
        "description": f"Write the marker OK to {file}.",
        "depends_on": list(depends_on),
        "files_allowed": [file],
        "acceptance_criteria": [f"{file} contains OK"],
        "checks": {"test": {"cmd": f"grep -q OK {file}"}},
    }


def _boot(tmp_path: Path, tasks: list[dict], scenario: dict, *, rid="R1", retries=2):
    """Build a graph over a fresh demo repo with the given plan and scenario."""
    repo = make_demo_repo(tmp_path / "demo")
    cfg = make_config(repo, tmp_path, retries=retries)
    run_dir = RunDir.create(Path(cfg.runs_dir), rid)
    write_plan(run_dir, tasks, rid=rid)
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    adapter = RecordingMock(scenario_path)
    graph = build_graph(cfg=cfg, adapter=adapter, run_dir=run_dir)
    return cfg, run_dir, graph, adapter


def _approve_and_run(graph, run_dir: RunDir, *, rid="R1", thread=THREAD):
    """First invoke to the approval gate, then resume with the plan approval."""
    first = run_once(graph, rid, thread=thread)
    assert first.get("__interrupt__"), "the approval gate must interrupt the first invoke"
    return graph.invoke(
        Command(resume={"approved": True, "pre_authorized": [], "plan_sha256": plan_sha256(run_dir)}),
        config={"configurable": {"thread_id": thread}},
    )


def _repairs(final: dict) -> list[str]:
    """The repair task ids present in the final plan (``tasks``)."""
    return [tid for tid in final["tasks"] if tid.endswith("-repair")]


def _failed_check(name: str, evidence_dir: Path) -> CheckResult:
    """A synthetic, always-red check result for a surgical injection."""
    return CheckResult(
        name=name,
        cmd="injected",
        cwd=None,
        exit_code=1,
        duration_s=0.0,
        output_path=evidence_dir / f"{name}.injected.log",
    )


def _conflict_once_for(branch_name: str, monkeypatch, *, conflicts=("b.txt",)):
    """Patch ``integrate.merge_no_ff`` to report a conflict for one branch, once."""
    real = integrate_nodes.merge_no_ff
    fired = {"n": 0}

    def conflicted(worktree, branch):
        if branch == branch_name and not fired["n"]:
            fired["n"] += 1
            return False, list(conflicts)
        return real(worktree, branch)

    monkeypatch.setattr(integrate_nodes, "merge_no_ff", conflicted)
    return fired


# --- 1. merge conflict → one repair → parent integrated -----------------------


def test_a_merge_conflict_spawns_one_repair_that_resolves_the_parent(tmp_path, monkeypatch):
    scenario = {
        "A": {
            "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "B": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        # The repair runs the ordinary pipeline via its own scenario entries.
        "B-repair": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    cfg, run_dir, graph, adapter = _boot(
        tmp_path, [_task("A", file="a.txt"), _task("B", file="b.txt")], scenario
    )
    fired = _conflict_once_for("agent/B", monkeypatch)

    final = _approve_and_run(graph, run_dir)

    assert fired["n"] == 1
    assert _repairs(final) == ["B-repair"], "exactly one repair must be spawned"

    repair = final["task_state"]["B-repair"]
    assert repair["origin"] == "repair"
    assert repair["parent"] == "B"
    assert final["tasks"]["B-repair"]["depends_on"] == []
    assert final["tasks"]["B-repair"]["files_allowed"] == ["b.txt"]
    assert repair["repair_event"] == head_sha(cfg.target_repo / cfg.worktree_root / "B")

    # The repair ran its own implement + review in its own branch/worktree.
    titles = [call["title"] for call in adapter.calls]
    assert "R1/B-repair/implement" in titles
    assert "R1/B-repair/review" in titles
    assert (cfg.target_repo / cfg.worktree_root / "B-repair").is_dir()
    assert "agent/B-repair" in git(cfg.target_repo, "branch", "--list", "agent/B-repair").stdout

    # B resolves to Integrated when its repair merges, and B's work is on the branch.
    assert final["task_state"]["A"]["status"] == TStatus.INTEGRATED
    assert final["task_state"]["B"]["status"] == TStatus.INTEGRATED
    assert repair["status"] == TStatus.INTEGRATED
    assert final.get("escalation") is None
    integration = cfg.target_repo / cfg.worktree_root / "orch-integration"
    assert "OK" in (integration / "b.txt").read_text(encoding="utf-8")


# --- 2. one repair per failure event / no nested repairs ----------------------


def test_a_failing_repair_escalates_repair_exhausted_without_spawning_another(tmp_path, monkeypatch):
    scenario = {
        "B": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "B-repair": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    _cfg, run_dir, graph, _adapter = _boot(tmp_path, [_task("B", file="b.txt")], scenario)
    _conflict_once_for("agent/B", monkeypatch)

    # The repair's own post-merge check fails: it must escalate, not self-repair.
    real_checks = integrate_nodes.run_checks
    fired = {"n": 0}

    def postmerge_red(checks, *, worktree, evidence_dir, task_id, env, timeout_s):
        if task_id == "B-repair.postmerge" and not fired["n"]:
            fired["n"] += 1
            return [_failed_check("post_merge_checks", evidence_dir)]
        return real_checks(
            checks, worktree=worktree, evidence_dir=evidence_dir, task_id=task_id, env=env, timeout_s=timeout_s
        )

    monkeypatch.setattr(integrate_nodes, "run_checks", postmerge_red)

    paused = _approve_and_run(graph, run_dir)

    assert fired["n"] == 1
    assert paused.get("__interrupt__"), "a repair that cannot integrate must pause the run"
    assert paused["escalation"]["kind"] == "repair_exhausted"
    assert paused["escalation"]["task_id"] == "B-repair"
    assert _repairs(paused) == ["B-repair"], "no nested repair may be spawned"


def test_reentering_a_failed_parent_whose_event_already_has_a_repair_escalates(tmp_path, monkeypatch):
    """The cap is keyed by failure event, so the same event cannot spawn twice."""
    scenario = {
        "B": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "B-repair": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("B", file="b.txt")], scenario)
    _conflict_once_for("agent/B", monkeypatch)
    final = _approve_and_run(graph, run_dir)
    assert final["task_state"]["B-repair"]["status"] == TStatus.INTEGRATED

    # Re-enter B's integrate at the *same* failure event: a repair already exists.
    _conflict_once_for("agent/B", monkeypatch)
    state = {
        **final,
        "task_state": {
            **final["task_state"],
            "B": {**final["task_state"]["B"], "status": TStatus.APPROVED},
        },
    }
    update = integrate_nodes.integrate(state, {"task_id": "B"}, cfg=cfg, adapter=adapter, run_dir=run_dir)

    assert update.get("escalation", {}).get("kind") == "repair_exhausted"
    assert update["escalation"]["task_id"] == "B"
    assert "tasks" not in update, "the cap must not spawn a second repair"


# --- 3. post-merge check failure → repair with the pinned reason --------------


def test_a_red_post_merge_check_spawns_a_repair_named_for_the_reason(tmp_path, monkeypatch):
    scenario = {
        "A": {
            "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "A-repair": {
            "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    cfg, run_dir, graph, _adapter = _boot(tmp_path, [_task("A", file="a.txt")], scenario)

    real_checks = integrate_nodes.run_checks
    fired = {"n": 0}

    def postmerge_red(checks, *, worktree, evidence_dir, task_id, env, timeout_s):
        if task_id == "A.postmerge" and not fired["n"]:
            fired["n"] += 1
            return [_failed_check("post_merge_checks", evidence_dir)]
        return real_checks(
            checks, worktree=worktree, evidence_dir=evidence_dir, task_id=task_id, env=env, timeout_s=timeout_s
        )

    monkeypatch.setattr(integrate_nodes, "run_checks", postmerge_red)

    final = _approve_and_run(graph, run_dir)

    assert fired["n"] == 1, "only A's first post-merge run is made red"
    assert _repairs(final) == ["A-repair"]
    assert final["tasks"]["A-repair"]["title"] == "Repair: post_merge_checks for A"
    assert "post_merge_checks" in final["tasks"]["A-repair"]["description"]
    assert "A" in final["tasks"]["A-repair"]["description"]
    assert final["tasks"]["A-repair"]["acceptance_criteria"][-1] == (
        "Repair resolves the post_merge_checks integration failure"
    )
    assert final["task_state"]["A-repair"]["repair_event"] == head_sha(
        cfg.target_repo / cfg.worktree_root / "A"
    )
    assert final["task_state"]["A"]["status"] == TStatus.INTEGRATED
    assert final["task_state"]["A-repair"]["status"] == TStatus.INTEGRATED
    assert final.get("escalation") is None


# --- 4. idempotent re-merge ---------------------------------------------------


def test_reintegrating_an_already_merged_task_is_a_no_op(tmp_path, monkeypatch):
    scenario = {
        "A": {
            "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A", file="a.txt")], scenario)
    final = _approve_and_run(graph, run_dir)

    integration = cfg.target_repo / cfg.worktree_root / "orch-integration"
    assert len(git(integration, "log", "--oneline", "--merges").stdout.splitlines()) == 1

    calls = {"n": 0}
    real_merge = integrate_nodes.merge_no_ff

    def counting_merge(worktree, branch):
        calls["n"] += 1
        return real_merge(worktree, branch)

    monkeypatch.setattr(integrate_nodes, "merge_no_ff", counting_merge)

    # Re-enter exactly as a crash-resume would: the checkpoint hands A back APPROVED.
    state = {
        **final,
        "task_state": {
            **final["task_state"],
            "A": {**final["task_state"]["A"], "status": TStatus.APPROVED},
        },
    }
    first = integrate_nodes.integrate(state, {"task_id": "A"}, cfg=cfg, adapter=adapter, run_dir=run_dir)
    second = integrate_nodes.integrate(state, {"task_id": "A"}, cfg=cfg, adapter=adapter, run_dir=run_dir)

    assert first["task_state"]["A"]["status"] == TStatus.INTEGRATED
    assert second == first, "the second integration must be a no-op"
    assert calls["n"] == 0, "an already-merged branch must never be merged again"
    assert len(git(integration, "log", "--oneline", "--merges").stdout.splitlines()) == 1


# --- 5. parallel wave with ownership deferral --------------------------------


def test_a_task_sharing_a_file_is_deferred_until_its_owner_integrates(tmp_path, monkeypatch):
    scenario = {
        "A": {
            "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "B": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "C": {
            "implement": [{"edits": [{"file": "a.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    cfg, run_dir, graph, _adapter = _boot(
        tmp_path,
        [_task("A", file="a.txt"), _task("B", file="b.txt"), _task("C", file="a.txt")],
        scenario,
    )

    import orch.graph as graph_mod

    real_dispatch = graph_mod.compute_dispatch
    waves: list[tuple[list[str], str]] = []

    def recording(**kwargs):
        decision = real_dispatch(**kwargs)
        waves.append((list(decision.dispatch), decision.state))
        return decision

    monkeypatch.setattr(graph_mod, "compute_dispatch", recording)

    final = _approve_and_run(graph, run_dir)

    first_wave = next(dispatch for dispatch, _state in waves if dispatch)
    assert first_wave == ["A", "B"], f"C must be deferred out of wave 1, got {first_wave}"
    for tid in ("A", "B", "C"):
        assert final["task_state"][tid]["status"] == TStatus.INTEGRATED, tid
    assert final.get("escalation") is None

    # C's branch was cut from an integration tip that already held A's merge.
    c_worktree = cfg.target_repo / cfg.worktree_root / "C"
    assert is_ancestor(c_worktree, "agent/A"), "C branched before A integrated"
