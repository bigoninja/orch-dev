"""End-to-end failure paths: fix loop, blocked/violation recovery, escalation + resume.

Each test drives the real compiled graph against a throwaway repo, exactly as the
CLI will: the first invoke stops at the approval gate, the harness resumes with the
human's approval, and any later Escalation pauses the run on a checkpointed thread.

The pinned failure contract under test:

* a fixable failure — BLOCKED, a ``files_allowed`` violation, red checks, or a
  review reject — routes to ``t_fix`` while ``fix_attempts < cfg.retries``, and only
  then escalates ``task_retries_exhausted``;
* a fix reuses the implementer's session (one continuous worker session per task);
* a violation is never committed, however it is discovered;
* a paused run resumes from its checkpoint: a retry resets the task to ``PENDING``
  with ``human_retry`` set, and a crash mid-pipeline neither loses nor doubles work.
"""

from pathlib import Path

import pytest
import yaml
from langgraph.types import Command

from orch.adapters import MockAdapter
from orch.artifacts import RunDir
from orch.errors import PlanValidationError
from orch.graph import build_graph
from orch.scheduler import DispatchDecision
from orch.state import TStatus

from conftest import git, make_config, make_demo_repo, plan_sha256, run_once, write_plan

#: The checkpointed thread every invoke for one scenario shares.
THREAD = "failures"


class RecordingMock(MockAdapter):
    """A ``MockAdapter`` that records every ``run`` call (role/session/title/prompt).

    The fix loop's session reuse and the findings fed to a fixer are observable only
    from the calls the adapter receives, so the tests capture them here.
    """

    def __init__(self, scenario_path: Path):
        super().__init__(scenario_path)
        self.calls: list[dict] = []

    def run(self, **kwargs):  # type: ignore[override]
        result = super().run(**kwargs)
        self.calls.append({**kwargs, "result_session": result.session_id})
        return result


def _task(tid: str, *, file: str = "app.txt", depends_on: tuple[str, ...] = ()) -> dict:
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


def _approval(run_dir: RunDir) -> dict:
    return {"approved": True, "pre_authorized": [], "plan_sha256": plan_sha256(run_dir)}


def _approve_and_run(graph, run_dir: RunDir, *, rid="R1", thread=THREAD):
    """First invoke to the approval gate, then resume with the plan approval."""
    first = run_once(graph, rid, thread=thread)
    assert first.get("__interrupt__"), "the approval gate must interrupt the first invoke"
    return graph.invoke(
        Command(resume=_approval(run_dir)),
        config={"configurable": {"thread_id": thread}},
    )


def _resume(graph, value, *, thread=THREAD):
    return graph.invoke(Command(resume=value), config={"configurable": {"thread_id": thread}})


def _calls(adapter: RecordingMock, role: str) -> list[dict]:
    return [c for c in adapter.calls if c["role"] == role]


# --- 1. reject → fix → approve ------------------------------------------------


def test_reject_routes_to_a_fix_that_reuses_the_implementer_session(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"edits": [{"file": "app.txt", "content": "OK one"}]},
                {"edits": [{"file": "app.txt", "content": "OK two"}]},
            ],
            "review": [
                {"verdict": "reject", "findings": ["line 2 wrong"]},
                {"verdict": "approve"},
            ],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    final = _approve_and_run(graph, run_dir)

    state = final["task_state"]["A"]
    assert state["status"] == TStatus.INTEGRATED, state["status"]
    assert state["fix_attempts"] == 1

    fixes = _calls(adapter, "fix")
    assert len(fixes) == 1, "exactly one fix attempt must run"
    # The fix is the same worker session, not a fresh one (design §5.4 step 6).
    assert fixes[0]["session_id"] == "ses_mock_A_1"
    assert fixes[0]["title"] == "R1/A/fix1"
    assert "line 2 wrong" in fixes[0]["prompt"], "the reviewer findings must reach the fixer"
    assert final.get("escalation") is None


# --- 2. red checks → fix → green ----------------------------------------------


def test_red_checks_route_to_a_fix_and_turn_green(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"edits": [{"file": "app.txt", "content": "nope"}]},
                {"edits": [{"file": "app.txt", "content": "OK"}]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    final = _approve_and_run(graph, run_dir)

    state = final["task_state"]["A"]
    assert state["status"] == TStatus.INTEGRATED, state["status"]
    assert state["fix_attempts"] == 1

    fixes = _calls(adapter, "fix")
    assert len(fixes) == 1
    assert "check failed: test" in fixes[0]["prompt"]


# --- 3. retries exhausted → pause → human retry -------------------------------


def test_retries_exhausted_pauses_then_a_human_retry_completes(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"edits": [{"file": "app.txt", "content": "nope 1"}]},
                {"edits": [{"file": "app.txt", "content": "nope 2"}]},
                {"edits": [{"file": "app.txt", "content": "nope 3"}]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    paused = _approve_and_run(graph, run_dir)

    assert paused.get("__interrupt__"), "exhausted retries must pause the run"
    assert paused["escalation"]["kind"] == "task_retries_exhausted"
    assert paused["escalation"]["task_id"] == "A"
    assert paused["task_state"]["A"]["status"] == TStatus.ESCALATED
    assert paused["task_state"]["A"]["fix_attempts"] == 2

    # The human fixes the environment and retries the task: repair the scenario so
    # the next implement attempt passes.
    adapter.scenario["A"]["implement"].append(
        {"edits": [{"file": "app.txt", "content": "OK"}]}
    )
    final = _resume(graph, {"action": "retry", "task_id": "A"})

    state = final["task_state"]["A"]
    assert state["status"] == TStatus.INTEGRATED, state["status"]
    assert state["attempts"] == 1, "the retry resets the attempt budget"
    assert state["fix_attempts"] == 0
    assert state["human_retry"] is True
    assert final.get("escalation") is None
    assert final["summary"] == "run R1: 1/1 integrated"


def test_a_retry_with_a_bogus_task_id_parks_the_real_task(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"edits": [{"file": "app.txt", "content": "nope 1"}]},
                {"edits": [{"file": "app.txt", "content": "nope 2"}]},
                {"edits": [{"file": "app.txt", "content": "nope 3"}]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    _cfg, run_dir, graph, _adapter = _boot(tmp_path, [_task("A")], scenario)
    paused = _approve_and_run(graph, run_dir)
    assert paused["escalation"]["kind"] == "task_retries_exhausted"

    # The resume value is untrusted: an id the plan never heard of must not be
    # dispatched (which would raise ``KeyError`` in ``TaskDef``) — the task that
    # actually escalated is parked instead.
    final = _resume(graph, {"action": "retry", "task_id": "NOPE"})

    assert final["task_state"]["A"]["status"] == TStatus.ESCALATED
    assert final["escalation"]["kind"] == "task_retries_exhausted"
    assert final["escalation"]["task_id"] == "A"


def test_a_retry_naming_a_different_task_leaves_it_untouched(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"edits": [{"file": "app.txt", "content": "nope 1"}]},
                {"edits": [{"file": "app.txt", "content": "nope 2"}]},
                {"edits": [{"file": "app.txt", "content": "nope 3"}]},
            ],
            "review": [{"verdict": "approve"}],
        },
        "B": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    # ``B`` depends on ``A``, so it stays PENDING while ``A`` exhausts its retries.
    _cfg, run_dir, graph, _adapter = _boot(
        tmp_path, [_task("A"), _task("B", file="b.txt", depends_on=("A",))], scenario
    )
    paused = _approve_and_run(graph, run_dir)
    assert paused["escalation"]["task_id"] == "A"
    assert paused["task_state"]["B"]["status"] == TStatus.PENDING

    # A retry that names a *different* (but valid) task must not reset or dispatch it;
    # the task that escalated is parked.
    final = _resume(graph, {"action": "retry", "task_id": "B"})

    assert final["task_state"]["A"]["status"] == TStatus.ESCALATED
    state_b = final["task_state"]["B"]
    assert state_b["status"] == TStatus.PENDING, "B must not be reset or dispatched"
    assert state_b["attempts"] == 0
    assert final["escalation"]["kind"] == "task_retries_exhausted"


# --- 4. blocked → fix recovery ------------------------------------------------


def test_blocked_implement_recovers_via_a_fix_that_consumes_last_error(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"blocked": "cannot do it"},
                {"edits": [{"file": "app.txt", "content": "OK"}]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    final = _approve_and_run(graph, run_dir)

    state = final["task_state"]["A"]
    assert state["status"] == TStatus.INTEGRATED, state["status"]
    assert state["fix_attempts"] == 1

    fixes = _calls(adapter, "fix")
    assert len(fixes) == 1
    assert "BLOCKED: cannot do it" in fixes[0]["prompt"]
    # ``last_error`` is a one-shot marker: it survives until the fix consumes it, and
    # never persists past that fix into the integrated record.
    assert state["last_error"] is None


def test_a_marker_consumed_by_a_fix_never_shadows_a_later_review_reject(tmp_path):
    scenario = {
        "A": {
            "implement": [
                {"blocked": "cannot do it"},
                {"edits": [{"file": "app.txt", "content": "OK"}]},
                {"edits": [{"file": "app.txt", "content": "OK two"}]},
            ],
            "review": [
                {"verdict": "reject", "findings": ["needs a comment"]},
                {"verdict": "approve"},
            ],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    final = _approve_and_run(graph, run_dir)

    fixes = _calls(adapter, "fix")
    assert len(fixes) == 2, "BLOCKED and then the reject each need one fix"
    assert "BLOCKED: cannot do it" in fixes[0]["prompt"]
    # The reject's fix must see the verdict's finding — not the BLOCKED text the first
    # fix already consumed.
    assert "needs a comment" in fixes[1]["prompt"]
    assert "BLOCKED" not in fixes[1]["prompt"]
    assert final["task_state"]["A"]["status"] == TStatus.INTEGRATED


# --- 5. violation: stray edit is never committed ------------------------------


def test_a_violation_routes_to_a_fix_and_is_never_committed(tmp_path):
    # ``stray.txt`` is outside the task's ``files_allowed``: attempt 1 creates it; the
    # fix removes it, restoring a clean boundary.
    scenario = {
        "A": {
            "implement": [
                {
                    "edits": [
                        {"file": "app.txt", "content": "OK"},
                        {"file": "stray.txt", "content": "stray"},
                    ]
                },
                {"edits": [{"file": "app.txt", "content": "OK"}], "delete": ["stray.txt"]},
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    final = _approve_and_run(graph, run_dir)

    state = final["task_state"]["A"]
    assert state["status"] == TStatus.INTEGRATED, state["status"]
    assert state["fix_attempts"] == 1

    fixes = _calls(adapter, "fix")
    assert len(fixes) == 1
    assert "edited disallowed path: stray.txt" in fixes[0]["prompt"], (
        "the violation must be recorded and classified into the fix findings"
    )

    # Nothing on the task branch may contain the stray edit, at any commit.
    touched = git(
        cfg.target_repo, "diff", "--name-only", f"{cfg.base_ref}..agent/A"
    ).stdout.splitlines()
    assert touched == ["app.txt"], f"agent/A touched {touched}"
    for sha in git(cfg.target_repo, "log", "--format=%H", "agent/A").stdout.splitlines():
        names = git(cfg.target_repo, "show", "--name-only", "--format=", sha).stdout
        assert "stray.txt" not in names, f"commit {sha} contains the stray path"


def test_a_violation_the_fix_does_not_clean_up_escalates_without_committing(tmp_path):
    stray = {
        "edits": [
            {"file": "app.txt", "content": "OK"},
            {"file": "stray.txt", "content": "stray"},
        ]
    }
    scenario = {
        "A": {
            "implement": [dict(stray), dict(stray), dict(stray)],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    paused = _approve_and_run(graph, run_dir)

    assert paused.get("__interrupt__")
    assert paused["escalation"]["kind"] == "task_retries_exhausted"
    state = paused["task_state"]["A"]
    assert state["status"] == TStatus.ESCALATED
    assert state["violation"] == ["stray.txt"]
    assert state["fix_attempts"] == 2

    touched = git(
        cfg.target_repo, "diff", "--name-only", f"{cfg.base_ref}..agent/A"
    ).stdout.splitlines()
    assert touched == [], f"a violating task must never be committed, touched {touched}"


# --- 6. deadlock --------------------------------------------------------------


def test_a_cyclic_plan_is_rejected_before_any_repo_state(tmp_path):
    repo = make_demo_repo(tmp_path / "demo")
    cfg = make_config(repo, tmp_path)
    run_dir = RunDir.create(Path(cfg.runs_dir), "R1")
    write_plan(run_dir, [_task("A", depends_on=("B",)), _task("B", depends_on=("A",))], rid="R1")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("{}\n", encoding="utf-8")
    graph = build_graph(cfg=cfg, adapter=MockAdapter(scenario_path), run_dir=run_dir)

    with pytest.raises(PlanValidationError):
        run_once(graph, "R1", thread=THREAD)

    assert git(cfg.target_repo, "branch", "--list", cfg.integration_branch).stdout.strip() == ""
    assert not (cfg.target_repo / cfg.worktree_root / "orch-integration").exists()


def test_deadlock_pauses_and_a_retry_re_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orch.graph.compute_dispatch", lambda **kw: DispatchDecision([], "deadlock")
    )
    scenario = {
        "A": {
            "implement": [{"edits": [{"file": "app.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        }
    }
    _cfg, run_dir, graph, _adapter = _boot(tmp_path, [_task("A")], scenario)

    paused = _approve_and_run(graph, run_dir)

    assert paused.get("__interrupt__")
    assert paused["escalation"]["kind"] == "deadlock"
    assert paused["escalation"]["task_id"] is None
    assert paused["task_state"]["A"]["status"] == TStatus.PENDING

    # ``{"action": "retry"}`` with no task and a deadlock clears the escalation and
    # re-enters the scheduler, which deadlocks again — a second, non-crashing pause.
    again = _resume(graph, {"action": "retry"})

    assert again.get("__interrupt__"), "the deadlock must re-pause, not spin or crash"
    assert again["escalation"]["kind"] == "deadlock"
    assert again["task_state"]["A"]["status"] == TStatus.PENDING


# --- 7. crash-resume: no lost work, no double merge ---------------------------


def test_a_crash_after_the_first_integrate_resumes_without_double_merging(
    tmp_path, monkeypatch
):
    scenario = {
        "A": {
            "implement": [{"edits": [{"file": "app.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
        "B": {
            "implement": [{"edits": [{"file": "b.txt", "content": "OK"}]}],
            "review": [{"verdict": "approve"}],
        },
    }
    cfg, run_dir, graph, adapter = _boot(
        tmp_path,
        [_task("A"), _task("B", file="b.txt", depends_on=("A",))],
        scenario,
    )

    # The adapter call sequence is A/implement, A/review, then — after A integrates —
    # B/implement. Crashing on that third call is a crash *after the first integrate*.
    real_run = adapter.run
    calls = {"n": 0, "fired": False}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 3 and not calls["fired"]:
            calls["fired"] = True
            raise KeyboardInterrupt
        return real_run(**kwargs)

    monkeypatch.setattr(adapter, "run", flaky)

    first = run_once(graph, "R1", thread=THREAD)
    assert first.get("__interrupt__")
    with pytest.raises(KeyboardInterrupt):
        graph.invoke(
            Command(resume=_approval(run_dir)),
            config={"configurable": {"thread_id": THREAD}},
        )

    final = graph.invoke(None, config={"configurable": {"thread_id": THREAD}})

    for tid in ("A", "B"):
        assert final["task_state"][tid]["status"] == TStatus.INTEGRATED, (
            f"{tid} ended {final['task_state'][tid]['status']!r}"
        )

    integration = cfg.target_repo / cfg.worktree_root / "orch-integration"
    merges = git(integration, "log", "--oneline", "--merges").stdout.splitlines()
    assert len(merges) == 2, f"expected one merge per task, got {merges}"


def test_a_stray_edit_introduced_by_a_fix_is_caught_at_commit_time(tmp_path):
    """The live pilot's gap, pinned: a fix that makes the checks pass but ALSO strays
    outside ``files_allowed`` must be caught when the commit gate RE-DERIVES the
    boundary — the implement-time marker alone was consumed by the fix.

    Sequence: attempt 1 leaves the check RED (clean edits only) -> fix #1 turns the
    check GREEN but adds ``stray.txt`` -> t_commit refuses and routes one more fix ->
    fix #2 removes the stray -> the task integrates with no commit ever containing it.
    """
    scenario = {
        "A": {
            "implement": [
                {"edits": [{"file": "app.txt", "content": "NOPE"}]},  # check red, no violation
                {
                    "edits": [
                        {"file": "app.txt", "content": "OK"},
                        {"file": "stray.txt", "content": "stray"},
                    ]
                },  # fix #1: green checks, but strays
                {"delete": ["stray.txt"]},  # fix #2 cleans the boundary
            ],
            "review": [{"verdict": "approve"}],
        }
    }
    cfg, run_dir, graph, adapter = _boot(tmp_path, [_task("A")], scenario)

    final = _approve_and_run(graph, run_dir)

    state = final["task_state"]["A"]
    assert state["status"] == TStatus.INTEGRATED, state["status"]
    assert state["fix_attempts"] == 2

    fixes = _calls(adapter, "fix")
    assert len(fixes) == 2
    assert "edited disallowed path: stray.txt" in fixes[1]["prompt"], (
        "the commit gate must classify the fix-round stray into the next fix findings"
    )

    # Nothing on the task branch may contain the stray edit, at any commit.
    touched = git(
        cfg.target_repo, "diff", "--name-only", f"{cfg.base_ref}..agent/A"
    ).stdout.splitlines()
    assert touched == ["app.txt"], f"agent/A touched {touched}"
