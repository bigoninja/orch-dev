"""End-to-end governance tests: risky review, plan-hash guard, destructive scan.

Everything runs through the same in-process CLI contract as ``test_cli_e2e.py``
(one process, one command, one exit code) and reuses its helpers: a throwaway demo
repo, a throwaway config, and the token-free mock adapter selected through
``ORCH_ADAPTER``/``ORCH_MOCK_SCENARIO``. The pause mechanics are identical — the
first invoke stops at the approval gate, ``approve`` resumes with the pinned
approval payload, an escalation stops on a checkpointed thread that ``resume``
re-enters.

Four pinned governance rules are exercised:

* a reviewer's ``risky: true`` pauses unless the task was pre-authorized at the
  gate (``approve --accept-risky``);
* a ``ModelUnavailableError`` surfaces as ``model_unavailable`` and is *not*
  substituted — a fixed cause is retried on the same model;
* ``plan.yaml`` is immutable after approval: a plain ``resume`` refuses once the
  checkpointed hash no longer matches the file;
* a destructive command in the implement log escalates at once, without spending a
  fix attempt; and a dirty *tracked* tree refuses the run before any setup.
"""

from orch.cli import main

from conftest import git
from test_cli_e2e import (
    SCENARIO,
    TASK,
    _boot,
    _scenario,
    _state,
)

# --- (a) risky review ----------------------------------------------------------

#: ``files_allowed`` matches the default ``**/gateway/**`` risky pattern, so the plan
#: flags this task risky before it is dispatched. The path is deliberately *nested in
#: a directory the repo does not have yet*: ``src/gateway/`` is wholly new, so the
#: ``files_allowed`` boundary check only passes if ``git status --porcelain`` lists the
#: file rather than the collapsed directory (see ``gitops._porcelain_lines``).
RISKY_TASK = {
    "id": "R",
    "title": "write the gateway token helper",
    "description": "Write the gateway token helper.",
    "depends_on": [],
    "files_allowed": ["src/gateway/token.py"],
    "acceptance_criteria": ["src/gateway/token.py contains OK"],
    "checks": {"test": {"cmd": "grep -q OK src/gateway/token.py"}},
}

#: The reviewer approves, but flags the change risky — a runtime risk surprise.
RISKY_SCENARIO = {
    "R": {
        "implement": [{"edits": [{"file": "src/gateway/token.py", "content": "OK"}]}],
        "review": [{"verdict": "approve", "risky": True}],
    }
}

#: The pinned model is unavailable on the first attempt; the run must not substitute.
MODEL_ERROR_SCENARIO = {
    "A": {
        "implement": [{"raise_model_error": True}],
        "review": [{"verdict": "approve"}],
    }
}

#: Three failing implements spend the fix budget (``retries: 2``) and pause.
EXHAUSTED_SCENARIO = {
    "A": {
        "implement": [
            {"edits": [{"file": "a.txt", "content": "nope 1"}]},
            {"edits": [{"file": "a.txt", "content": "nope 2"}]},
            {"edits": [{"file": "a.txt", "content": "nope 3"}]},
        ],
        "review": [{"verdict": "approve"}],
    }
}

#: The implement log names a destructive command — a hard stop, not a fixable fault.
DESTRUCTIVE_SCENARIO = {
    "A": {
        "implement": [
            {
                "edits": [{"file": "a.txt", "content": "OK"}],
                "text": "about to run: git push origin main",
            },
            # A spare attempt: an unintended fix route would consume it and fail the
            # scenario loudly instead of silently passing the assertions below.
            {"edits": [{"file": "a.txt", "content": "OK"}]},
        ],
        "review": [{"verdict": "approve"}],
    }
}


def test_risky_review_pauses_without_pre_authorization(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(
        tmp_path, demo_repo, monkeypatch, RISKY_SCENARIO, tasks=[RISKY_TASK]
    )
    assert main(["run", "--config", str(cfg_path)]) == 0
    assert "awaiting approval" in capsys.readouterr().out

    rc = main(["approve", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 2, out
    assert "risk_reviewer" in out
    state = _state(run_dir)
    assert state["escalation"]["kind"] == "risk_reviewer"
    assert state["escalation"]["task_id"] == "R"
    # The verdict is stored even though it paused the run.
    assert state["task_state"]["R"]["verdicts"][-1]["risky"] is True
    assert not state.get("summary"), "the run must not have finished"


def test_pre_authorized_risky_review_proceeds_to_integration(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(
        tmp_path, demo_repo, monkeypatch, RISKY_SCENARIO, tasks=[RISKY_TASK]
    )
    assert main(["run", "--config", str(cfg_path)]) == 0
    capsys.readouterr()

    rc = main(["approve", "--accept-risky", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    state = _state(run_dir)
    assert state["task_state"]["R"]["status"] == "integrated"
    assert state["escalation"] is None


def test_model_unavailable_escalates_then_a_clean_rerun_completes(
    tmp_path, demo_repo, monkeypatch, capsys
):
    """No substitution: the failure surfaces, and the fixed cause resumes cleanly."""
    cfg_path, run_dir = _boot(
        tmp_path, demo_repo, monkeypatch, MODEL_ERROR_SCENARIO, tasks=[TASK]
    )

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 2, out
    assert "model_unavailable" in out
    state = _state(run_dir)
    assert state["escalation"]["kind"] == "model_unavailable"
    assert state["escalation"]["task_id"] == "A"
    assert "model unavailable" in state["escalation"]["message"]

    # The cause is fixed (the pinned model is back): the same run resumes and
    # completes. The model string itself is pinned by tests/unit/test_factory.py.
    _scenario(tmp_path, SCENARIO)

    rc = main(["resume", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    assert _state(run_dir)["task_state"]["A"]["status"] == "integrated"


# --- (b) plan-hash guard -------------------------------------------------------


def test_resume_refuses_after_plan_yaml_changed_until_it_is_restored(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(
        tmp_path, demo_repo, monkeypatch, EXHAUSTED_SCENARIO, tasks=[TASK]
    )
    assert main(["run", "--yes", "--config", str(cfg_path)]) == 2
    capsys.readouterr()
    approved_plan = run_dir.plan_path.read_text(encoding="utf-8")

    # The plan is edited *after* approval, while the run sits on an escalation.
    run_dir.plan_path.write_text(approved_plan + "# edited after approval\n", encoding="utf-8")

    rc = main(["resume", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 1, out
    assert "plan.yaml changed after approval" in out
    assert "orch approve --edit" in out
    # Nothing was invoked: the run is still parked on its original escalation.
    state = _state(run_dir)
    assert state["escalation"]["kind"] == "task_retries_exhausted"
    assert state["task_state"]["A"]["status"] == "escalated"

    # Restore the file: the pinned hash matches again and resume proceeds.
    run_dir.plan_path.write_text(approved_plan, encoding="utf-8")
    _scenario(tmp_path, SCENARIO)

    rc = main(["resume", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    assert _state(run_dir)["task_state"]["A"]["status"] == "integrated"


def test_approve_edit_sends_the_current_files_hash(
    tmp_path, demo_repo, monkeypatch, capsys
):
    """``approve --edit`` is the re-approval path: it hashes the edited file."""
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO, tasks=[TASK])
    assert main(["run", "--config", str(cfg_path)]) == 0
    capsys.readouterr()

    # The editor appends a comment — a byte change, so a stale hash would refuse.
    editor = tmp_path / "editor.sh"
    editor.write_text("#!/bin/sh\nprintf '# re-approved\\n' >> \"$1\"\n", encoding="utf-8")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    rc = main(["approve", "--edit", "--run", "R1", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    assert "# re-approved" in run_dir.plan_path.read_text(encoding="utf-8")


# --- (c) destructive scan ------------------------------------------------------


def test_destructive_command_escalates_without_spending_a_fix(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(
        tmp_path, demo_repo, monkeypatch, DESTRUCTIVE_SCENARIO, tasks=[TASK]
    )

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 2, out
    assert "destructive_command" in out
    state = _state(run_dir)
    assert state["escalation"]["kind"] == "destructive_command"
    assert state["escalation"]["task_id"] == "A"
    # The message is the matched deny-list patterns, joined.
    assert state["escalation"]["message"] == r"git\s+push\b"
    # Immediate: one implement attempt, no fix attempt entered the budget.
    assert state["task_state"]["A"]["attempts"] == 1
    assert state["task_state"]["A"]["fix_attempts"] == 0
    assert not state.get("summary"), "the run must not have finished"


# --- (d) dirty tree ------------------------------------------------------------


def test_dirty_tracked_file_refuses_the_run_and_names_the_path(
    tmp_path, demo_repo, monkeypatch, capsys
):
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO, tasks=[TASK])
    (demo_repo / "a.txt").write_text("uncommitted edit", encoding="utf-8")

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 2, out
    assert "dirty_tree" in out
    assert "a.txt" in out
    assert _state(run_dir)["escalation"]["kind"] == "dirty_tree"
    # Refused before any repo state was created: no worktree, no integration branch.
    assert not (demo_repo / ".worktrees").exists()
    assert git(demo_repo, "branch", "--list", "orch/integration").stdout == ""


def test_untracked_files_do_not_block_the_run(
    tmp_path, demo_repo, monkeypatch, capsys
):
    """Only *tracked* changes are dirty: an untracked stray file is allowed."""
    cfg_path, run_dir = _boot(tmp_path, demo_repo, monkeypatch, SCENARIO, tasks=[TASK])
    (demo_repo / "stray.txt").write_text("untracked", encoding="utf-8")
    assert git(demo_repo, "status", "--porcelain").stdout.startswith("??")

    rc = main(["run", "--yes", "--config", str(cfg_path)])

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "run R1: 1/1 integrated" in out
    assert _state(run_dir)["task_state"]["A"]["status"] == "integrated"
