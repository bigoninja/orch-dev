import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from orch.adapters.base import AgentAdapter, AgentRunResult
from orch.adapters.mock import MockAdapter, MockScenarioError, MockScenarioExhausted

SCEN = {"T1": {
  "implement": [
    {"edits": [{"file": "app.txt", "content": "OK one"}]},                 # uncommitted
    {"edits": [{"file": "app.txt", "content": "OK two"}], "commit": True},
    {"blocked": "cannot do it"},
  ],
  "review": [{"verdict": "reject", "findings": ["line 2 wrong"]},
             {"verdict": "approve"}],
}}

def _adapter(tmp_path, scen=SCEN):
    f = tmp_path / "scen.yaml"; f.write_text(yaml.safe_dump(scen)); return MockAdapter(f)

def test_implement_writes_edits(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    a = _adapter(tmp_path)
    r = a.run(role="implement", prompt="p", cwd=wd, session_id=None, title="t", env={})
    assert (wd / "app.txt").read_text() == "OK one"
    assert r.exit_code == 0 and r.session_id.startswith("ses_mock_T1")

def _init_git(wd: Path) -> None:
    subprocess.run(["git", "init", "-b", "main", str(wd)], check=True, capture_output=True)
    (wd / "app.txt").write_text("v0")
    subprocess.run(["git", "-C", str(wd), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(wd), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-m", "init"], check=True, capture_output=True)


def test_commit_flag_creates_commit(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir(); _init_git(wd)
    a = _adapter(tmp_path)
    r1 = a.run(role="implement", prompt="p", cwd=wd, session_id=None, title="r/T1/implement", env={})
    assert (wd / "app.txt").read_text() == "OK one"
    assert subprocess.run(["git", "-C", str(wd), "status", "--porcelain"],
                          capture_output=True, text=True).stdout.strip()  # attempt 1: dirty
    a.run(role="implement", prompt="p", cwd=wd, session_id=r1.session_id, title="r/T1/fix", env={})
    assert (wd / "app.txt").read_text() == "OK two"
    assert subprocess.run(["git", "-C", str(wd), "status", "--porcelain"],
                          capture_output=True, text=True).stdout.strip() == ""  # attempt 2: committed by mock


def test_blocked_and_review_sequence(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir(); _init_git(wd)
    a = _adapter(tmp_path)
    a.run(role="implement", prompt="p", cwd=wd, session_id=None, title="r/T1/implement", env={})
    a.run(role="implement", prompt="p", cwd=wd, session_id=None, title="r/T1/implement", env={})
    r3 = a.run(role="implement", prompt="p", cwd=wd, session_id=None, title="r/T1/implement", env={})
    assert r3.text.startswith("BLOCKED:")
    v1 = a.run(role="review", prompt="p", cwd=wd, session_id=None, title="r/T1/review", env={})
    v2 = a.run(role="review", prompt="p", cwd=wd, session_id=None, title="r/T1/review", env={})
    assert json.loads(v1.text)["verdict"] == "reject"
    assert json.loads(v2.text)["verdict"] == "approve"


# --- cross-task contract: the mock must satisfy the AgentAdapter boundary and
# --- the reviewer verdict contract in the spec, §8.


def test_mock_conforms_to_adapter_protocol(tmp_path):
    a = _adapter(tmp_path)
    assert isinstance(a, AgentAdapter)


def test_agent_run_result_is_frozen():
    r = AgentRunResult(session_id=None, exit_code=0, stdout="", stderr="",
                       timed_out=False, text="", raw_json=[])
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.exit_code = 1


def test_provided_session_id_is_echoed(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    a = _adapter(tmp_path)
    r = a.run(role="implement", prompt="p", cwd=wd, session_id="ses_custom",
              title="r/T1/implement", env={})
    assert r.session_id == "ses_custom"


def test_review_text_matches_verdict_contract(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    a = _adapter(tmp_path)
    v = a.run(role="review", prompt="p", cwd=wd, session_id=None, title="r/T1/review", env={})
    obj = json.loads(v.text)
    assert set(obj) == {"verdict", "risky", "findings", "summary"}
    assert obj["verdict"] == "reject"
    assert obj["risky"] is False
    assert obj["findings"] == [{"severity": "major", "file": "", "note": "line 2 wrong"}]
    assert isinstance(obj["summary"], str)


def test_review_risky_passthrough(tmp_path):
    scen = {"T2": {"review": [{"verdict": "approve", "risky": True}]}}
    wd = tmp_path / "wd"; wd.mkdir()
    a = _adapter(tmp_path, scen)
    obj = json.loads(a.run(role="review", prompt="p", cwd=wd, session_id=None,
                           title="r/T2/review", env={}).text)
    assert obj["verdict"] == "approve" and obj["risky"] is True


def test_missing_task_scenario_fails_loud(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    a = _adapter(tmp_path)
    with pytest.raises(MockScenarioError):
        a.run(role="implement", prompt="p", cwd=wd, session_id=None,
              title="r/T9/implement", env={})


def test_exhausted_scenario_fails_loud(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir(); _init_git(wd)
    a = _adapter(tmp_path)
    for _ in range(3):  # the scenario has exactly three implement attempts
        a.run(role="implement", prompt="p", cwd=wd, session_id=None,
              title="r/T1/implement", env={})
    with pytest.raises(MockScenarioExhausted):
        a.run(role="implement", prompt="p", cwd=wd, session_id=None,
              title="r/T1/implement", env={})
