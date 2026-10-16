import json, os, time
from pathlib import Path

import pytest

from orch.adapters.opencode import OpenCodeAdapter, parse_jsonl
from orch.errors import ModelUnavailableError

REAL_LINES = '''{"type":"step_start","timestamp":1,"sessionID":"ses_probe123","part":{"id":"p1","sessionID":"ses_probe123","type":"step-start"}}
{"type":"text","timestamp":2,"sessionID":"ses_probe123","part":{"id":"p2","sessionID":"ses_probe123","type":"text","text":"ok"}}
'''

def test_parse_jsonl_extracts_session_and_text():
    lines, sid, text = parse_jsonl(REAL_LINES)
    assert sid == "ses_probe123" and text == "ok" and len(lines) == 2

def test_parse_jsonl_tolerates_noise():
    _, sid, text = parse_jsonl("warning: something\n" + REAL_LINES + "not json\n")
    assert sid == "ses_probe123" and text == "ok"

# shim-based, no real model calls. Helper builds a fake "opencode" in a temp bin/:
def _shim_bin(tmp_path: Path, mode: str) -> str:
    bin_dir = tmp_path / "bin"; bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "opencode"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        "args = sys.argv[1:]\n"
        "open(os.environ['SHIM_ARGS_FILE'], 'a').write(json.dumps(args) + '\\n')\n"
        "open(os.environ['SHIM_PWD_FILE'], 'w').write(os.environ.get('PWD', ''))\n"
        "mode = os.environ.get('SHIM_MODE', 'ok')\n"
        "if mode == 'sleep':\n    time.sleep(30)\n"
        "if mode == 'model-error':\n    sys.stderr.write('model not found: nope/nope\\n'); sys.exit(1)\n"
        "if mode == 'stderr-unauthorized':\n"
        "    sys.stderr.write('unauthorized: bad api key\\n')\n"
        "if mode == 'stdout-401':\n"
        "    print(json.dumps({'type':'text','timestamp':1,'sessionID':'ses_shim',"
        "'part':{'type':'text','text':'HTTP 401 handling'}}))\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'type':'text','timestamp':1,'sessionID':'ses_shim','part':{'type':'text','text':'ok'}}))\n"
    )
    shim.chmod(0o755)
    return str(bin_dir)


def _run_with_shim(tmp_path, monkeypatch, mode, *, session_id=None, timeout_s=10, env=None):
    args_file = tmp_path / "args.jsonl"
    pwd_file = tmp_path / "pwd.txt"
    monkeypatch.setenv("PATH", _shim_bin(tmp_path, mode) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SHIM_ARGS_FILE", str(args_file))
    monkeypatch.setenv("SHIM_PWD_FILE", str(pwd_file))
    monkeypatch.setenv("SHIM_MODE", mode)
    adapter = OpenCodeAdapter(model="opencode-go/deepseek-v4.1-flash", timeout_s=timeout_s, extra_env={})
    r = adapter.run(role="implement", prompt="hello", cwd=tmp_path, session_id=session_id,
                    title="r/T1/implement", env=env or {})
    return r, [json.loads(l) for l in args_file.read_text().splitlines()], pwd_file.read_text()


def test_success_run_and_argv(tmp_path, monkeypatch):
    r, argv_calls, _ = _run_with_shim(tmp_path, monkeypatch, "ok")
    assert r.exit_code == 0 and r.session_id == "ses_shim" and r.text == "ok"
    argv = argv_calls[0]
    assert argv[:3] == ["run", "--format", "json"] and "--auto" in argv
    assert argv[argv.index("--model") + 1] == "opencode-go/deepseek-v4.1-flash"


def test_pwd_env_tracks_cwd_so_the_project_resolves_to_the_worktree(tmp_path, monkeypatch):
    # opencode resolves its project (where tools read/write) from $PWD, not the
    # process cwd; a stale inherited PWD would send a worktree run to the host
    # project. Force a wrong PWD and require the adapter to override it with cwd.
    monkeypatch.setenv("PWD", "/nonexistent/stale-pwd")
    _, _, pwd = _run_with_shim(tmp_path, monkeypatch, "ok")
    assert pwd == str(tmp_path)


def test_caller_env_pwd_wins_over_the_derived_one(tmp_path, monkeypatch):
    # Precedence pin: an explicit caller-supplied PWD is honored.
    _, _, pwd = _run_with_shim(tmp_path, monkeypatch, "ok", env={"PWD": "/explicit/pwd"})
    assert pwd == "/explicit/pwd"


def test_session_flag_passed(tmp_path, monkeypatch):
    _, argv_calls, _ = _run_with_shim(tmp_path, monkeypatch, "ok", session_id="ses_x")
    assert argv_calls[0][argv_calls[0].index("--session") + 1] == "ses_x"


def test_model_unavailable_raises(tmp_path, monkeypatch):
    with pytest.raises(ModelUnavailableError):
        _run_with_shim(tmp_path, monkeypatch, "model-error")


def test_stdout_401_text_is_not_a_model_error(tmp_path, monkeypatch):
    # A numeric "401" in the agent's own transcript (stdout) is prose, not a
    # provider failure: the run must succeed normally.
    r, _, _ = _run_with_shim(tmp_path, monkeypatch, "stdout-401")
    assert r.exit_code == 0
    assert "HTTP 401 handling" in r.text


def test_stderr_unauthorized_is_a_model_error(tmp_path, monkeypatch):
    # The same word signature on stderr is the provider/auth failure it looks like.
    with pytest.raises(ModelUnavailableError):
        _run_with_shim(tmp_path, monkeypatch, "stderr-unauthorized")


def test_timeout_kills_process_group(tmp_path, monkeypatch):
    t0 = time.monotonic()
    r, _, _ = _run_with_shim(tmp_path, monkeypatch, "sleep", timeout_s=1)
    assert r.timed_out is True and time.monotonic() - t0 < 5
