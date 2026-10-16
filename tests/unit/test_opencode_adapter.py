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
        "mode = os.environ.get('SHIM_MODE', 'ok')\n"
        "if mode == 'sleep':\n    time.sleep(30)\n"
        "if mode == 'model-error':\n    sys.stderr.write('model not found: nope/nope\\n'); sys.exit(1)\n"
        "print(json.dumps({'type':'text','timestamp':1,'sessionID':'ses_shim','part':{'type':'text','text':'ok'}}))\n"
    )
    shim.chmod(0o755)
    return str(bin_dir)


def _run_with_shim(tmp_path, monkeypatch, mode, *, session_id=None, timeout_s=10):
    args_file = tmp_path / "args.jsonl"
    monkeypatch.setenv("PATH", _shim_bin(tmp_path, mode) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SHIM_ARGS_FILE", str(args_file))
    monkeypatch.setenv("SHIM_MODE", mode)
    adapter = OpenCodeAdapter(model="opencode-go/deepseek-v4.1-flash", timeout_s=timeout_s, extra_env={})
    r = adapter.run(role="implement", prompt="hello", cwd=tmp_path, session_id=session_id,
                    title="r/T1/implement", env={})
    return r, [json.loads(l) for l in args_file.read_text().splitlines()]


def test_success_run_and_argv(tmp_path, monkeypatch):
    r, argv_calls = _run_with_shim(tmp_path, monkeypatch, "ok")
    assert r.exit_code == 0 and r.session_id == "ses_shim" and r.text == "ok"
    argv = argv_calls[0]
    assert argv[:3] == ["run", "--format", "json"] and "--auto" in argv
    assert argv[argv.index("--model") + 1] == "opencode-go/deepseek-v4.1-flash"


def test_session_flag_passed(tmp_path, monkeypatch):
    _, argv_calls = _run_with_shim(tmp_path, monkeypatch, "ok", session_id="ses_x")
    assert argv_calls[0][argv_calls[0].index("--session") + 1] == "ses_x"


def test_model_unavailable_raises(tmp_path, monkeypatch):
    with pytest.raises(ModelUnavailableError):
        _run_with_shim(tmp_path, monkeypatch, "model-error")


def test_timeout_kills_process_group(tmp_path, monkeypatch):
    t0 = time.monotonic()
    r, _ = _run_with_shim(tmp_path, monkeypatch, "sleep", timeout_s=1)
    assert r.timed_out is True and time.monotonic() - t0 < 5
