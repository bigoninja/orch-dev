from pathlib import Path

from orch.checks import CheckResult, checks_green, run_checks
from orch.plan import Check


def _run(checks: dict[str, Check], tmp_path: Path, *, env=None, timeout_s=10):
    worktree = tmp_path / "wt"
    (worktree / "sub").mkdir(parents=True, exist_ok=True)
    evidence = tmp_path / "evidence"
    return run_checks(
        checks,
        worktree=worktree,
        evidence_dir=evidence,
        task_id="T1",
        env=env or {},
        timeout_s=timeout_s,
    )


def _log(result: CheckResult) -> str:
    return result.output_path.read_bytes().decode("utf-8", errors="replace")


def _stdout(result: CheckResult) -> str:
    """The command's own stdout only — never the header, which echoes cmd/cwd."""
    return _log(result).split("STDOUT\n", 1)[1].split("\nSTDERR\n", 1)[0]


def test_pass_and_fail_exit_codes(tmp_path):
    results = _run({"pass": Check(cmd="true"), "fail": Check(cmd="exit 3")}, tmp_path)
    by_name = {r.name: r for r in results}
    assert by_name["pass"].exit_code == 0
    assert by_name["fail"].exit_code == 3
    assert [r.name for r in results] == ["pass", "fail"]
    assert checks_green(results) is False
    assert checks_green([by_name["pass"]]) is True


def test_cwd_honored(tmp_path):
    (r,) = _run({"pwd": Check(cmd="pwd", cwd="sub")}, tmp_path)
    assert r.exit_code == 0
    assert r.cwd == "sub"
    assert _stdout(r).strip().endswith("sub")
    assert r.output_path.name == "T1.check.pwd.log"
    assert r.output_path.parent == tmp_path / "evidence"


def test_cwd_defaults_to_worktree_root(tmp_path):
    (r,) = _run({"pwd": Check(cmd="pwd")}, tmp_path)
    assert r.cwd is None
    assert _stdout(r).strip() == str(tmp_path / "wt")


def test_env_var_visible_to_command(tmp_path):
    (r,) = _run({"env": Check(cmd="echo $MYVAR-$MYVAR")}, tmp_path, env={"MYVAR": "seen-by-child"})
    assert r.exit_code == 0
    assert _stdout(r).strip() == "seen-by-child-seen-by-child"


def test_timeout_marks_minus_nine_and_kills_fast(tmp_path):
    (r,) = _run({"slow": Check(cmd="sleep 5")}, tmp_path, timeout_s=1)
    assert r.exit_code == -9
    assert r.duration_s < 3


def test_binary_output_captured_without_crash(tmp_path):
    (r,) = _run({"bin": Check(cmd=r"printf '\xff\xfe'")}, tmp_path)
    assert r.exit_code == 0
    data = r.output_path.read_bytes()
    assert b"\xff\xfe" in data
    data.decode("utf-8", errors="replace")  # evidence is readable back as text


def test_evidence_naming_and_order(tmp_path):
    checks = {f"c{i}": Check(cmd="true") for i in range(3)}
    results = _run(checks, tmp_path)
    assert [r.name for r in results] == ["c0", "c1", "c2"]
    assert {r.output_path.name for r in results} == {
        "T1.check.c0.log",
        "T1.check.c1.log",
        "T1.check.c2.log",
    }
    assert all(r.output_path.exists() for r in results)


def test_header_records_name_cmd_cwd_exit_duration(tmp_path):
    (r,) = _run({"hdr": Check(cmd="echo hi")}, tmp_path)
    log = _log(r)
    assert "hdr" in log and "echo hi" in log and "STDOUT" in log and "STDERR" in log
    assert "exit" in log.lower() and "duration" in log.lower()


def test_stderr_is_captured_separately(tmp_path):
    (r,) = _run({"err": Check(cmd="echo boom >&2")}, tmp_path)
    assert r.exit_code == 0
    log = _log(r)
    stdout_section, stderr_section = log.split("STDOUT\n", 1)[1].split("\nSTDERR\n", 1)
    assert stdout_section.strip() == ""
    assert stderr_section.strip() == "boom"


def test_duration_is_measured(tmp_path):
    (r,) = _run({"quick": Check(cmd="true")}, tmp_path)
    assert 0 <= r.duration_s < 3
    assert isinstance(r.output_path, Path)
