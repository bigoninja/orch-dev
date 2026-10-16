"""``allowed_violation`` — the ``files_allowed`` boundary at commit time.

Ownership is declared as globs (``src/**``, ``db/migrations/**``), so the check
must be glob-aware: a literal-path comparison would flag every edit under a glob
as a stray. A literal pattern still matches only itself, and anything unmatched
is a violation (the boundary fails closed).
"""

import subprocess
from pathlib import Path

from orch.nodes.task import allowed_violation, work_role


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)], check=True, capture_output=True
    )
    return repo


def test_glob_pattern_allows_a_nested_change(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.js").write_text("ok\n")

    assert allowed_violation(repo, ["src/**"]) == []


def test_glob_pattern_flags_a_path_outside_it(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "src").mkdir()
    (repo / "src" / "app.js").write_text("ok\n")
    (repo / "other.txt").write_text("stray\n")

    assert allowed_violation(repo, ["src/**"]) == ["other.txt"]


def test_literal_pattern_still_matches_only_itself(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("ok\n")
    (repo / "src").mkdir()
    (repo / "src" / "x.py").write_text("x\n")

    # The literal file is allowed; the nested path is not.
    assert allowed_violation(repo, ["a.txt"]) == ["src/x.py"]


def test_an_exact_path_pattern_allows_that_exact_path(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "pkg" / "a.py").parent.mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("ok\n")
    (repo / "pkg" / "b.py").write_text("stray\n")

    assert allowed_violation(repo, ["pkg/a.py"]) == ["pkg/b.py"]


def test_work_role_is_repair_for_a_repair_task():
    """A repair is its own model role, distinct from ordinary implement/fix work."""
    assert work_role({"origin": "repair"}, "implement") == "repair"
    assert work_role({"origin": "repair"}, "fix") == "repair"


def test_work_role_is_the_default_for_an_ordinary_task():
    assert work_role({}, "implement") == "implement"
    assert work_role({"origin": None}, "fix") == "fix"
    assert work_role({"origin": "planned"}, "implement") == "implement"
