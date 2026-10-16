import subprocess
from pathlib import Path

import pytest

from orch.errors import GitError
from orch.gitops import (
    branch_exists,
    changed_files,
    commit_all,
    create_branch,
    diff_three_dot,
    ensure_worktree,
    git,
    head_sha,
    is_ancestor,
    is_git_repo,
    merge_no_ff,
    tracked_dirty,
    untracked,
)


# --- real-repo helpers (plain subprocess, independent of the code under test) ---


def _raw(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _init(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    _raw(repo, "init", "-b", "main")
    _raw(repo, "config", "user.name", "test")
    _raw(repo, "config", "user.email", "test@local")
    return repo


def _commit(repo: Path, message: str) -> None:
    _raw(repo, "add", "-A")
    assert _raw(repo, "commit", "-m", message).returncode == 0


def _repo_with_commit(tmp_path: Path) -> Path:
    repo = _init(tmp_path / "repo")
    (repo / "base.txt").write_text("line1\n")
    _commit(repo, "init")
    return repo


# --- git wrapper ---


def test_git_returns_stripped_stdout(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_git_raises_git_error_with_stderr(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitError) as exc:
        git(plain, "rev-parse", "HEAD")
    assert "not a git repository" in str(exc.value)


def test_git_check_false_does_not_raise(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert git(plain, "rev-parse", "HEAD", check=False) == ""


# --- repo detection ---


def test_is_git_repo_true_for_repo_false_for_plain_dir(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert is_git_repo(repo) is True
    assert is_git_repo(plain) is False
    assert is_git_repo(tmp_path / "missing") is False


# --- dirty / untracked classification ---


def test_tracked_dirty_distinguishes_modified_vs_untracked(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    assert tracked_dirty(repo) == []
    assert untracked(repo) == []

    (repo / "base.txt").write_text("line1\nline2\n")  # modified tracked
    (repo / "new.txt").write_text("new\n")  # untracked

    assert tracked_dirty(repo) == ["base.txt"]
    assert untracked(repo) == ["new.txt"]


# --- branches + worktrees ---


def test_branch_exists_and_create(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    assert branch_exists(repo, "main") is True
    assert branch_exists(repo, "nope") is False

    create_branch(repo, "feat/y", "main")
    assert branch_exists(repo, "feat/y") is True


def test_ensure_worktree_creates_and_reuses(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    create_branch(repo, "feat/x", "main")

    first = ensure_worktree(repo, wt, "feat/x")
    assert first == wt
    assert wt.is_dir()
    assert (wt / "base.txt").read_text() == "line1\n"
    assert git(wt, "rev-parse", "--abbrev-ref", "HEAD") == "feat/x"

    # calling twice returns the same path without error (idempotent reuse)
    second = ensure_worktree(repo, wt, "feat/x")
    assert second == wt


def test_ensure_worktree_refuses_invalid_existing_path(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "junk.txt").write_text("junk\n")
    create_branch(repo, "feat/x", "main")

    with pytest.raises(GitError):
        ensure_worktree(repo, wt, "feat/x")

    # never deletes or recreates: the pre-existing path is untouched
    assert (wt / "junk.txt").read_text() == "junk\n"


def test_changed_files_in_worktree_after_adding_file(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    create_branch(repo, "feat/x", "main")
    ensure_worktree(repo, wt, "feat/x")

    (wt / "new.txt").write_text("hi\n")
    assert changed_files(wt) == ["new.txt"]


def test_changed_files_parses_rename_syntax(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    _raw(repo, "mv", "base.txt", "renamed.txt")
    assert changed_files(repo) == ["renamed.txt"]


# --- diffs / commits ---


def test_diff_three_dot_contains_the_change(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    create_branch(repo, "feat/x", "main")
    ensure_worktree(repo, wt, "feat/x")

    (wt / "base.txt").write_text("line1\nchanged\n")
    commit_all(wt, "change base")

    diff = diff_three_dot(wt, "main")
    assert "+changed" in diff


def test_commit_all_returns_none_when_clean(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    assert commit_all(repo, "nothing to do") is None


def test_commit_all_returns_sha_when_dirty(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    (repo / "base.txt").write_text("line1\nmore\n")

    sha = commit_all(repo, "change base")
    assert sha is not None
    assert sha == head_sha(repo)
    assert len(sha) == 40
    assert changed_files(repo) == []
    assert git(repo, "log", "-1", "--pretty=%s") == "change base"


# --- merges ---


def test_merge_no_ff_success(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    create_branch(repo, "feature", "main")
    ensure_worktree(repo, wt, "feature")
    (wt / "feature.txt").write_text("feature\n")
    commit_all(wt, "feature work")

    ok, conflicts = merge_no_ff(repo, "feature")
    assert ok is True
    assert conflicts == []
    assert (repo / "feature.txt").exists()
    # no-ff: a merge commit exists
    assert len(git(repo, "log", "-1", "--pretty=%P").split()) == 2


def test_merge_no_ff_conflict_reports_files(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    create_branch(repo, "feature", "main")
    ensure_worktree(repo, wt, "feature")

    (wt / "base.txt").write_text("feature-line\n")  # both branches touch line 1
    commit_all(wt, "feature change")
    (repo / "base.txt").write_text("main-line\n")
    commit_all(repo, "main change")

    ok, conflicts = merge_no_ff(repo, "feature")
    assert ok is False
    assert conflicts == ["base.txt"]
    # merge was aborted: tree clean, no merge in progress
    assert changed_files(repo) == []
    assert not (repo / ".git" / "MERGE_HEAD").exists()


def test_is_ancestor_before_and_after_merge(tmp_path: Path):
    repo = _repo_with_commit(tmp_path)
    wt = tmp_path / "wt"
    create_branch(repo, "feature", "main")
    ensure_worktree(repo, wt, "feature")
    (wt / "feature.txt").write_text("feature\n")
    commit_all(wt, "feature work")

    assert is_ancestor(repo, "feature", of="HEAD") is False
    ok, _ = merge_no_ff(repo, "feature")
    assert ok is True
    assert is_ancestor(repo, "feature", of="HEAD") is True
