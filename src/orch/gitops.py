"""Git plumbing: worktrees, branches, commits, diffs, merges.

Every operation shells out to the ``git`` binary; there is no git library in
this project. ``git()`` raises :class:`orch.errors.GitError` on non-zero exit
when ``check=True`` (the default), including stderr.
"""

import subprocess
from pathlib import Path

from orch.errors import GitError


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run ``git -C <repo> <args>`` and return stripped stdout.

    With ``check=True`` (default), a non-zero exit raises ``GitError`` with the
    command and its stderr; with ``check=False`` the (possibly empty) stdout is
    returned regardless.
    """
    r = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if check and r.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({r.returncode}) in {repo}: {r.stderr.strip()}"
        )
    return r.stdout.strip()


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Raw git invocation for probes where a non-zero exit is a normal answer."""
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


def is_git_repo(p: Path) -> bool:
    """True when ``p`` is a directory inside a git work tree."""
    if not p.is_dir():
        return False
    r = _run(p, "rev-parse", "--is-inside-work-tree")
    return r.returncode == 0 and r.stdout.strip() == "true"


def _porcelain_path(line: str) -> str:
    """Extract the path from one ``git status --porcelain`` line."""
    path = line[3:].strip()
    if " -> " in path:  # renames: "<old> -> <new>"
        path = path.split(" -> ", 1)[1]
    return path


def _porcelain_lines(repo: Path) -> list[str]:
    """Raw ``git status --porcelain`` lines, unstripped.

    Uses ``_run`` rather than ``git`` on purpose: ``git`` strips surrounding
    whitespace from stdout, which would eat the leading status-column space of
    the first line (e.g. ``" M file"``) and shift every path by one character.
    Failures still raise ``GitError`` so a broken working dir is never silently
    reported as clean.

    ``--untracked-files=all`` is load-bearing: without it git collapses a wholly
    untracked directory to a single ``dir/`` line, which would hide the files an
    agent just wrote and read as a ``files_allowed`` violation. With it every
    untracked file is reported by its real path (``tracked_dirty`` still filters
    the ``??`` lines, so the dirty-tree refusal is unaffected).
    """
    r = _run(repo, "status", "--porcelain", "--untracked-files=all")
    if r.returncode != 0:
        raise GitError(
            f"git status failed ({r.returncode}) in {repo}: {r.stderr.strip()}"
        )
    return r.stdout.splitlines()


def tracked_dirty(repo: Path) -> list[str]:
    """Paths of tracked files with staged or unstaged changes (no untracked)."""
    return [
        _porcelain_path(ln) for ln in _porcelain_lines(repo) if not ln.startswith("??")
    ]


def untracked(repo: Path) -> list[str]:
    """Paths of untracked files."""
    return [
        _porcelain_path(ln) for ln in _porcelain_lines(repo) if ln.startswith("??")
    ]


def branch_exists(repo: Path, name: str) -> bool:
    """True when a local branch ``name`` exists."""
    r = _run(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    return r.returncode == 0


def create_branch(repo: Path, name: str, start_ref: str) -> None:
    """Create local branch ``name`` pointing at ``start_ref``."""
    git(repo, "branch", name, start_ref)


def ensure_worktree(repo: Path, wt: Path, branch: str) -> Path:
    """Return a worktree at ``wt`` checked out on ``branch``.

    Contract: the caller guarantees ``branch`` already exists (via
    ``create_branch``). If ``wt`` is already a valid worktree on ``branch`` it
    is reused; if the path exists but is not such a worktree, ``GitError`` is
    raised — the path is never deleted or recreated.
    """
    if wt.exists():
        inside = _run(wt, "rev-parse", "--is-inside-work-tree")
        if inside.returncode == 0 and inside.stdout.strip() == "true":
            head = _run(wt, "rev-parse", "--abbrev-ref", "HEAD")
            if head.stdout.strip() == branch:
                return wt
        raise GitError(f"path exists but is not a worktree on {branch}: {wt}")
    git(repo, "worktree", "add", str(wt), branch)
    return wt


def changed_files(workdir: Path) -> list[str]:
    """All changed paths from ``git status --porcelain`` (incl. untracked)."""
    return [
        _porcelain_path(ln) for ln in _porcelain_lines(workdir) if ln.strip()
    ]


def diff_three_dot(workdir: Path, base: str, head: str = "HEAD") -> str:
    """Diff of ``base...head`` (changes on ``head`` since the merge base)."""
    return git(workdir, "diff", "--no-color", f"{base}...{head}")


def commit_all(workdir: Path, message: str) -> str | None:
    """Stage everything and commit; return the new sha, or ``None`` if clean.

    Identity is supplied repo-locally on the command line so tests are hermetic.
    """
    if not _porcelain_lines(workdir):
        return None
    git(workdir, "add", "-A")
    git(
        workdir,
        "-c",
        "user.name=orch",
        "-c",
        "user.email=orch@local",
        "commit",
        "-m",
        message,
    )
    return head_sha(workdir)


def head_sha(workdir: Path) -> str:
    """Full sha of ``HEAD``."""
    return git(workdir, "rev-parse", "HEAD")


def merge_no_ff(workdir: Path, branch: str) -> tuple[bool, list[str]]:
    """Merge ``branch`` with ``--no-ff``; return ``(ok, conflicted_files)``.

    On conflict the merge is aborted and the unmerged paths are returned.
    """
    r = subprocess.run(
        ["git", "-C", str(workdir), "merge", "--no-ff", "--no-edit", branch],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return True, []
    conflicts = git(workdir, "diff", "--name-only", "--diff-filter=U").splitlines()
    git(workdir, "merge", "--abort")
    return False, conflicts


def is_ancestor(workdir: Path, ref: str, of: str = "HEAD") -> bool:
    """True when ``ref`` is an ancestor of ``of`` (default ``HEAD``)."""
    r = _run(workdir, "merge-base", "--is-ancestor", ref, of)
    return r.returncode == 0
