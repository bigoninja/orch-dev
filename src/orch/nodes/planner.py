"""The planner node: one agent turn that turns a brief into ``plan.yaml``.

The planner is the only pre-run model call. It is handed the brief, an optional
source document, and a deterministic digest of the target repo's own surfaces
(``gather_repo_facts``) so it can prefer real commands over invented ones, then
its reply is parsed and validated *before* anything is written:

* a reply that does not parse → the raw text is preserved at
  ``plan.invalid.yaml`` and :class:`PlanValidationError` propagates;
* a structurally valid plan with semantic issues (duplicate ids, unknown deps,
  cycles, unknown check keys) → the parsed-and-risky-flagged plan is written to
  ``plan.invalid.yaml`` and the issues are raised — ``plan.yaml`` is **never**
  written for an invalid plan;
* a valid plan → the risky-flagged plan is written to ``plan.yaml`` atomically
  and returned.

Validation failures are the human's to fix at the approval gate, so the node
fails loudly instead of repairing anything.
"""

import json
from pathlib import Path

import yaml

from orch.adapters.base import AgentAdapter
from orch.artifacts import RunDir, atomic_write
from orch.config import Config
from orch.errors import GitError, PlanValidationError
from orch.gitops import git
from orch.nodes import passthrough_env
from orch.plan import Plan, dump_plan, flag_risky, parse_planner_output, validate_plan
from orch.prompts import build_planner_prompt

#: Lines of ``README.md``/``AGENTS.md`` included in the repo facts.
HEAD_LINES = 80
#: Entries of ``git ls-files`` included in the repo facts.
LS_FILES_MAX = 200


def _head(text: str, limit: int) -> str:
    """First ``limit`` lines of ``text``, joined without a trailing newline."""
    return "\n".join(text.splitlines()[:limit])


def _package_scripts(path: Path) -> str | None:
    """Render ``package.json``'s ``scripts`` object as ``name: command`` lines."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if not isinstance(scripts, dict) or not scripts:
        return None
    return "\n".join(f"{name}: {command}" for name, command in scripts.items())


def gather_repo_facts(repo: Path) -> str:
    """A labeled, deterministic digest of the target repo for the planner prompt.

    Sections, in order: ``README.md`` (head 80 lines), ``AGENTS.md`` (head 80),
    ``git ls-files`` (head 200 entries), and ``package.json`` scripts when
    present. A missing, unreadable, or untracked-file-less source is skipped
    rather than raising; the order never depends on the filesystem.
    """
    repo = Path(repo)
    sections: list[tuple[str, str]] = []

    for name in ("README.md", "AGENTS.md"):
        path = repo / name
        if not path.is_file():
            continue
        try:
            body = _head(path.read_text(encoding="utf-8"), HEAD_LINES)
        except (OSError, UnicodeDecodeError):
            continue
        sections.append((name, body))

    try:
        entries = git(repo, "ls-files").splitlines()
    except GitError:
        entries = []
    tracked = [entry for entry in entries[:LS_FILES_MAX] if entry]
    if tracked:
        sections.append(("git ls-files", "\n".join(tracked)))

    scripts = _package_scripts(repo / "package.json")
    if scripts:
        sections.append(("package.json scripts", scripts))

    if not sections:
        return "(no repository facts available)"
    return "\n\n".join(f"## {label}\n\n{body}" for label, body in sections)


def run_planner(
    *,
    cfg: Config,
    adapter: AgentAdapter,
    brief: str,
    source: str | None,
    run_dir: RunDir,
) -> Plan:
    """Run the planner once; write ``plan.yaml`` only for a valid plan.

    Returns the risky-flagged :class:`~orch.plan.Plan` on success. Raises
    :class:`PlanValidationError` (after writing ``plan.invalid.yaml``) when the
    reply does not parse or the plan has semantic issues.
    """
    source_text = Path(source).read_text() if source else None
    prompt = build_planner_prompt(
        brief=brief,
        source_text=source_text,
        repo_facts=gather_repo_facts(cfg.target_repo),
    )
    result = adapter.run(
        role="planner",
        prompt=prompt,
        cwd=cfg.target_repo,
        session_id=None,
        title=f"{run_dir.root.name}/planner",
        env=passthrough_env(cfg),
    )

    invalid_path = run_dir.root / "plan.invalid.yaml"
    try:
        plan = parse_planner_output(result.text)
    except PlanValidationError:
        atomic_write(invalid_path, result.text)
        raise

    issues = validate_plan(plan, cfg.risky_patterns)
    plan = flag_risky(plan, cfg.risky_patterns)
    if issues:
        dump_plan(plan, invalid_path)
        raise PlanValidationError(
            [f"{tid or '-'}: {code}: {msg}" for tid, code, msg in issues]
        )

    atomic_write(
        run_dir.plan_path,
        yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False),
    )
    return plan
