"""Prompt builders for the planner, implementer, fixer and reviewer roles.

These strings are contracts: the pipeline nodes call the builders verbatim and
``tests/unit/test_prompts.py`` pins the full rendered output. Changing a
template therefore requires deliberately updating the golden expectations.
"""

from pathlib import Path

from orch.plan import Check, TaskDef

# Kept verbatim: the reviewer's reply is machine-parsed against this spec.
VERDICT_JSON_SPEC = (
    'Return ONLY a JSON object: {"verdict":"approve"|"reject","risky":true|false,'
    '"findings":[{"severity":"blocker|major|minor","file":"","note":""}],"summary":""}.'
)

# The plan contract, emitted as JSON — the same choice the reviewer verdict already
# makes (VERDICT_JSON_SPEC). JSON has no block scalars and no quote-escaping rules,
# so a shell command containing quotes cannot corrupt the parse; a YAML plan did
# exactly that, and run 20261016-163406 was refused for it.
PLAN_JSON_SPEC = """\
```json
{
  "run_id": "<string>",
  "brief": "<string>",
  "source": "<string or null>",
  "tasks": [
    {
      "id": "<string>",
      "title": "<string>",
      "description": "<string>",
      "depends_on": ["<task id>", "..."],
      "files_allowed": ["<path or glob>", "..."],
      "acceptance_criteria": ["<string>", "..."],
      "checks": {
        "<check key>": {"cmd": "<shell command>", "cwd": "<optional relative directory>"}
      },
      "risky": false
    }
  ]
}
```"""

TASK_PROMPT_TEMPLATE = """\
You are implementing a single task in an automated development pipeline.

Run: {run_id}
Task: {task_id} - {title}

## Description

{description}

## Files you may change

Work only within these files. Do not create, edit, or delete any other file:
{files}

## Acceptance criteria

{criteria}

## Checks

Run each check and make it pass. Do not weaken or delete a check:
{checks}

## Repository notes

{repo_note}

## When blocked

If you cannot complete the task, stop and reply with a single first line beginning exactly:
BLOCKED: <reason>
"""

FIX_PROMPT_TEMPLATE = """\
You are fixing a task that failed its checks or review in an automated development pipeline.

Task: {task_id} - {title}

## Findings to address

{findings}

## Required before you finish

Address every finding above. Then run each check below and make it pass; do not weaken or delete a check.

## Checks

{checks}
"""

REVIEW_PROMPT_TEMPLATE = """\
You are the reviewer for one task in an automated development pipeline.

Run: {run_id}
Task: {task_id} - {title}
Integration branch: {integration_branch}

## Artifacts

Diff: {diff_path}
Evidence files:
{evidence}

## Acceptance criteria

{criteria}

## Instructions

Review the diff against the acceptance criteria above. You are a fresh reviewer with no prior context; you may run any commands in the worktree to verify the claims in the diff rather than trusting them.

## Verdict

{verdict_spec}
"""

PLANNER_PROMPT_TEMPLATE = """\
You are the planner for an automated development pipeline.

## Brief

{brief}

## Source document

{source}

## Repository facts

{repo_facts}

## Output format

Reply with exactly one fenced ```json block and nothing else. It must validate against this schema:

{schema}

Shell commands go verbatim in the JSON string: escape only `"` and `\\`. A single quote inside a command needs no escaping.

Choose the checks from real commands that exist in the repository; prefer the scripts from package.json shown in the repository facts above over inventing commands.
"""


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _format_check(key: str, check: Check) -> str:
    line = f"- {key}: `{check.cmd}`"
    if check.cwd:
        line += f" (cwd: {check.cwd})"
    return line


def _checks_block(task: TaskDef) -> str:
    return "\n".join(_format_check(key, check) for key, check in task.checks.items())


def build_task_prompt(task: TaskDef, *, run_id: str, repo_note: str) -> str:
    """Prompt for the implementer of one task."""
    return TASK_PROMPT_TEMPLATE.format(
        run_id=run_id,
        task_id=task.id,
        title=task.title,
        description=task.description,
        files=_bullet_list(task.files_allowed),
        criteria=_bullet_list(task.acceptance_criteria),
        checks=_checks_block(task),
        repo_note=repo_note,
    )


def build_fix_prompt(task: TaskDef, findings: list[str]) -> str:
    """Prompt for a repair attempt on a task that failed checks or review."""
    return FIX_PROMPT_TEMPLATE.format(
        task_id=task.id,
        title=task.title,
        findings=_bullet_list(findings),
        checks=_checks_block(task),
    )


def build_review_prompt(
    task: TaskDef,
    *,
    run_id: str,
    diff_path: Path,
    evidence_paths: list[Path],
    integration_branch: str,
) -> str:
    """Prompt for a fresh reviewer session on a committed task."""
    evidence = "\n".join(f"- {path}" for path in evidence_paths) if evidence_paths else "- (none)"
    return REVIEW_PROMPT_TEMPLATE.format(
        run_id=run_id,
        task_id=task.id,
        title=task.title,
        integration_branch=integration_branch,
        diff_path=diff_path,
        evidence=evidence,
        criteria=_bullet_list(task.acceptance_criteria),
        verdict_spec=VERDICT_JSON_SPEC,
    )


def build_planner_prompt(*, brief: str, source_text: str | None, repo_facts: str) -> str:
    """Prompt for the planner; its reply is parsed by ``plan.parse_planner_output``."""
    return PLANNER_PROMPT_TEMPLATE.format(
        brief=brief,
        source=source_text if source_text is not None else "(none)",
        repo_facts=repo_facts,
        schema=PLAN_JSON_SPEC,
    )
