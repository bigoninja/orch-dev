"""Prompt builders and planner-output extraction.

The golden tests below pin the FULL rendered prompt strings: they are the
contract Tasks 12/13/15 rely on, so any change to a template must be a
deliberate change to these expectations.
"""

from pathlib import Path

import pytest

from orch.errors import PlanValidationError
from orch.plan import Check, TaskDef, parse_planner_output, validate_plan
from orch.prompts import (
    build_fix_prompt,
    build_planner_prompt,
    build_review_prompt,
    build_task_prompt,
)

RUN_ID = "20261016-120000"
REPO_NOTE = "Repo uses uv; run `uv run pytest` from the repo root."

VERDICT_JSON_LINE = (
    'Return ONLY a JSON object: {"verdict":"approve"|"reject","risky":true|false,'
    '"findings":[{"severity":"blocker|major|minor","file":"","note":""}],"summary":""}.'
)

TASK = TaskDef(
    id="T3",
    title="Add health endpoint",
    description="Add a GET /healthz endpoint that returns 200 with {ok: true}.",
    depends_on=["T1", "T2"],
    files_allowed=["src/api/health.py", "tests/api/test_health.py"],
    acceptance_criteria=["GET /healthz returns 200", 'response body is {"ok": true}'],
    checks={
        "test": Check(cmd="pytest -q tests/api", cwd="backend"),
        "lint": Check(cmd="ruff check src/api"),
    },
)

FINDINGS = [
    "Reviewer: health check returns 503 when the database is down",
    "Check `test` failed: 2 tests failed",
]

DIFF_PATH = Path("runs/20261016-120000/evidence/T3.diff")
EVIDENCE_PATHS = [Path("runs/20261016-120000/evidence/T3.test.txt")]
INTEGRATION_BRANCH = "orch/integration"

BRIEF = "Add GitHub OAuth login."
SOURCE_TEXT = "design.md: users authenticate with GitHub OAuth."
REPO_FACTS = "README.md (head):\n# demo\npackage.json scripts:\n- test: vitest\n- lint: eslint ."

EXPECTED_TASK_PROMPT = "\n".join([
    "You are implementing a single task in an automated development pipeline.",
    "",
    f"Run: {RUN_ID}",
    "Task: T3 - Add health endpoint",
    "",
    "## Description",
    "",
    "Add a GET /healthz endpoint that returns 200 with {ok: true}.",
    "",
    "## Files you may change",
    "",
    "Work only within these files. Do not create, edit, or delete any other file:",
    "- src/api/health.py",
    "- tests/api/test_health.py",
    "",
    "## Acceptance criteria",
    "",
    "- GET /healthz returns 200",
    '- response body is {"ok": true}',
    "",
    "## Checks",
    "",
    "Run each check and make it pass. Do not weaken or delete a check:",
    "- test: `pytest -q tests/api` (cwd: backend)",
    "- lint: `ruff check src/api`",
    "",
    "## Repository notes",
    "",
    REPO_NOTE,
    "",
    "## When blocked",
    "",
    "If you cannot complete the task, stop and reply with a single first line beginning exactly:",
    "BLOCKED: <reason>",
]) + "\n"

EXPECTED_FIX_PROMPT = "\n".join([
    "You are fixing a task that failed its checks or review in an automated development pipeline.",
    "",
    "Task: T3 - Add health endpoint",
    "",
    "## Findings to address",
    "",
    "- Reviewer: health check returns 503 when the database is down",
    "- Check `test` failed: 2 tests failed",
    "",
    "## Required before you finish",
    "",
    "Address every finding above. Then run each check below and make it pass; do not weaken or delete a check.",
    "",
    "## Checks",
    "",
    "- test: `pytest -q tests/api` (cwd: backend)",
    "- lint: `ruff check src/api`",
]) + "\n"

EXPECTED_REVIEW_PROMPT = "\n".join([
    "You are the reviewer for one task in an automated development pipeline.",
    "",
    f"Run: {RUN_ID}",
    "Task: T3 - Add health endpoint",
    f"Integration branch: {INTEGRATION_BRANCH}",
    "",
    "## Artifacts",
    "",
    f"Diff: {DIFF_PATH}",
    "Evidence files:",
    f"- {EVIDENCE_PATHS[0]}",
    "",
    "## Acceptance criteria",
    "",
    "- GET /healthz returns 200",
    '- response body is {"ok": true}',
    "",
    "## Instructions",
    "",
    "Review the diff against the acceptance criteria above. You are a fresh reviewer with no prior context; you may run any commands in the worktree to verify the claims in the diff rather than trusting them.",
    "",
    "## Verdict",
    "",
    VERDICT_JSON_LINE,
]) + "\n"

EXPECTED_PLANNER_PROMPT = "\n".join([
    "You are the planner for an automated development pipeline.",
    "",
    "## Brief",
    "",
    BRIEF,
    "",
    "## Source document",
    "",
    SOURCE_TEXT,
    "",
    "## Repository facts",
    "",
    "README.md (head):",
    "# demo",
    "package.json scripts:",
    "- test: vitest",
    "- lint: eslint .",
    "",
    "## Output format",
    "",
    "Reply with exactly one fenced ```json block and nothing else. It must validate against this schema:",
    "",
    "```json",
    "{",
    "  \"run_id\": \"<string>\",",
    "  \"brief\": \"<string>\",",
    "  \"source\": \"<string or null>\",",
    "  \"tasks\": [",
    "    {",
    "      \"id\": \"<string>\",",
    "      \"title\": \"<string>\",",
    "      \"description\": \"<string>\",",
    "      \"depends_on\": [\"<task id>\", \"...\"],",
    "      \"files_allowed\": [\"<path or glob>\", \"...\"],",
    "      \"acceptance_criteria\": [\"<string>\", \"...\"],",
    "      \"checks\": {",
    "        \"<check key>\": {\"cmd\": \"<shell command>\", \"cwd\": \"<optional relative directory>\"}",
    "      },",
    "      \"risky\": false",
    "    }",
    "  ]",
    "}",
    "```",
    "",
    "Shell commands go verbatim in the JSON string: escape only `\"` and `\\`. A single quote inside a command needs no escaping.",
    "",
    "Choose the checks from real commands that exist in the repository; prefer the scripts from package.json shown in the repository facts above over inventing commands.",
]) + "\n"


# --- golden prompts: the full rendered string is the contract -----------------


def test_task_prompt_golden() -> None:
    assert build_task_prompt(TASK, run_id=RUN_ID, repo_note=REPO_NOTE) == EXPECTED_TASK_PROMPT


def test_fix_prompt_golden() -> None:
    assert build_fix_prompt(TASK, FINDINGS) == EXPECTED_FIX_PROMPT


def test_review_prompt_golden() -> None:
    rendered = build_review_prompt(
        TASK,
        run_id=RUN_ID,
        diff_path=DIFF_PATH,
        evidence_paths=EVIDENCE_PATHS,
        integration_branch=INTEGRATION_BRANCH,
    )
    assert rendered == EXPECTED_REVIEW_PROMPT


def test_planner_prompt_golden() -> None:
    assert build_planner_prompt(brief=BRIEF, source_text=SOURCE_TEXT, repo_facts=REPO_FACTS) == (
        EXPECTED_PLANNER_PROMPT
    )


# --- behavioral assertions ----------------------------------------------------


def test_task_prompt_pins_blocked_line_and_files_boundary() -> None:
    rendered = build_task_prompt(TASK, run_id=RUN_ID, repo_note=REPO_NOTE)
    assert "BLOCKED: <reason>" in rendered
    assert rendered.count("BLOCKED:") == 1
    assert "Work only within these files" in rendered
    for path in TASK.files_allowed:
        assert path in rendered
    for criterion in TASK.acceptance_criteria:
        assert criterion in rendered


def test_task_prompt_includes_check_commands_and_cwd() -> None:
    rendered = build_task_prompt(TASK, run_id=RUN_ID, repo_note=REPO_NOTE)
    assert "`pytest -q tests/api` (cwd: backend)" in rendered
    assert "`ruff check src/api`" in rendered


def test_fix_prompt_lists_findings_and_demands_verification() -> None:
    rendered = build_fix_prompt(TASK, FINDINGS)
    for finding in FINDINGS:
        assert f"- {finding}" in rendered
    assert "Then run each check below and make it pass" in rendered


def test_review_prompt_demands_exact_json_verdict() -> None:
    rendered = build_review_prompt(
        TASK,
        run_id=RUN_ID,
        diff_path=DIFF_PATH,
        evidence_paths=EVIDENCE_PATHS,
        integration_branch=INTEGRATION_BRANCH,
    )
    assert VERDICT_JSON_LINE in rendered
    for key in ('"verdict"', '"risky"', '"findings"', '"summary"'):
        assert key in rendered
    assert str(DIFF_PATH) in rendered
    assert str(EVIDENCE_PATHS[0]) in rendered
    assert "run any commands in the worktree" in rendered


def test_review_prompt_handles_no_evidence_files() -> None:
    rendered = build_review_prompt(
        TASK,
        run_id=RUN_ID,
        diff_path=DIFF_PATH,
        evidence_paths=[],
        integration_branch=INTEGRATION_BRANCH,
    )
    assert "Evidence files:\n- (none)" in rendered


def test_planner_prompt_demands_one_json_block_with_schema_fields() -> None:
    rendered = build_planner_prompt(brief=BRIEF, source_text=None, repo_facts=REPO_FACTS)
    assert "exactly one fenced ```json block" in rendered
    for field in (
        '"run_id":',
        '"brief":',
        '"source":',
        '"tasks":',
        '"id":',
        '"title":',
        '"description":',
        '"depends_on":',
        '"files_allowed":',
        '"acceptance_criteria":',
        '"checks":',
        '"risky":',
    ):
        assert field in rendered
    assert "## Source document\n\n(none)" in rendered
    assert "package.json" in rendered


# --- parse_planner_output -----------------------------------------------------

VALID_PLAN_JSON = """\
{
  "run_id": "20261016-120000",
  "brief": "Add GitHub OAuth login.",
  "source": null,
  "tasks": [
    {
      "id": "T1",
      "title": "Add auth module",
      "description": "Add the GitHub OAuth client.",
      "depends_on": [],
      "files_allowed": ["src/auth.py"],
      "acceptance_criteria": ["auth module exists"],
      "checks": {"test": {"cmd": "pytest -q tests/test_auth.py"}},
      "risky": false
    }
  ]
}
"""

CYCLE_PLAN_JSON = """\
{
  "run_id": "20261016-120000",
  "brief": "Cycle fixture.",
  "source": null,
  "tasks": [
    {
      "id": "T1", "title": "One", "description": "One.",
      "depends_on": ["T2"], "files_allowed": ["src/one.py"],
      "acceptance_criteria": ["one exists"],
      "checks": {"test": {"cmd": "true"}}, "risky": false
    },
    {
      "id": "T2", "title": "Two", "description": "Two.",
      "depends_on": ["T1"], "files_allowed": ["src/two.py"],
      "acceptance_criteria": ["two exists"],
      "checks": {"test": {"cmd": "true"}}, "risky": false
    }
  ]
}
"""


def test_parse_planner_output_with_chatter_around_block() -> None:
    text = f"Sure, here is the plan:\n\n```json\n{VALID_PLAN_JSON}```\n\nWant changes?\n"
    plan = parse_planner_output(text)
    assert plan.run_id == "20261016-120000"
    assert plan.brief == "Add GitHub OAuth login."
    assert plan.source is None
    assert [t.id for t in plan.tasks] == ["T1"]
    assert plan.tasks[0].checks["test"].cmd == "pytest -q tests/test_auth.py"


def test_parse_planner_output_takes_last_run_id_block() -> None:
    first = VALID_PLAN_JSON.replace("20261016-120000", "old-run")
    text = f"```json\n{first}```\nstale, ignore that one\n```json\n{VALID_PLAN_JSON}```\n"
    assert parse_planner_output(text).run_id == "20261016-120000"


def test_parse_planner_output_without_block_raises() -> None:
    with pytest.raises(PlanValidationError):
        parse_planner_output("I could not produce a plan today.")


def test_parse_planner_output_without_run_id_raises() -> None:
    with pytest.raises(PlanValidationError):
        parse_planner_output('```json\n{"foo": "bar"}\n```\n')


def test_parse_planner_output_unparseable_json_raises() -> None:
    with pytest.raises(PlanValidationError):
        parse_planner_output('```json\n{"run_id": [unclosed}\n```\n')


def test_parse_planner_output_invalid_schema_raises() -> None:
    with pytest.raises(PlanValidationError):
        parse_planner_output('```json\n{"run_id": "r1", "brief": "b"}\n```\n')


def test_parse_planner_output_does_not_validate() -> None:
    """Validation is a separate step: a cyclic plan parses, validate_plan flags it."""
    plan = parse_planner_output(f"```json\n{CYCLE_PLAN_JSON}```\n")
    assert [t.id for t in plan.tasks] == ["T1", "T2"]
    issues = validate_plan(plan, [])
    assert any(code == "cycle" for _, code, _ in issues)


# --- regression: quoted shell commands must not refuse the plan ----------------

#: The verbatim command that refused run 20261016-163406 as an unparseable YAML
#: plan. As JSON it is just a string with escaped double quotes — no block scalars,
#: no quote-escaping rules.
SHELL_QUOTED_CMD_PLAN = """\
```json
{
  "run_id": "r1",
  "brief": "b",
  "source": null,
  "tasks": [
    {
      "id": "T1",
      "title": "t",
      "description": "d",
      "depends_on": [],
      "files_allowed": ["web/**"],
      "acceptance_criteria": ["a"],
      "checks": {
        "no-platform-display-id": {
          "cmd": "! grep -q \\"actor.id === 'key:platform'\\" src/lib/authz.ts",
          "cwd": "web"
        }
      },
      "risky": false
    }
  ]
}
```"""


def test_parse_planner_output_accepts_a_quoted_shell_command() -> None:
    """A quoted check command must survive intact — the defect that refused a real run."""
    plan = parse_planner_output(SHELL_QUOTED_CMD_PLAN)
    check = plan.tasks[0].checks["no-platform-display-id"]
    assert check.cmd == """! grep -q "actor.id === 'key:platform'" src/lib/authz.ts"""
    assert check.cwd == "web"


def test_planner_prompt_requires_json_and_states_the_escaping_rule() -> None:
    """The contract must be JSON, and it must say how to escape a command's quotes."""
    rendered = build_planner_prompt(brief=BRIEF, source_text=None, repo_facts=REPO_FACTS)
    assert "fenced ```json block" in rendered
    assert "single quote inside a command needs no escaping" in rendered
