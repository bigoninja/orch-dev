import re
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orch.errors import PlanValidationError

# Check keys are descriptive and free-form (e.g. "typecheck", "migrate-0022"); uniqueness per task is enforced by the dict itself.

_FENCED_YAML_RE = re.compile(r"```yaml[^\n]*\n(.*?)```", re.DOTALL)

# (task_id | None, code, message)
Issue = tuple[str | None, str, str]


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd: str
    cwd: str | None = None


class TaskDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    title: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    files_allowed: list[str] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    checks: dict[str, Check] = Field(min_length=1)
    post_merge_checks: dict[str, Check] | None = None
    risky: bool = False
    risky_reasons: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    brief: str
    source: str | None = None
    tasks: list[TaskDef] = Field(min_length=1)


def _cycle_remaining_ids(tasks: list[TaskDef]) -> list[str]:
    """Kahn's algorithm: return ids left with unsatisfied in-degree (i.e. in a cycle)."""
    ids = list(dict.fromkeys(t.id for t in tasks))
    indegree = dict.fromkeys(ids, 0)
    dependents: dict[str, list[str]] = {i: [] for i in ids}
    for task in tasks:
        for dep in task.depends_on:
            if dep in indegree:
                dependents[dep].append(task.id)
                indegree[task.id] += 1
    ready = [i for i in ids if indegree[i] == 0]
    resolved = 0
    while ready:
        node = ready.pop(0)
        resolved += 1
        for dependent in dependents[node]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    return [i for i in ids if indegree[i] > 0]


def validate_plan(plan: Plan, risky_patterns: list[str]) -> list[Issue]:
    issues: list[Issue] = []

    seen: set[str] = set()
    for task in plan.tasks:
        if task.id in seen:
            issues.append((task.id, "duplicate_id", f"duplicate task id {task.id!r}"))
        seen.add(task.id)

    ids = {t.id for t in plan.tasks}
    for task in plan.tasks:
        for dep in task.depends_on:
            if dep not in ids:
                issues.append((task.id, "unknown_dep", f"depends_on {dep!r} is not a known task id"))

    remaining = _cycle_remaining_ids(plan.tasks)
    if remaining:
        issues.append((None, "cycle", "dependency cycle among tasks: " + ", ".join(sorted(remaining))))

    for task in plan.tasks:
        if not task.checks:
            issues.append((task.id, "no_checks", "task has no checks"))
        for key in task.checks:
            if not str(key).strip():
                issues.append((task.id, "empty_check_key", "check keys must be non-empty"))

    return issues


def flag_risky(plan: Plan, patterns: list[str]) -> Plan:
    flagged = plan.model_copy(deep=True)
    for task in flagged.tasks:
        reasons = sorted(
            {p for p in patterns if any(PurePosixPath(f).full_match(p) for f in task.files_allowed)}
        )
        task.risky = bool(reasons)
        task.risky_reasons = reasons
    return flagged


def dump_plan(plan: Plan, path: Path) -> None:
    try:
        path.write_text(yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False))
    except OSError as e:
        raise PlanValidationError([f"cannot write plan {path}: {e}"]) from e


def load_plan(path: Path) -> Plan:
    try:
        raw = yaml.safe_load(path.read_text())
        return Plan.model_validate(raw)
    except (yaml.YAMLError, ValidationError, OSError) as e:
        raise PlanValidationError([f"invalid plan {path}: {e}"]) from e


def parse_planner_output(text: str) -> Plan:
    """Extract the last fenced ```yaml block containing ``run_id:`` and validate its shape.

    Deliberately does NOT call ``validate_plan``: semantic validation (duplicate ids,
    unknown deps, cycles, check keys) stays a separate step so that a structurally
    valid but semantically flawed plan can still be written to ``plan.invalid.yaml``.
    """
    blocks = _FENCED_YAML_RE.findall(text)
    if not blocks:
        raise PlanValidationError(["planner output contains no ```yaml block"])

    candidates: list[object] = []
    last_yaml_error: str | None = None
    for block in blocks:
        try:
            loaded = yaml.safe_load(block)
        except yaml.YAMLError as e:
            last_yaml_error = f"planner yaml block is not valid yaml: {e}"
            continue
        if isinstance(loaded, dict) and "run_id" in loaded:
            candidates.append(loaded)
    if not candidates:
        if last_yaml_error is not None:
            raise PlanValidationError([last_yaml_error])
        raise PlanValidationError(["planner output has no ```yaml block containing run_id"])

    try:
        return Plan.model_validate(candidates[-1])
    except ValidationError as e:
        raise PlanValidationError([f"planner output does not match the plan schema: {e}"]) from e

