import pytest
from pathlib import Path
from orch.plan import Check, Plan, TaskDef, load_plan, dump_plan, validate_plan, flag_risky
from orch.errors import PlanValidationError

MB = lambda tid, deps=(), files=("a.py",): TaskDef(
    id=tid, title=tid, description="d", depends_on=list(deps), files_allowed=list(files),
    acceptance_criteria=["works"], checks={"test": Check(cmd="true")})

def _plan(*tasks): return Plan(run_id="r1", brief="b", source=None, tasks=list(tasks))

def test_valid_plan_has_no_issues():
    p = flag_risky(_plan(MB("A"), MB("B", deps=("A",))), ["db/migrations/**"])
    assert validate_plan(p, ["db/migrations/**"]) == []

def test_duplicate_ids():
    issues = validate_plan(_plan(MB("A"), MB("A")), [])
    assert any(code == "duplicate_id" for _, code, _ in issues)

def test_unknown_dependency():
    issues = validate_plan(_plan(MB("A", deps=("NOPE",))), [])
    assert any(code == "unknown_dep" for _, code, _ in issues)

def test_cycle_detected():
    issues = validate_plan(_plan(MB("A", deps=("B",)), MB("B", deps=("A",))), [])
    assert any(code == "cycle" for _, code, _ in issues)

def test_empty_checks_rejected():
    t = MB("A"); t2 = t.model_copy(deep=True); t2.checks = {}
    issues = validate_plan(_plan(t2), [])
    assert any(code == "no_checks" for _, code, _ in issues)

def test_risky_flagging_full_path_globs():
    p = flag_risky(_plan(MB("A", files=("db/migrations/0021_x.sql",)), MB("B")), 
                   ["db/migrations/**", "**/auth/**"])
    assert p.tasks[0].risky is True and p.tasks[0].risky_reasons == ["db/migrations/**"]
    assert p.tasks[1].risky is False

def test_load_dump_roundtrip(tmp_path: Path):
    p = _plan(MB("A")); f = tmp_path / "plan.yaml"; dump_plan(p, f)
    assert load_plan(f) == p

def test_invalid_yaml_shapes_rejected(tmp_path: Path):
    f = tmp_path / "plan.yaml"; f.write_text("tasks: 3\n")
    with pytest.raises(PlanValidationError):
        load_plan(f)
