"""Planner node, repo-fact gathering, the adapter factory, and ``orch plan``.

These are the Task 15 contracts: ``gather_repo_facts`` (labeled, deterministic,
missing files skipped), ``run_planner`` (writes ``plan.yaml`` only when the plan
validates, else ``plan.invalid.yaml`` + ``PlanValidationError``), the mock
adapter's own ``planner`` bucket, and the CLI's exit-code contract.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from orch.adapters.factory import factory
from orch.adapters.mock import MockAdapter
from orch.adapters.opencode import OpenCodeAdapter
from orch.artifacts import RunDir
from orch.cli import main
from orch.config import Config
from orch.errors import ModelUnavailableError, OrchError, PlanValidationError
from orch.nodes.planner import gather_repo_facts, run_planner
from orch.plan import load_plan

# --- fixtures ---------------------------------------------------------------


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    return repo


def _commit(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-m", "init"],
        check=True, capture_output=True,
    )


def _repo_with_facts(tmp_path: Path) -> Path:
    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("# Project\nA sample project.\n")
    (repo / "AGENTS.md").write_text("# Agents\nAlways run the checks.\n")
    (repo / "package.json").write_text(
        json.dumps({"name": "x", "scripts": {"test": "pytest -q", "lint": "ruff check ."}})
    )
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('hi')\n")
    _commit(repo)
    return repo


def _cfg(tmp_path: Path, repo: Path, **overrides) -> Config:
    data = {
        "target_repo": repo,
        "runs_dir": str(tmp_path / "runs"),
        "state_dir": str(tmp_path / "state"),
    }
    data.update(overrides)
    return Config(**data)


def _run_dir(tmp_path: Path, run_id: str = "run-1") -> RunDir:
    return RunDir.create(tmp_path / "runs", run_id)


def _mock_planner(tmp_path: Path, text: str) -> MockAdapter:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(yaml.safe_dump({"planner": [{"text": text}]}))
    return MockAdapter(scenario)


def _fence(body: str) -> str:
    return "```yaml\n" + body.strip() + "\n```\n"


VALID_PLAN = _fence(
    """
run_id: run-pr
brief: test brief
source: null
tasks:
  - id: T1
    title: Add health endpoint
    description: Add GET /healthz returning 200.
    depends_on: []
    files_allowed:
      - db/migrations/0001_init.sql
    acceptance_criteria:
      - GET /healthz returns 200
    checks:
      test: {cmd: "pytest -q"}
"""
)

CYCLE_PLAN = _fence(
    """
run_id: run-cyc
brief: test brief
source: null
tasks:
  - id: A
    title: A
    description: first
    depends_on: [B]
    files_allowed: [a.py]
    acceptance_criteria: [x]
    checks:
      test: {cmd: "true"}
  - id: B
    title: B
    description: second
    depends_on: [A]
    files_allowed: [b.py]
    acceptance_criteria: [x]
    checks:
      test: {cmd: "true"}
"""
)


# --- (a) valid plan is written and risky-flagged in the file ----------------


def test_valid_plan_is_written_and_risky_flagged(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg = _cfg(tmp_path, repo, risky_patterns=["db/migrations/**"])
    run_dir = _run_dir(tmp_path)
    adapter = _mock_planner(tmp_path, VALID_PLAN)

    plan = run_planner(
        cfg=cfg, adapter=adapter, brief="add health", source=None, run_dir=run_dir
    )

    assert run_dir.plan_path.exists()
    written = load_plan(run_dir.plan_path)
    assert written == plan
    task = written.tasks[0]
    assert task.risky is True
    assert task.risky_reasons == ["db/migrations/**"]
    assert not (run_dir.root / "plan.invalid.yaml").exists()


# --- (b) a semantically invalid plan (cycle) is never written as plan.yaml --


def test_invalid_cycle_writes_invalid_not_plan(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg = _cfg(tmp_path, repo)
    run_dir = _run_dir(tmp_path)
    adapter = _mock_planner(tmp_path, CYCLE_PLAN)

    with pytest.raises(PlanValidationError) as excinfo:
        run_planner(cfg=cfg, adapter=adapter, brief="b", source=None, run_dir=run_dir)

    invalid = run_dir.root / "plan.invalid.yaml"
    assert invalid.exists()
    assert not run_dir.plan_path.exists()
    # The invalid file holds the parsed-and-flagged plan, so it still loads.
    parsed = load_plan(invalid)
    assert {t.id for t in parsed.tasks} == {"A", "B"}
    assert any(i.startswith("-: cycle:") for i in excinfo.value.issues)


# --- (c) an unparseable reply preserves the raw text ------------------------


def test_parse_failure_preserves_raw_reply(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg = _cfg(tmp_path, repo)
    run_dir = _run_dir(tmp_path)
    raw = "I could not produce a plan, sorry."
    adapter = _mock_planner(tmp_path, raw)

    with pytest.raises(PlanValidationError):
        run_planner(cfg=cfg, adapter=adapter, brief="b", source=None, run_dir=run_dir)

    assert (run_dir.root / "plan.invalid.yaml").read_text() == raw
    assert not run_dir.plan_path.exists()


def test_source_document_is_read_into_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``--from`` reads the file and threads it into the planner prompt."""
    repo = _git_repo(tmp_path)
    cfg = _cfg(tmp_path, repo)
    run_dir = _run_dir(tmp_path)
    source = tmp_path / "source.md"
    source.write_text("SOURCE-MARKER")
    captured: dict = {}

    adapter = _mock_planner(tmp_path, VALID_PLAN)
    real_run = adapter.run

    def spy(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return real_run(**kwargs)

    adapter.run = spy  # type: ignore[method-assign]
    run_planner(cfg=cfg, adapter=adapter, brief="b", source=str(source), run_dir=run_dir)
    assert "SOURCE-MARKER" in captured["prompt"]


# --- (d) planner model error is an OrchError the CLI renders ---------------


class _ModelDownAdapter:
    """A stand-in adapter whose every call fails like an unusable pinned model."""

    def run(self, **_kwargs):
        raise ModelUnavailableError("pinned model 'x' is unavailable: unauthorized")


def test_run_planner_wraps_a_model_error_as_orch_error(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg = _cfg(tmp_path, repo)
    run_dir = _run_dir(tmp_path)

    with pytest.raises(OrchError, match="planner model unavailable"):
        run_planner(
            cfg=cfg,
            adapter=_ModelDownAdapter(),
            brief="b",
            source=None,
            run_dir=run_dir,
        )

    assert not run_dir.plan_path.exists()


def test_gather_repo_facts_labels_and_orders_sections(tmp_path: Path):
    repo = _repo_with_facts(tmp_path)
    facts = gather_repo_facts(repo)

    assert "## README.md" in facts and "A sample project." in facts
    assert "## AGENTS.md" in facts and "Always run the checks." in facts
    assert "## git ls-files" in facts and "src/app.py" in facts
    assert "## package.json scripts" in facts
    assert "test: pytest -q" in facts and "lint: ruff check ." in facts

    order = [
        facts.index("## README.md"),
        facts.index("## AGENTS.md"),
        facts.index("## git ls-files"),
        facts.index("## package.json scripts"),
    ]
    assert order == sorted(order)


def test_gather_repo_facts_skips_missing_files(tmp_path: Path):
    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("# only readme\n")
    _commit(repo)
    facts = gather_repo_facts(repo)
    assert "## README.md" in facts
    assert "## AGENTS.md" not in facts
    assert "## package.json scripts" not in facts


def test_gather_repo_facts_heads_long_files(tmp_path: Path):
    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("\n".join(f"line {i}" for i in range(200)) + "\n")
    _commit(repo)
    facts = gather_repo_facts(repo)
    assert "line 79" in facts
    assert "line 80" not in facts


# --- (e) the mock adapter's own planner bucket ------------------------------


def test_mock_bucket_maps_planner_to_its_own_key():
    assert MockAdapter._bucket("planner", "planner") == "planner"
    assert MockAdapter._bucket("planner", None) == "planner"
    assert MockAdapter._bucket("implement", "planner") == "planner"
    # A review still wins, and fix phases still share the implement bucket.
    assert MockAdapter._bucket("review", "review") == "review"
    assert MockAdapter._bucket("fix", "fix1") == "implement"


def test_mock_planner_role_returns_its_own_text(tmp_path: Path):
    from orch.adapters.base import AgentRunResult

    adapter = _mock_planner(tmp_path, "PLANNER-TEXT")
    result = adapter.run(
        role="planner", prompt="p", cwd=tmp_path, session_id=None,
        title="run-1/planner", env={},
    )
    assert isinstance(result, AgentRunResult)
    assert result.text == "PLANNER-TEXT"


# --- factory ----------------------------------------------------------------


def test_factory_returns_opencode_adapter_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ORCH_ADAPTER", raising=False)
    cfg = Config(target_repo=Path("."), timeout_minutes=5)
    adapter = factory(cfg)
    assert isinstance(adapter, OpenCodeAdapter)
    assert adapter.model == cfg.model
    assert adapter.timeout_s == 5 * 60


def test_factory_returns_mock_adapter_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("planner: []\n")
    monkeypatch.setenv("ORCH_ADAPTER", "mock")
    monkeypatch.setenv("ORCH_MOCK_SCENARIO", str(scenario))
    adapter = factory(Config(target_repo=Path(".")))
    assert isinstance(adapter, MockAdapter)


# --- CLI `orch plan` --------------------------------------------------------


def _write_config(tmp_path: Path, repo: Path, **overrides) -> Path:
    data = {
        "target_repo": str(repo),
        "runs_dir": str(tmp_path / "runs"),
        "state_dir": str(tmp_path / "state"),
    }
    data.update(overrides)
    cfg_file = tmp_path / "orch.config.yaml"
    cfg_file.write_text(yaml.safe_dump(data))
    return cfg_file


def _set_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> None:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(yaml.safe_dump({"planner": [{"text": text}]}))
    monkeypatch.setenv("ORCH_ADAPTER", "mock")
    monkeypatch.setenv("ORCH_MOCK_SCENARIO", str(scenario))


def test_cli_plan_writes_plan_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    repo = _git_repo(tmp_path)
    cfg_file = _write_config(tmp_path, repo)
    _set_mock(monkeypatch, tmp_path, VALID_PLAN)

    rc = main(["plan", "add a health endpoint", "--config", str(cfg_file), "--run", "run-cli"])
    out = capsys.readouterr().out

    assert rc == 0
    plan_path = tmp_path / "runs" / "run-cli" / "plan.yaml"
    assert plan_path.exists()
    assert str(plan_path) in out
    assert "risky" in out.lower()


def test_cli_plan_reports_risky_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    repo = _git_repo(tmp_path)
    cfg_file = _write_config(tmp_path, repo, risky_patterns=["db/migrations/**"])
    _set_mock(monkeypatch, tmp_path, VALID_PLAN)

    rc = main(["plan", "b", "--config", str(cfg_file), "--run", "run-risky"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "T1" in out


def test_cli_plan_invalid_plan_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    repo = _git_repo(tmp_path)
    cfg_file = _write_config(tmp_path, repo)
    _set_mock(monkeypatch, tmp_path, CYCLE_PLAN)

    rc = main(["plan", "b", "--config", str(cfg_file), "--run", "run-bad"])
    out = capsys.readouterr().out

    assert rc == 1
    assert "plan.invalid.yaml" in out
    assert "cycle" in out
    assert not (tmp_path / "runs" / "run-bad" / "plan.yaml").exists()


def test_cli_plan_bad_config_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture):
    cfg_file = tmp_path / "missing.config.yaml"
    rc = main(["plan", "b", "--config", str(cfg_file)])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.startswith("error:")


def test_cli_plan_model_error_exits_one_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    repo = _git_repo(tmp_path)
    cfg_file = _write_config(tmp_path, repo)

    # The planner's adapter is built through the factory; force it to a model that
    # is unavailable, which must surface as a clean error, not an uncaught raise.
    monkeypatch.setattr(
        "orch.adapters.factory.factory", lambda _cfg: _ModelDownAdapter()
    )

    rc = main(["plan", "b", "--config", str(cfg_file), "--run", "run-down"])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "error:" in out
    assert "planner model unavailable" in out
