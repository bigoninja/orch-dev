"""Unit tests for :func:`orch.adapters.factory` — the no-substitution law.

A real run always drives the pinned model: the factory hands back an
:class:`OpenCodeAdapter` whose ``model`` is exactly ``cfg.model``. The mock is
reachable only through the two test-only environment variables, and neither one
alone is enough.
"""

import pytest

from orch.adapters.factory import factory
from orch.adapters.mock import MockAdapter
from orch.adapters.opencode import OpenCodeAdapter
from orch.config import Config


def _cfg(tmp_path, **over) -> Config:
    return Config(target_repo=tmp_path, **over)


def test_factory_builds_the_real_adapter_with_the_pinned_model(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_ADAPTER", raising=False)
    monkeypatch.delenv("ORCH_MOCK_SCENARIO", raising=False)
    cfg = _cfg(tmp_path, model="vendor/pinned-model")

    adapter = factory(cfg)

    assert isinstance(adapter, OpenCodeAdapter)
    assert adapter.model == cfg.model
    assert adapter.timeout_s == cfg.timeout_minutes * 60


def test_factory_only_selects_the_mock_under_the_mock_env(tmp_path, monkeypatch):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("ORCH_ADAPTER", "mock")
    monkeypatch.setenv("ORCH_MOCK_SCENARIO", str(scenario))

    assert isinstance(factory(_cfg(tmp_path)), MockAdapter)


def test_factory_does_not_select_the_mock_on_the_scenario_var_alone(tmp_path, monkeypatch):
    """Only ``ORCH_ADAPTER=mock`` selects the mock; a stray scenario path is inert."""
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("ORCH_ADAPTER", raising=False)
    monkeypatch.setenv("ORCH_MOCK_SCENARIO", str(scenario))

    assert isinstance(factory(_cfg(tmp_path)), OpenCodeAdapter)


def test_factory_passes_role_models_through_to_the_adapter(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_ADAPTER", raising=False)
    monkeypatch.delenv("ORCH_MOCK_SCENARIO", raising=False)
    cfg = _cfg(
        tmp_path,
        model="vendor/pinned-model",
        role_models={"planner": "vendor/planner-model"},
    )

    adapter = factory(cfg)

    assert adapter.model == "vendor/pinned-model"
    assert adapter.role_models == {"planner": "vendor/planner-model"}
