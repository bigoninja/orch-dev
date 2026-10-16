"""Selects the agent adapter a run uses.

A real run always drives the pinned model through :class:`OpenCodeAdapter` — the
no-substitution law forbids swapping it. Tests (and only tests) opt into the
token-free :class:`MockAdapter` by exporting ``ORCH_ADAPTER=mock`` together with
``ORCH_MOCK_SCENARIO=<path>``; no other value selects the mock.
"""

import os
from pathlib import Path

from orch.adapters.base import AgentAdapter
from orch.adapters.mock import MockAdapter
from orch.adapters.opencode import OpenCodeAdapter
from orch.config import Config


def factory(cfg: Config) -> AgentAdapter:
    """Return the adapter for ``cfg``: the mock when the env asks for it, else real."""
    if os.environ.get("ORCH_ADAPTER") == "mock":
        return MockAdapter(Path(os.environ["ORCH_MOCK_SCENARIO"]))
    return OpenCodeAdapter(
        model=cfg.model,
        timeout_s=cfg.timeout_minutes * 60,
        extra_env={},
    )
