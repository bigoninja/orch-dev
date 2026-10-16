"""Adapters: the boundary every LLM role runs behind."""

from orch.adapters.base import AgentAdapter, AgentRunResult
from orch.adapters.mock import MockAdapter

__all__ = ["AgentAdapter", "AgentRunResult", "MockAdapter"]
