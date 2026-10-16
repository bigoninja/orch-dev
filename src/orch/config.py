from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orch.errors import ConfigError
from orch.gitops import is_git_repo

DEFAULT_RISKY_PATTERNS = [
    "db/migrations/**", "**/auth/**", "**/payments/**", "**/gateway/**",
    ".github/**", "**/package.json", "**/package-lock.json", "**/*.env*",
]

TEMPLATE_NOTE = "# orch configuration — see docs/superpowers/specs/2026-10-16-agent-dev-orchestrator-design.md\n"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_repo: Path
    base_ref: str = "main"
    integration_branch: str = "orch/integration"
    worktree_root: str = ".worktrees"
    model: str = "opencode-go/deepseek-v4.1-flash"
    concurrency: int = Field(3, ge=1, le=8)
    retries: int = Field(2, ge=0, le=5)
    timeout_minutes: int = Field(45, ge=5)
    env_passthrough: list[str] = Field(default_factory=list)
    risky_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_RISKY_PATTERNS))
    notify: str = "none"
    state_dir: str = "state"
    runs_dir: str = "runs"


def load_config(path: Path) -> Config:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        return Config.model_validate(raw)
    except (yaml.YAMLError, ValidationError, OSError) as e:
        raise ConfigError(f"invalid config {path}: {e}") from e


def init_config(target_repo: Path, out: Path) -> Config:
    if not target_repo.is_dir() or not is_git_repo(target_repo):
        raise ConfigError(f"target_repo is not a git repository: {target_repo}")
    cfg = Config(target_repo=target_repo.resolve())
    out.write_text(TEMPLATE_NOTE + yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return cfg
