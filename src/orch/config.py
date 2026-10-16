from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from orch.errors import ConfigError
from orch.gitops import is_git_repo

DEFAULT_RISKY_PATTERNS = [
    "db/migrations/**", "**/auth/**", "**/payments/**", "**/gateway/**",
    ".github/**", "**/package.json", "**/package-lock.json", "**/*.env*",
]

#: The adapter roles a run may pin a model to. ``repair`` is a repair task's
#: implement/fix turn (a repair re-enters the ordinary pipeline, so it has no
#: node of its own); ``review`` covers both ordinary and repair reviews.
ROLE_NAMES = frozenset({"planner", "implement", "review", "fix", "repair"})

TEMPLATE_NOTE = "# orch configuration — see README.md\n"


def resolve_role_model(model: str, role_models: dict[str, str], role: str) -> str:
    """The model pinned for ``role``: its override, else the run default.

    One implementation with two callers — the adapter (which pins it on the
    ``opencode run`` command line) and the pipeline nodes (which label the live
    activity channel) must never disagree about which model a role is running.
    """
    return role_models.get(role, model)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_repo: Path
    base_ref: str = "main"
    integration_branch: str = "orch/integration"
    worktree_root: str = ".worktrees"
    model: str = "opencode/deepseek-v4.1-flash-free"
    #: Optional per-role model overrides; an unlisted role uses ``model``. These
    #: are configured pins, not runtime substitutions: an unavailable model for
    #: any role still escalates rather than falling back to another.
    role_models: dict[str, str] = Field(default_factory=dict)
    concurrency: int = Field(3, ge=1, le=8)
    retries: int = Field(2, ge=0, le=5)
    timeout_minutes: int = Field(45, ge=5)
    # Extra vars force-exported to worker/check subprocesses. The child also
    # inherits the full process environment, so this documents intent — it does
    # NOT restrict. Never list PWD (the adapter owns it for worktree isolation).
    env_passthrough: list[str] = Field(default_factory=list)
    risky_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_RISKY_PATTERNS))
    notify: str = "none"
    state_dir: str = "state"
    runs_dir: str = "runs"

    @field_validator("role_models")
    @classmethod
    def _validate_role_models(cls, value: dict[str, str]) -> dict[str, str]:
        """Reject unknown roles and blank models early, at config load."""
        unknown = sorted(set(value) - ROLE_NAMES)
        if unknown:
            raise ValueError(
                f"unknown role(s) in role_models: {unknown}; "
                f"expected a subset of {sorted(ROLE_NAMES)}"
            )
        if any(not model for model in value.values()):
            raise ValueError("role_models values must be non-empty")
        return value

    def model_for(self, role: str) -> str:
        """The model this run pins for ``role`` (see :func:`resolve_role_model`)."""
        return resolve_role_model(self.model, self.role_models, role)


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
