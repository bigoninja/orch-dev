"""Editing a config's role → model pins without costing the file its comments.

A ``--config`` file *replaces* ``orch.config.yaml`` rather than layering on it, so
the file that decides which model a role runs is whichever one a run is handed.
Those files carry the human's rationale in comments — the shopdash pilot's header
is 25 lines explaining exactly why each role is pinned where it is — so edits
round-trip through ``ruamel.yaml`` rather than re-serializing through
``yaml.safe_dump``, which would delete every one of them.

Validation is not re-implemented here: a mutated config is handed back to
:func:`orch.config.load_config`, and if the real validator rejects it the original
file is restored. There is exactly one definition of a valid config, and this
module cannot drift from it.
"""

from pathlib import Path

from ruamel.yaml import YAML, YAMLError

from orch.config import ROLE_NAMES, load_config
from orch.errors import ConfigError

#: Display order for ``orch models list`` — the pipeline's own order, not
#: alphabetical, because that is how a human reads a run. Anything not named here
#: is appended, so adding a role cannot silently drop it from the list.
DISPLAY_ORDER = ("planner", "implement", "fix", "repair", "review")


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    return yaml


def _check_role(role: str) -> None:
    if role not in ROLE_NAMES:
        raise ConfigError(f"unknown role {role!r}; expected one of {sorted(ROLE_NAMES)}")


def _load(path: Path):
    try:
        return _yaml().load(path.read_text(encoding="utf-8")) or {}
    except (OSError, YAMLError) as e:
        raise ConfigError(f"invalid config {path}: {e}") from e


def _dump(path: Path, data) -> None:
    from io import StringIO

    buffer = StringIO()
    _yaml().dump(data, buffer)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _commit(path: Path, data) -> None:
    """Write the edit, but never leave an invalid config on disk."""
    original = path.read_text(encoding="utf-8")
    _dump(path, data)
    try:
        load_config(path)
    except ConfigError:
        path.write_text(original, encoding="utf-8")
        raise


def roles_in_order() -> list[str]:
    """Every role, in :data:`DISPLAY_ORDER`, with any newcomer appended."""
    return [role for role in DISPLAY_ORDER if role in ROLE_NAMES] + sorted(
        ROLE_NAMES - set(DISPLAY_ORDER)
    )


def describe_models(path: Path) -> list[tuple[str, str, bool]]:
    """``(role, model, is_pinned)`` for every role — what ``list`` prints."""
    cfg = load_config(path)
    return [(role, cfg.model_for(role), role in cfg.role_models) for role in roles_in_order()]


def set_role_model(path: Path, role: str, model: str) -> None:
    """Pin ``role`` to ``model`` in the config at ``path``, comments intact."""
    _check_role(role)
    if not model:
        raise ConfigError("a role's model must be non-empty")
    data = _load(path)
    roles = data.get("role_models")
    if roles is None:
        data["role_models"] = {role: model}
    else:
        roles[role] = model
    _commit(path, data)


def unset_role_model(path: Path, role: str) -> None:
    """Drop ``role``'s pin so it falls back to the run default; idempotent."""
    _check_role(role)
    data = _load(path)
    roles = data.get("role_models")
    if roles is None or role not in roles:
        return
    del roles[role]
    _commit(path, data)
