"""``orch models`` — assigning a model to a role from the CLI.

A ``--config`` file *replaces* ``orch.config.yaml`` rather than layering on it, so
the file that matters is the one a run is given. Editing it must not cost the
human the comments that explain it: these configs carry the rationale for every
pin (the pilot's is 25 lines of it), so the writer round-trips rather than
re-serializes.
"""

import pytest

from orch.cli import main
from orch.config import load_config


BASE = """\
# orch pilot config — the header a human wrote
# second comment line
target_repo: /tmp/repo
model: vendor/default
role_models:
  # the planner runs here
  planner: vendor/plan
  implement: vendor/impl
"""


def _write(tmp_path, text: str = BASE):
    path = tmp_path / "orch.config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_set_writes_the_role_model(tmp_path, capsys):
    path = _write(tmp_path)

    rc = main(["models", "set", "implement", "vendor/fast", "--config", str(path)])

    assert rc == 0, capsys.readouterr().out
    assert load_config(path).role_models["implement"] == "vendor/fast"


def test_set_keeps_every_comment_and_the_untouched_roles(tmp_path, capsys):
    path = _write(tmp_path)

    main(["models", "set", "implement", "vendor/fast", "--config", str(path)])

    text = path.read_text(encoding="utf-8")
    assert "# orch pilot config — the header a human wrote" in text
    assert "# the planner runs here" in text
    assert load_config(path).role_models["planner"] == "vendor/plan"


def test_set_adds_a_role_that_was_not_listed(tmp_path, capsys):
    path = _write(tmp_path)

    rc = main(["models", "set", "review", "vendor/slow", "--config", str(path)])

    assert rc == 0, capsys.readouterr().out
    assert load_config(path).role_models["review"] == "vendor/slow"


def test_set_creates_the_role_models_block_when_it_is_absent(tmp_path, capsys):
    path = _write(tmp_path, "target_repo: /tmp/repo\nmodel: vendor/default\n")

    rc = main(["models", "set", "fix", "vendor/fast", "--config", str(path)])

    assert rc == 0, capsys.readouterr().out
    assert load_config(path).role_models["fix"] == "vendor/fast"


def test_set_refuses_an_unknown_role(tmp_path, capsys):
    path = _write(tmp_path)

    rc = main(["models", "set", "architect", "vendor/x", "--config", str(path)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "unknown role" in out
    assert "vendor/x" not in path.read_text(encoding="utf-8"), "nothing was written"


def test_set_refuses_a_blank_model(tmp_path, capsys):
    path = _write(tmp_path)

    rc = main(["models", "set", "implement", "", "--config", str(path)])

    assert rc == 1
    assert "non-empty" in capsys.readouterr().out


def test_unset_removes_one_role_and_keeps_the_rest(tmp_path, capsys):
    path = _write(tmp_path)

    rc = main(["models", "unset", "planner", "--config", str(path)])

    assert rc == 0, capsys.readouterr().out
    roles = load_config(path).role_models
    assert "planner" not in roles
    assert roles["implement"] == "vendor/impl"


def test_list_shows_each_role_and_where_its_model_comes_from(tmp_path, capsys):
    path = _write(tmp_path)

    rc = main(["models", "list", "--config", str(path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "planner" in out and "vendor/plan" in out
    # An unlisted role still runs — on the run default, and the list must say so.
    assert "review" in out and "vendor/default" in out
