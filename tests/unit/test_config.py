from pathlib import Path
import subprocess
import pytest
from orch.config import Config, load_config, init_config, DEFAULT_RISKY_PATTERNS
from orch.errors import ConfigError


def _git_repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(r)], check=True, capture_output=True)
    return r


def test_load_roundtrip(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg_file = tmp_path / "orch.config.yaml"
    cfg = init_config(repo, cfg_file)
    assert cfg_file.exists()
    loaded = load_config(cfg_file)
    assert loaded == cfg
    assert loaded.target_repo == repo
    assert loaded.model == "opencode/deepseek-v4.1-flash-free"
    assert loaded.concurrency == 3 and loaded.retries == 2 and loaded.timeout_minutes == 45
    assert loaded.risky_patterns == DEFAULT_RISKY_PATTERNS


def test_unknown_field_rejected(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg_file = tmp_path / "orch.config.yaml"
    init_config(repo, cfg_file)
    cfg_file.write_text(cfg_file.read_text() + "\nbogus: 1\n")
    with pytest.raises(ConfigError):
        load_config(cfg_file)


def test_bad_concurrency_rejected(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg_file = tmp_path / "orch.config.yaml"
    init_config(repo, cfg_file)
    text = cfg_file.read_text().replace("concurrency: 3", "concurrency: 99")
    cfg_file.write_text(text)
    with pytest.raises(ConfigError):
        load_config(cfg_file)


def test_init_refuses_non_git_dir(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ConfigError):
        init_config(plain, tmp_path / "orch.config.yaml")


def test_override_config_loads(tmp_path: Path):
    repo = _git_repo(tmp_path)
    cfg_file = tmp_path / "orch.config.yaml"
    cfg_file.write_text(
        f"target_repo: {repo}\n"
        "base_ref: release/1\n"
        "integration_branch: orch/release-1\n"
        "env_passthrough:\n- A\n- B\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.target_repo == repo
    assert cfg.base_ref == "release/1"
    assert cfg.integration_branch == "orch/release-1"
    assert cfg.env_passthrough == ["A", "B"]
    # everything else inherits the config defaults
    assert cfg.model == "opencode/deepseek-v4.1-flash-free"
    assert cfg.concurrency == 3
    assert cfg.retries == 2
    assert cfg.timeout_minutes == 45
