"""Every process resolves the FastEmbed cache the way the launchd shell does."""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.embeddings import provider as emb_provider


def test_cache_path_comes_from_the_dotenv_chain_when_env_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    low = tmp_path / "repo.env"
    high = tmp_path / "install.env"
    low.write_text("FASTEMBED_CACHE_PATH=/from/repo\n")
    high.write_text("FASTEMBED_CACHE_PATH=/from/install\n")
    monkeypatch.setattr("engram.config.DEFAULT_ENV_FILES", (str(low), str(high)))
    assert emb_provider._dotenv_fastembed_cache_path() == "/from/install"  # last file wins
    monkeypatch.setattr("engram.config.DEFAULT_ENV_FILES", (str(tmp_path / "missing.env"),))
    assert emb_provider._dotenv_fastembed_cache_path() == ""


def test_real_env_var_still_outranks_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "install.env"
    env_file.write_text(f"FASTEMBED_CACHE_PATH={tmp_path / 'from-file'}\n")
    monkeypatch.setattr("engram.config.DEFAULT_ENV_FILES", (str(env_file),))
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "from-env"))
    assert emb_provider.default_fastembed_cache_dir() == str(tmp_path / "from-env")
    monkeypatch.delenv("FASTEMBED_CACHE_PATH")
    assert emb_provider.default_fastembed_cache_dir() == str(tmp_path / "from-file")
