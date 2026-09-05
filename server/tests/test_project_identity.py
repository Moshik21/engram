"""One repository is one project, on both sides of the scope multiplier."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_surface import _project_from_cwd
from engram.ingestion.project_identity import accepted_project_names, project_name, repo_root
from engram.retrieval.pipeline import _episode_project_multipliers

pytestmark = pytest.mark.asyncio


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "Engram"
    (root / ".git").mkdir(parents=True)
    (root / "server" / "engram").mkdir(parents=True)
    (root / "dashboard").mkdir()
    return root


def test_identity_is_the_repository_root_name(tmp_path: Path):
    root = _repo(tmp_path)
    assert repo_root(root / "server" / "engram") == root
    assert project_name(root / "server" / "engram") == "Engram"
    assert project_name(root) == "Engram"
    assert _project_from_cwd(str(root / "server")) == "Engram"
    assert project_name(tmp_path / "loose-dir") == "loose-dir"  # no repo: basename
    assert project_name(Path.home()) is None
    assert project_name("/") is None
    assert project_name(None) is None


def test_accepted_names_cover_legacy_subdirectory_tags(tmp_path: Path):
    root = _repo(tmp_path)
    names = accepted_project_names(root)
    assert {"engram", "server", "dashboard"} <= names
    assert "shielded-bid" not in names
    assert "server" in accepted_project_names(root / "server")


class _Store:
    def __init__(self, rows):
        self.rows = rows

    async def get_episode_by_id(self, episode_id, group_id):
        r = self.rows.get(episode_id)
        return (
            None
            if r is None
            else SimpleNamespace(id=episode_id, content=r[0], project=r[1], source=None)
        )


async def test_subdirectory_tagged_rows_are_not_demoted_for_their_own_repo(tmp_path: Path):
    root = _repo(tmp_path)
    cfg = ActivationConfig(
        recall_other_project_multiplier=0.5,
        recall_short_episode_floor_chars=0,
        recall_machinery_episode_multiplier=1.0,
    )
    store = _Store(
        {
            "server-tag": ("[assistant|server] the verdict is in " * 3, None),
            "engram-tag": ("[user|Engram] why was Thompson removed " * 3, None),
            "field-server": ("no header here " * 6, "server"),
            "other": ("[user|shielded-bid] unrelated " * 3, None),
        }
    )
    mult = await _episode_project_multipliers(
        store, "default", list(store.rows), str(root), cfg, None
    )
    assert mult == {"other": 0.5}
