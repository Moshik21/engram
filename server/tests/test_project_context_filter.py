"""The project filter keeps the whole repository's rows, whole-word matched."""

from __future__ import annotations

from pathlib import Path

from engram.retrieval.context_builder import _prefer_project_context_results


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "Engram"
    (root / ".git").mkdir(parents=True)
    (root / "server").mkdir()
    (root / "app").mkdir()
    return root


def _row(content: str) -> dict:
    return {
        "result_type": "episode",
        "episode": {"id": "x", "content": content, "source": "auto:response"},
    }


def test_subdirectory_tagged_rows_survive_the_project_filter(tmp_path: Path):
    root = _repo(tmp_path)
    rows = [
        _row("[assistant|Engram] a root-tagged answer"),
        _row("[assistant|server] the CI run was cancel-in-progress, not hung"),
        _row("[user|shielded-bid] unrelated project"),
    ]
    kept = _prefer_project_context_results(rows, project_path=str(root))
    assert [r["episode"]["content"][:18] for r in kept] == [
        "[assistant|Engram]",
        "[assistant|server]",
    ]


def test_short_names_match_whole_words_only(tmp_path: Path):
    root = _repo(tmp_path)
    rows = [_row("[user|Engram] ok"), _row("the application layer of another project")]
    kept = _prefer_project_context_results(rows, project_path=str(root))
    assert len(kept) == 1  # 'app' must not match 'application'


def test_no_project_match_falls_back_to_everything(tmp_path: Path):
    root = _repo(tmp_path)
    rows = [_row("[user|other] a"), _row("[user|other] b")]
    assert len(_prefer_project_context_results(rows, project_path=str(root))) == 2


def test_recall_surface_copy_of_the_filter_agrees(tmp_path: Path):
    from engram.retrieval.recall_surface import (
        _prefer_project_context_results as recall_filter,
    )

    root = _repo(tmp_path)
    rows = [
        _row("[assistant|Engram] a root-tagged answer"),
        _row("[assistant|server] the CI run was cancel-in-progress, not hung"),
        _row("[user|shielded-bid] unrelated project"),
        _row("the application layer of another project"),
    ]
    kept = recall_filter(rows, project_path=str(root))
    assert [r["episode"]["content"][:18] for r in kept] == [
        "[assistant|Engram]",
        "[assistant|server]",
    ]
