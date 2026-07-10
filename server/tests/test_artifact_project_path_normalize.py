"""Tests for project path normalization in artifact listing."""

from __future__ import annotations

from engram.retrieval.artifacts import _normalize_project_path


def test_normalize_project_path_resolves_tmp_symlink() -> None:
    left = _normalize_project_path("/tmp/engram-followup-test")
    right = _normalize_project_path("/private/tmp/engram-followup-test")
    assert left == right
