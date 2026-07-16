"""Tests for project path normalization in artifact listing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from engram.retrieval.artifacts import _normalize_project_path


def test_normalize_project_path_is_stable() -> None:
    path = "/tmp/engram-followup-test"
    once = _normalize_project_path(path)
    twice = _normalize_project_path(path)
    assert once == twice
    assert once is not None


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/private/tmp").exists(),
    reason="macOS /tmp -> /private/tmp symlink only",
)
def test_normalize_project_path_resolves_tmp_symlink() -> None:
    left = _normalize_project_path("/tmp/engram-followup-test")
    right = _normalize_project_path("/private/tmp/engram-followup-test")
    assert left == right
