"""Tests for the offline capture queue."""

from __future__ import annotations

import json
from pathlib import Path

from engram.utils.offline_queue import append_to_queue, drain_queue


class TestOfflineQueue:
    def test_append_and_drain(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        append_to_queue({"content": "hello", "source": "test"}, queue_path)
        append_to_queue({"content": "world", "source": "test"}, queue_path)

        entries = drain_queue(queue_path)
        assert len(entries) == 2
        assert entries[0]["content"] == "hello"
        assert entries[1]["content"] == "world"

        # Queue should be empty after drain
        assert not queue_path.exists()
        assert drain_queue(queue_path) == []

    def test_drain_empty(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "nonexistent.jsonl"
        assert drain_queue(queue_path) == []

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            json.dumps({"content": "good"}) + "\n"
            + "not valid json\n"
            + json.dumps({"content": "also good"}) + "\n"
        )

        entries = drain_queue(queue_path)
        assert len(entries) == 2
        assert entries[0]["content"] == "good"
        assert entries[1]["content"] == "also good"

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            "\n"
            + json.dumps({"content": "data"}) + "\n"
            + "\n"
        )

        entries = drain_queue(queue_path)
        assert len(entries) == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "deep" / "nested" / "queue.jsonl"
        append_to_queue({"content": "test"}, queue_path)
        assert queue_path.exists()

        entries = drain_queue(queue_path)
        assert len(entries) == 1
