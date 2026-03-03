"""Tests for LoCoMo benchmark adapter and metrics."""

from __future__ import annotations

import json

from engram.benchmark.locomo.adapter import (
    conversation_to_episodes,
    load_locomo_dataset,
    probes_to_queries,
)
from engram.benchmark.locomo.answer_composer import compose_answer
from engram.benchmark.locomo.metrics import exact_match, normalize_answer, token_f1


class TestNormalizeAnswer:
    def test_basic_normalization(self):
        assert normalize_answer("The  Quick Brown Fox!") == "quick brown fox"

    def test_removes_articles(self):
        assert normalize_answer("a cat and an apple") == "cat and apple"

    def test_removes_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Alice works at Google", "Alice works at Google") == 1.0

    def test_case_insensitive(self):
        assert exact_match("alice WORKS at google", "Alice works at Google") == 1.0

    def test_different(self):
        assert exact_match("Alice works at Meta", "Alice works at Google") == 0.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1("Alice works at Google", "Alice works at Google") == 1.0

    def test_partial_overlap(self):
        f1 = token_f1("Alice works at Google", "Alice works at Meta")
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        assert token_f1("cat dog", "bird fish") == 0.0

    def test_empty_both(self):
        assert token_f1("", "") == 1.0

    def test_empty_prediction(self):
        assert token_f1("", "hello world") == 0.0


class TestConversationToEpisodes:
    def test_basic_conversion(self):
        from engram.benchmark.locomo.adapter import LoCoMoConversation

        conv = LoCoMoConversation(
            conversation_id="conv_1",
            turns=[
                {"text": "Hello!", "speaker": "Alice"},
                {"text": "Hi there!", "speaker": "Bob"},
                {"text": "How are you?", "speaker": "Alice"},
            ],
            probes=[],
        )
        episodes = conversation_to_episodes(conv)
        assert len(episodes) == 3
        assert "Alice: Hello!" in episodes[0].content
        assert "Bob: Hi there!" in episodes[1].content
        assert episodes[0].source == "locomo:conv_1"

    def test_content_fallback(self):
        """Falls back to 'content' key if 'text' not present."""
        from engram.benchmark.locomo.adapter import LoCoMoConversation

        conv = LoCoMoConversation(
            conversation_id="conv_2",
            turns=[{"content": "Hello!"}],
            probes=[],
        )
        episodes = conversation_to_episodes(conv)
        assert len(episodes) == 1
        assert "Hello!" in episodes[0].content


class TestProbesToQueries:
    def test_basic_conversion(self):
        from engram.benchmark.locomo.adapter import LoCoMoProbe

        probes = [
            LoCoMoProbe("p1", "Where does Alice work?", "Google", "factual"),
            LoCoMoProbe("p2", "Who is Bob?", "Engineer", "identity"),
        ]
        queries = probes_to_queries(probes)
        assert len(queries) == 2
        assert queries[0] == ("Where does Alice work?", "Google", "factual")

    def test_filters_empty(self):
        from engram.benchmark.locomo.adapter import LoCoMoProbe

        probes = [
            LoCoMoProbe("p1", "", "answer", ""),  # empty question
            LoCoMoProbe("p2", "question", "", ""),  # empty answer
            LoCoMoProbe("p3", "valid?", "yes", ""),
        ]
        queries = probes_to_queries(probes)
        assert len(queries) == 1


class TestComposeAnswer:
    def test_basic_composition(self):
        summaries = ["Alice is an engineer.", "She works at Google."]
        result = compose_answer(summaries)
        assert "Alice is an engineer." in result
        assert "She works at Google." in result

    def test_truncation(self):
        summaries = ["A" * 100, "B" * 100, "C" * 100]
        result = compose_answer(summaries, max_length=200)
        assert len(result) <= 200

    def test_empty_summaries(self):
        assert compose_answer([]) == ""

    def test_max_three_summaries(self):
        summaries = ["One.", "Two.", "Three.", "Four."]
        result = compose_answer(summaries)
        assert "Four." not in result


class TestLoadLocomoDataset:
    def test_load_list_format(self, tmp_path):
        """Load LoCoMo in list-of-conversations format."""
        data = [
            {
                "conversation_id": "conv_1",
                "conversation": [
                    {"text": "Hello!", "speaker": "Alice"},
                    {"text": "Hi!", "speaker": "Bob"},
                ],
                "questions": [
                    {"question": "Who said hello?", "answer": "Alice"},
                ],
            },
        ]
        path = tmp_path / "locomo.json"
        path.write_text(json.dumps(data))

        convs = load_locomo_dataset(path)
        assert len(convs) == 1
        assert convs[0].conversation_id == "conv_1"
        assert len(convs[0].turns) == 2
        assert len(convs[0].probes) == 1
        assert convs[0].probes[0].question == "Who said hello?"

    def test_max_conversations(self, tmp_path):
        """Respects max_conversations limit."""
        data = [
            {"conversation_id": f"c{i}", "conversation": [], "questions": []} for i in range(10)
        ]
        path = tmp_path / "locomo.json"
        path.write_text(json.dumps(data))

        convs = load_locomo_dataset(path, max_conversations=3)
        assert len(convs) == 3
