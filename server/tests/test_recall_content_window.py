"""Graduated content window on recall items (REST + MCP).

Measured 2026-09-04/06 on the fresh store: the fresh-agent battery counts the
full content of the top-3 items, 59,165 chars for 14/14 answers against
13,335 chars for 4/14 from repo files; the gate is <= ~27k. The answer tokens
sit next to the query terms, so a long row is cut to a window centred on them
and the cut is marked, never silent.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from engram.config import ActivationConfig
from engram.retrieval import recall_surface as recall_surface_module
from engram.retrieval.content_window import (
    ELLIPSIS,
    ContentWindow,
    content_window_for,
    query_terms,
    window_content,
)
from engram.retrieval.presenter import (
    present_api_recall_item,
    present_mcp_recall_item,
    recall_contract_item,
)
from engram.retrieval.recall_surface import build_api_recall_surface, build_mcp_recall_surface

QUERY = "what is the flip condition for usage ranking"
ANSWER = "The flip condition for usage ranking is E equals A plus a latency spot-check."
FILLER = "The harbour log notes the tide, the gulls, and the ferry timetable again. "


def _long_content(head_repeats: int = 30, tail_repeats: int = 30) -> str:
    return FILLER * head_repeats + ANSWER + " " + FILLER * tail_repeats


def _inner(text: str) -> str:
    return text.removeprefix(f"{ELLIPSIS} ").removesuffix(f" {ELLIPSIS}")


def test_window_centres_on_the_query_term_cluster_not_the_head():
    content = _long_content()
    text, windowed = window_content(content, QUERY, window_chars=300)

    assert windowed is True
    assert "flip condition for usage ranking" in text
    assert "latency spot-check" in text
    assert len(text) <= 300 + 2 * (len(ELLIPSIS) + 1)
    # The head would have missed the answer: centring is load-bearing.
    assert "flip condition" not in content[:300]
    assert text.startswith(f"{ELLIPSIS} ")
    assert text.endswith(f" {ELLIPSIS}")


def test_content_at_or_under_the_window_is_untouched():
    short = FILLER * 2
    assert window_content(short, QUERY, window_chars=600) == (short, False)
    exact = "x" * 600
    assert window_content(exact, QUERY, window_chars=600) == (exact, False)


def test_zero_window_disables_the_cut():
    content = _long_content()
    assert window_content(content, QUERY, window_chars=0) == (content, False)


def test_no_match_falls_back_to_the_head_with_a_tail_marker_only():
    content = _long_content()
    text, windowed = window_content(content, "zzzzquasarflux xylofract", window_chars=200)

    assert windowed is True
    assert not text.startswith(ELLIPSIS)
    assert text.endswith(f" {ELLIPSIS}")
    assert content.startswith(_inner(text))


def test_cut_edges_land_on_whitespace_and_the_span_is_verbatim():
    content = _long_content()
    text, _ = window_content(content, QUERY, window_chars=300)
    inner = _inner(text)

    start = content.find(inner)
    assert start > 0
    assert content[start - 1].isspace()
    end = start + len(inner)
    assert end < len(content)
    assert content[end].isspace()


def test_distinct_terms_outrank_one_term_repeated():
    repeats = "usage usage usage usage usage usage usage usage. "
    content = FILLER * 20 + repeats * 6 + FILLER * 20 + ANSWER + " " + FILLER * 20
    text, _ = window_content(content, QUERY, window_chars=200)

    assert "flip condition" in text
    assert "usage usage usage" not in text


def test_query_terms_drop_glue_words_and_keep_order():
    assert query_terms(QUERY) == ["flip", "condition", "usage", "ranking"]
    assert query_terms("Why was Thompson sampling removed?") == ["thompson", "sampling", "removed"]


def test_config_default_is_on_and_stubs_get_the_default():
    assert ActivationConfig().recall_content_window_chars == 450
    assert content_window_for(SimpleNamespace(), "q").window_chars == 450
    assert content_window_for(SimpleNamespace(recall_content_window_chars=0), "q").window_chars == 0
    assert content_window_for(ActivationConfig(recall_content_window_chars=250), "q") == (
        ContentWindow(query="q", window_chars=250)
    )


def _raw_episode(content: str) -> dict:
    return {
        "result_type": "episode",
        "episode": {"id": "ep_long", "content": content, "source": "auto:hook"},
        "score": 0.7,
        "score_breakdown": {"semantic": 0.7},
        "linked_entities": [],
    }


def test_rest_item_marks_the_cut_with_full_chars_and_windowed():
    content = _long_content()
    window = ContentWindow(query=QUERY, window_chars=300)

    episode = present_api_recall_item(_raw_episode(content), content_window=window)["episode"]
    assert episode["windowed"] is True
    assert episode["fullChars"] == len(content)
    assert len(episode["content"]) < len(content)
    assert "flip condition for usage ranking" in episode["content"]

    short = present_api_recall_item(_raw_episode("short row"), content_window=window)["episode"]
    assert short == {**short, "content": "short row", "fullChars": 9, "windowed": False}


@pytest.mark.asyncio
async def test_mcp_item_marks_the_cut_with_full_chars_and_windowed():
    content = _long_content()
    item = await present_mcp_recall_item(
        _raw_episode(content),
        resolve_entity_name=AsyncMock(return_value=""),
        get_access_count=AsyncMock(return_value=0),
        content_window=ContentWindow(query=QUERY, window_chars=300),
    )
    assert item["windowed"] is True
    assert item["full_chars"] == len(content)
    assert "flip condition for usage ranking" in item["content"]
    assert len(item["content"]) < len(content)


def test_presenters_without_a_window_pass_content_through():
    content = _long_content()
    contract = recall_contract_item(_raw_episode(content))
    assert contract["content"] == content
    assert contract["windowed"] is False
    assert contract["full_chars"] == len(content)


def _manager(content: str, cfg: ActivationConfig) -> SimpleNamespace:
    return SimpleNamespace(
        _graph=SimpleNamespace(
            find_entities_exact_name=AsyncMock(return_value=[]),
            find_entity_candidates=AsyncMock(return_value=[]),
        ),
        recall=AsyncMock(return_value=[_raw_episode(content)]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        search_entities=AsyncMock(return_value={"entities": []}),
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(enabled=False, max_packets=0),
        get_memory_need_config=lambda: cfg,
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        get_recall_need_thresholds=Mock(return_value={}),
        get_last_near_miss_views=AsyncMock(return_value=[]),
        get_surprise_connection_views=AsyncMock(return_value=[]),
    )


def _cfg(window_chars: int) -> ActivationConfig:
    return ActivationConfig(
        recall_content_window_chars=window_chars,
        recall_budget_explicit_ms=2000,
        recall_fast_preflight_enabled=False,
        recall_packets_enabled=False,
    )


@pytest.mark.asyncio
async def test_rest_recall_surface_windows_items_and_feeds_the_echo_mask_the_same_text(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")
    surfaced = Mock()
    monkeypatch.setattr(recall_surface_module, "note_surfaced_texts_from_response", surfaced)
    content = _long_content()

    response = await build_api_recall_surface(
        _manager(content, _cfg(300)),
        group_id="default",
        query=QUERY,
        limit=3,
        operation_source="api_recall",
    )

    episode = response["items"][0]["episode"]
    assert episode["windowed"] is True
    assert episode["fullChars"] == len(content)
    assert "flip condition for usage ranking" in episode["content"]
    assert len(episode["content"]) < 400
    masked = surfaced.call_args.args[1]["results"][0]
    assert masked["content"] == episode["content"]


@pytest.mark.asyncio
async def test_mcp_recall_surface_windows_items(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")
    content = _long_content()
    cfg = _cfg(300)

    response = await build_mcp_recall_surface(
        _manager(content, cfg),
        group_id="default",
        query=QUERY,
        limit=3,
        cfg=cfg,
        resolve_entity_name=AsyncMock(return_value=""),
        get_access_count=AsyncMock(return_value=0),
    )

    item = response["results"][0]
    assert item["windowed"] is True
    assert item["full_chars"] == len(content)
    assert "flip condition for usage ranking" in item["content"]
    assert len(item["content"]) < 400


@pytest.mark.asyncio
async def test_rest_recall_surface_window_off_returns_full_content(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")
    content = _long_content()

    response = await build_api_recall_surface(
        _manager(content, _cfg(0)),
        group_id="default",
        query=QUERY,
        limit=3,
        operation_source="api_recall",
    )

    episode = response["items"][0]["episode"]
    assert episode["content"] == content
    assert episode["windowed"] is False
    assert episode["fullChars"] == len(content)


def test_spread_terms_stretch_the_window_within_the_cap():
    """Two distinct query terms ~900 chars apart: a 450 window would hold one; the
    stretched window (<= 3x) holds both; terms 2,000 apart fall back to one window."""
    from engram.retrieval.content_window import STRETCH_CAP, window_content

    filler = "lorem ipsum dolor sit amet " * 200
    near = "the cognitive_core_fixes plan " + filler[:850] + " is organised in waves " + filler
    text, cut = window_content(
        near, "which document holds the cognitive core fixes plan waves", window_chars=450
    )
    assert cut and "cognitive_core_fixes" in text and "waves" in text
    assert len(text) <= 450 * STRETCH_CAP + 100
    far = "the cognitive_core_fixes plan " + filler[:2000] + " is organised in waves " + filler
    text, cut = window_content(
        far, "which document holds the cognitive core fixes plan waves", window_chars=450
    )
    assert cut and len(text) <= 450 + 80  # not stretched past the cap
