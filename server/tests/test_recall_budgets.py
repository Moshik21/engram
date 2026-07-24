from __future__ import annotations

from engram.config import ActivationConfig
from engram.retrieval.budgets import (
    budget_profile_for_source,
    recall_budget_for_profile,
    surface_for_source,
)


def test_recall_budget_profiles_keep_startup_cache_only() -> None:
    cfg = ActivationConfig(
        recall_budget_startup_ms=125,
        recall_budget_auto_lite_ms=50,
        recall_budget_explicit_ms=900,
        auto_recall_token_budget=240,
    )

    startup = recall_budget_for_profile(cfg, "startup", surface="mcp")
    lite = recall_budget_for_profile(cfg, "auto_lite", surface="mcp", mode="medium")
    explicit = recall_budget_for_profile(cfg, "explicit", surface="rest", max_results=7)

    assert startup.budget_ms == 125
    assert startup.allow_cache_only is True
    assert startup.allow_deep_recall is False
    assert startup.allow_embeddings is False
    assert startup.budget_tokens == 240
    assert lite.budget_ms == 50
    assert lite.allow_deep_recall is False
    assert lite.allow_embeddings is True
    assert explicit.budget_ms == 900
    # Search default follows ActivationConfig.recall_budget_explicit_search_ms.
    assert explicit.max_search_ms == 2200
    assert explicit.max_results == 7
    assert explicit.allow_deep_recall is True


def test_explicit_recall_search_budget_is_configurable() -> None:
    cfg = ActivationConfig(
        recall_budget_explicit_ms=4000,
        recall_budget_explicit_search_ms=1500,
    )

    explicit = recall_budget_for_profile(cfg, "explicit", surface="rest")

    assert explicit.budget_ms == 4000
    assert explicit.max_search_ms == 1500


def test_recall_budget_source_and_surface_inference() -> None:
    assert budget_profile_for_source("mcp_session_prime") == "startup"
    assert budget_profile_for_source("recall_lite") == "auto_lite"
    assert budget_profile_for_source("auto_recall") == "auto_deep"
    assert budget_profile_for_source("chat_recall") == "chat"
    assert budget_profile_for_source("api_recall") == "explicit"
    assert surface_for_source("axi_home") == "axi"
    assert surface_for_source("mcp_auto_recall") == "mcp"
    assert surface_for_source("api_recall") == "rest"


def test_recall_budget_stage_timeout_uses_smaller_stage_budget() -> None:
    cfg = ActivationConfig(recall_budget_explicit_ms=100)
    budget = recall_budget_for_profile(cfg, "explicit", surface="rest")

    timeout = budget.stage_timeout_seconds(1)

    assert 0 <= timeout <= 0.001
