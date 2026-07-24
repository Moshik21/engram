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
    # M4: the deep-pipeline wait_for ceiling (max_search_ms) is now the WALL, so the
    # serial substage caps fit under it. recall_budget_explicit_search_ms is no longer
    # the ceiling — it is only the primary-search substage floor (candidate_pool).
    assert explicit.max_search_ms == 900
    assert explicit.max_results == 7
    assert explicit.allow_deep_recall is True


def test_explicit_recall_deep_pipeline_ceiling_follows_wall() -> None:
    # Default (flag on): max_search_ms == the wall, decoupled from the search floor.
    cfg = ActivationConfig(
        recall_budget_explicit_ms=4000,
        recall_budget_explicit_search_ms=1500,
    )

    explicit = recall_budget_for_profile(cfg, "explicit", surface="rest")

    assert explicit.budget_ms == 4000
    assert explicit.max_search_ms == 4000

    # Kill switch restores the legacy sub-wall ceiling from the search-floor knob.
    legacy = recall_budget_for_profile(
        ActivationConfig(
            recall_budget_explicit_ms=4000,
            recall_budget_explicit_search_ms=1500,
            recall_deep_pipeline_wall_budget_enabled=False,
        ),
        "explicit",
        surface="rest",
    )
    assert legacy.max_search_ms == 1500


def test_explicit_deep_pipeline_substage_caps_fit_wait_for_ceiling() -> None:
    """M4 DoD: the serial substage caps must sum <= the deep-pipeline wait_for
    ceiling (max_search_ms). Otherwise wait_for(manager.recall, ceiling) cancels
    the pipeline mid-flight (during chunk/materialize) even when every substage is
    inside its own cap, discarding already-materialized candidates for an empty
    rescue."""
    cfg = ActivationConfig()
    explicit = recall_budget_for_profile(cfg, "explicit", surface="rest")

    # The serial answer-producing critical path (see budgets.py M4 comment).
    substage_caps_ms = (
        cfg.retrieval_stats_timeout_ms
        + cfg.retrieval_primary_search_timeout_ms
        + cfg.retrieval_episode_search_timeout_ms
        + cfg.retrieval_cue_search_timeout_ms
        + cfg.retrieval_chunk_search_timeout_ms
        + cfg.retrieval_spread_timeout_ms
        + cfg.recall_primary_materialize_graph_timeout_ms
    )

    assert substage_caps_ms == 3925
    assert substage_caps_ms <= explicit.max_search_ms
    # And the ceiling never exceeds the wall, so a slow query stays bounded.
    assert explicit.max_search_ms <= explicit.budget_ms


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
