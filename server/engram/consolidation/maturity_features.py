"""Shared maturity feature extraction and caching helpers."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from engram.config import ActivationConfig

_MATURITY_FEATURE_POLICY_VERSION = "maturity_features_v1"
_SCHEMA_TIME_WINDOW_DAYS = 7.0


def compute_maturity_components(
    episode_count: int,
    temporal_span_days: float,
    rel_diversity: int,
    access_intervals: list[float],
    cfg: ActivationConfig,
) -> dict[str, float]:
    """Compute transparent maturity component scores and their weighted total."""
    source_score = min(1.0, episode_count / 10.0)
    temporal_score = min(1.0, temporal_span_days / 90.0)
    richness_score = min(1.0, rel_diversity / 8.0)
    regularity_score = compute_access_regularity(access_intervals)
    total = (
        cfg.maturation_source_weight * source_score
        + cfg.maturation_temporal_weight * temporal_score
        + cfg.maturation_richness_weight * richness_score
        + cfg.maturation_regularity_weight * regularity_score
    )
    return {
        "source_score": round(source_score, 4),
        "temporal_score": round(temporal_score, 4),
        "richness_score": round(richness_score, 4),
        "regularity_score": round(regularity_score, 4),
        "maturity_score": round(total, 4),
    }


def compute_access_regularity(access_intervals: list[float]) -> float:
    """Return [0,1] regularity based on the coefficient of variation."""
    if len(access_intervals) < 2:
        return 0.0
    mean_interval = sum(access_intervals) / len(access_intervals)
    if mean_interval <= 0:
        return 0.0
    std_dev = (
        sum((interval - mean_interval) ** 2 for interval in access_intervals)
        / len(access_intervals)
    ) ** 0.5
    return float(max(0.0, 1.0 - (std_dev / mean_interval)))


def get_cached_maturity_features(
    entity: Any,
    context: Any | None = None,
) -> dict[str, Any] | None:
    """Load a maturity bundle from cycle context or persisted entity attrs."""
    if context is not None:
        cached = getattr(context, "maturity_feature_cache", {}).get(getattr(entity, "id", ""))
        if isinstance(cached, dict):
            return dict(cached)

    attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
    cached = attrs.get(_MATURITY_FEATURE_POLICY_VERSION)
    if isinstance(cached, dict):
        return dict(cached)
    return None


def maturity_bundle_changed(
    cached: dict[str, Any] | None,
    current: dict[str, Any],
) -> bool:
    """Compare persisted/current bundles while ignoring volatile timestamps."""
    if not isinstance(cached, dict):
        return True

    def _stable_view(bundle: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in bundle.items() if key != "computed_at"}

    return _stable_view(cached) != _stable_view(current)


async def extract_maturity_features(
    entity: Any,
    graph_store: Any,
    activation_store: Any,
    group_id: str,
    cfg: ActivationConfig,
    context: Any | None = None,
    *,
    prefer_cached: bool = False,
) -> dict[str, Any]:
    """Compute a store-independent maturity feature bundle for an entity."""
    if prefer_cached:
        cached = get_cached_maturity_features(entity, context)
        if cached is not None:
            return cached

    attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
    recon_count = attrs.get("recon_count", 0)
    if not isinstance(recon_count, (int, float)):
        recon_count = 0

    identity_core = bool(getattr(entity, "identity_core", False))
    if identity_core:
        episode_count = 0
        temporal_span_days = 0.0
        relationship_richness = 0
        components = {
            "source_score": 1.0,
            "temporal_score": 1.0,
            "richness_score": 1.0,
            "regularity_score": 1.0,
            "maturity_score": 1.0,
        }
    else:
        episode_count = await graph_store.get_entity_episode_count(entity.id, group_id)
        min_created_at, max_created_at = await graph_store.get_entity_temporal_span(
            entity.id,
            group_id,
        )
        temporal_span_days = _compute_temporal_span_days(min_created_at, max_created_at)
        relationship_types = await graph_store.get_entity_relationship_types(entity.id, group_id)
        relationship_richness = len(relationship_types)

        state = await activation_store.get_activation(entity.id)
        access_history = sorted(getattr(state, "access_history", []) or [])
        access_intervals = [
            access_history[i + 1] - access_history[i] for i in range(len(access_history) - 1)
        ]
        components = compute_maturity_components(
            episode_count,
            temporal_span_days,
            relationship_richness,
            access_intervals,
            cfg,
        )
        components["maturity_score"] = round(
            min(1.25, components["maturity_score"] + min(0.10, float(recon_count) * 0.03)),
            4,
        )

    bundle = {
        "policy_version": _MATURITY_FEATURE_POLICY_VERSION,
        "computed_at": round(time.time(), 3),
        "episode_count": int(episode_count),
        "temporal_span_days": round(float(temporal_span_days), 2),
        "relationship_richness": int(relationship_richness),
        "access_regularity": round(float(components["regularity_score"]), 4),
        "recon_count": int(recon_count),
        "identity_core": identity_core,
        "support_windows": _estimate_support_windows(episode_count, temporal_span_days),
        **components,
    }

    if context is not None:
        context.maturity_feature_cache[entity.id] = dict(bundle)
    return bundle


def _compute_temporal_span_days(
    min_created_at: str | None,
    max_created_at: str | None,
) -> float:
    if not min_created_at or not max_created_at:
        return 0.0
    try:
        min_dt = datetime.fromisoformat(min_created_at)
        max_dt = datetime.fromisoformat(max_created_at)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, (max_dt - min_dt).total_seconds() / 86400.0)


def _estimate_support_windows(episode_count: int, temporal_span_days: float) -> int:
    if episode_count <= 0:
        return 0
    if temporal_span_days <= 0:
        return 1
    return max(1, int(temporal_span_days // _SCHEMA_TIME_WINDOW_DAYS) + 1)
