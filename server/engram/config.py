"""Pydantic Settings configuration for Engram."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_FILES = (
    str(Path.home() / ".engram" / ".env"),  # global config
    str(_REPO_ROOT / ".env"),  # repo-root local config
    ".env",  # cwd override
)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "info"


class SQLiteConfig(BaseModel):
    path: str = "~/.engram/engram.db"
    wal_mode: bool = True


class FalkorDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 6380
    password: str = ""
    graph_name: str = "engram"
    ssl: bool = False
    ssl_ca_cert: str = ""


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6381/0"
    password: str = ""
    ssl_cert_reqs: str = ""  # "required" | "optional" | "none"


class HelixDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 6969
    grpc_port: int = 6970
    transport: str = "http"  # "http" | "grpc" | "native" | "auto"
    api_endpoint: str = ""  # for cloud deployments
    api_key: str = ""
    config_path: str = "helixdb-cfg"
    verbose: bool = False
    max_workers: int = 4
    data_dir: str = ""  # Native transport: LMDB data dir (~/.helix/engram-native)


class PostgreSQLConfig(BaseModel):
    dsn: str = ""  # postgresql://user:pass@host:5432/engram
    min_pool_size: int = 2
    max_pool_size: int = 10


class EmbeddingConfig(BaseModel):
    provider: str = "auto"  # "auto" | "gemini" | "voyage" | "local" | "noop"
    model: str = "voyage-4-lite"  # voyage model name
    gemini_model: str = "gemini-embedding-2-preview"
    local_model: str = "nomic-ai/nomic-embed-text-v1.5"  # fastembed model
    dimensions: int = 0  # 0 = use provider default (gemini=3072, voyage=1024, local=768)
    storage_dimensions: int = 0  # 0 = same as native dimension (no truncation)
    api_key: str = ""  # voyage API key
    batch_size: int = 64
    fts_weight: float = 0.3
    vec_weight: float = 0.7
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 50


class AuthConfig(BaseModel):
    enabled: bool = False
    bearer_token: str = ""
    default_group_id: str = "default"
    # OIDC (Clerk) settings
    oidc_enabled: bool = False
    oidc_issuer: str = ""  # https://clerk.your-app.com
    oidc_audience: str = ""  # engram-api
    oidc_group_claim: str = "org_id"  # JWT claim → group_id


class RateLimitConfig(BaseModel):
    enabled: bool = False
    observe_per_min: int = 100
    remember_per_min: int = 20
    recall_per_min: int = 60
    trigger_per_hour: int = 2
    chat_per_min: int = 10


class EncryptionConfig(BaseModel):
    enabled: bool = False
    master_key: str = ""


class CORSConfig(BaseModel):
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )
    production_origin: str = ""  # e.g. "https://engram-dashboard.vercel.app"


class ActivationConfig(BaseModel):
    """All tunable activation engine parameters."""

    model_config = {"extra": "forbid"}

    decay_exponent: float = Field(default=0.5, ge=0.1, le=1.0)
    min_age_seconds: float = Field(default=1.0, ge=0.01)
    max_history_size: int = Field(default=200, ge=10, le=10000)
    B_mid: float = -4.0
    B_scale: float = Field(default=1.7, gt=0.0)
    spread_max_hops: int = Field(default=2, ge=1, le=5)
    spread_decay_per_hop: float = Field(default=0.5, ge=0.1, le=1.0)
    spread_firing_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    traversal_min_edge_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    spread_energy_budget: float = Field(default=50.0, gt=0.0)
    weight_semantic: float = Field(default=0.40, ge=0.0, le=1.0)
    weight_activation: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_spreading: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_edge_proximity: float = Field(default=0.15, ge=0.0, le=1.0)
    seed_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    activation_ttl_days: int = Field(default=90, ge=1, le=365)
    exploration_weight: float = Field(default=0.05, ge=0.0, le=0.5)
    rediscovery_weight: float = Field(default=0.02, ge=0.0, le=0.5)
    rediscovery_halflife_days: float = Field(default=30.0, gt=0.0, le=365.0)
    retrieval_top_k: int = Field(default=50, ge=5, le=500)
    retrieval_top_n: int = Field(default=10, ge=1, le=100)

    # --- Spreading strategy ---
    spreading_strategy: str = Field(default="bfs", pattern="^(bfs|ppr|actr)$")

    # --- PPR parameters ---
    ppr_alpha: float = Field(default=0.15, ge=0.01, le=0.5)
    ppr_max_iterations: int = Field(default=20, ge=5, le=100)
    ppr_epsilon: float = Field(default=1e-6, gt=0.0)
    ppr_expansion_hops: int = Field(default=3, ge=1, le=6)

    # --- Thompson Sampling ---
    ts_enabled: bool = Field(default=True)
    ts_weight: float = Field(default=0.08, ge=0.0, le=0.5)
    ts_positive_increment: float = Field(default=1.0, gt=0.0)
    ts_negative_increment: float = Field(default=1.0, gt=0.0)

    # --- Fan-based spreading (ACT-R S_ji) ---
    fan_s_max: float = Field(default=3.5, gt=0.0, le=5.0)
    fan_s_min: float = Field(default=0.3, ge=0.0, le=2.0)

    # --- ACT-R spreading activation ---
    actr_total_w: float = Field(default=1.0, gt=0.0, le=3.0)
    actr_max_sources: int = Field(default=7, ge=1, le=20)

    # --- RRF fusion ---
    rrf_k: int = Field(default=60, ge=1, le=200)
    use_rrf: bool = Field(default=True)

    # --- Community-aware spreading ---
    community_spreading_enabled: bool = Field(default=True)
    community_bridge_boost: float = Field(default=1.5, ge=1.0, le=3.0)
    community_intra_dampen: float = Field(default=0.7, ge=0.1, le=1.0)
    community_stale_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    community_max_iterations: int = Field(default=10, ge=1, le=50)

    # --- Multi-pool candidate generation ---
    multi_pool_enabled: bool = Field(default=True)
    pool_search_limit: int = Field(default=30, ge=5, le=200)
    pool_activation_limit: int = Field(default=20, ge=5, le=100)
    pool_graph_seed_count: int = Field(default=10, ge=1, le=50)
    pool_graph_max_neighbors: int = Field(default=10, ge=1, le=50)
    pool_graph_limit: int = Field(default=20, ge=5, le=100)
    pool_wm_max_neighbors: int = Field(default=5, ge=1, le=20)
    pool_wm_limit: int = Field(default=15, ge=5, le=50)
    pool_total_limit: int = Field(default=80, ge=20, le=1000)

    # --- Entity query retrieval ---
    entity_query_retrieval_enabled: bool = Field(default=True)
    pool_entity_query_limit: int = Field(default=20, ge=1, le=100)

    # --- HyDE (Hypothetical Document Embedding) ---
    hyde_enabled: bool = Field(
        default=False,
        description="Use HyDE for vector search queries (requires LLM, disabled by default)",
    )
    hyde_model: str = Field(default="claude-haiku-4-5-20251001")

    # --- Graph query expansion (LLM-free HyDE alternative) ---
    graph_query_expansion_enabled: bool = Field(
        default=True,
        description="Expand queries using knowledge graph context (LLM-free HyDE alternative)",
    )
    template_reformulation_enabled: bool = Field(
        default=True,
        description="Convert questions to statement form for better embedding match",
    )

    # --- Re-ranker ---
    reranker_enabled: bool = Field(default=True)
    reranker_provider: str = Field(default="local", pattern="^(cohere|local|noop)$")
    reranker_local_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")
    reranker_top_n: int = Field(default=10, ge=1, le=50)

    # --- MMR diversity ---
    mmr_enabled: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- Implicit feedback ---
    feedback_enabled: bool = Field(default=True)
    feedback_ttl_days: int = Field(default=90, ge=1, le=365)

    # --- Context-gated spreading ---
    context_gating_enabled: bool = Field(default=True)
    context_gate_floor: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Structure-aware embeddings ---
    structure_aware_embeddings: bool = Field(default=True)
    predicate_natural_names: dict[str, str] = Field(
        default_factory=lambda: {
            "WORKS_AT": "works at",
            "EXPERT_IN": "expert in",
            "USES": "uses",
            "KNOWS": "knows",
            "MENTIONED_WITH": "mentioned with",
            "RELATED_TO": "related to",
            "CREATED": "created",
            "MENTORS": "mentors",
            "COLLABORATES_WITH": "collaborates with",
            "DEPENDS_ON": "depends on",
            "INTEGRATES_WITH": "integrates with",
            "BUILT_WITH": "built with",
            "LEADS": "leads",
            "RESEARCHES": "researches",
            "HEADQUARTERED_IN": "headquartered in",
            "LOCATED_IN": "located in",
            "DREAM_ASSOCIATED": "dream-associated with",
            "RECOVERING_FROM": "recovering from",
            "HAS_CONDITION": "has condition",
            "LIKES": "likes",
            "DISLIKES": "dislikes",
            "PREFERS": "prefers",
            "AIMS_FOR": "aims for",
            "REQUIRES": "requires",
            "STUDYING": "studying",
            "LED_TO": "led to",
            "CAUSED_BY": "caused by",
            "HAS_PART": "has part",
            "PARENT_OF": "parent of",
            "CHILD_OF": "child of",
            "TREATS": "treats",
            "TRIGGERED_BY": "triggered by",
        }
    )
    structure_max_relationships: int = Field(default=15, ge=1, le=50)

    # --- Working memory buffer ---
    working_memory_enabled: bool = Field(default=True)
    working_memory_capacity: int = Field(default=20, ge=5, le=100)
    working_memory_ttl_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    working_memory_seed_energy: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Episode retrieval ---
    episode_retrieval_enabled: bool = Field(default=True)
    episode_retrieval_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    episode_retrieval_max: int = Field(default=5, ge=0, le=20)
    recall_episode_content_limit: int = Field(
        default=15000,
        ge=0,
        le=50000,
        description="Max chars of episode content in recall results (0 = unlimited)",
    )
    recall_tier_aware_truncation_enabled: bool = Field(
        default=False,
        description=(
            "Vary truncation by memory tier: episodic=full,"
            " transitional=gist, semantic=summary"
        ),
    )
    recall_transitional_content_limit: int = Field(
        default=500,
        ge=0,
        le=50000,
        description="Max chars for transitional-tier episodes (0 = unlimited)",
    )
    recall_semantic_content_limit: int = Field(
        default=200,
        ge=0,
        le=50000,
        description="Max chars for semantic-tier episodes (0 = unlimited)",
    )

    # --- Entity-linked episode traversal ---
    entity_episode_traversal_enabled: bool = Field(
        default=True,
        description="Follow entity->episode graph links during recall",
    )
    entity_episode_max_entities: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max entities to traverse for linked episodes",
    )
    entity_episode_max_per_entity: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max episodes to fetch per traversed entity",
    )
    entity_episode_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Score weight for entity-linked episodes (multiplied by parent entity score)",
    )

    # --- Temporal contiguity ---
    temporal_contiguity_enabled: bool = Field(
        default=False,
        description="Boost adjacent episodes from the same session when one is recalled",
    )
    temporal_contiguity_max_adjacent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max adjacent episodes to fetch per recalled episode",
    )
    temporal_contiguity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Score weight for contiguous episodes (multiplied by parent episode score)",
    )

    # --- Temporal retrieval scoring ---
    temporal_retrieval_enabled: bool = Field(
        default=True,
        description="Apply recency/oldness boosts to episode scores based on temporal query cues",
    )
    recency_halflife_days: float = Field(
        default=30.0,
        gt=0.0,
        le=365.0,
        description="Half-life in days for recency exponential decay in temporal scoring",
    )

    # --- Chunk search ---
    chunk_search_enabled: bool = Field(
        default=True,
        description="Search episode chunks for sub-episode precision during recall",
    )
    chunk_topic_segmentation: bool = Field(
        default=True,
        description="Use embedding-based topic segmentation instead of size-based chunking",
    )
    chunk_topic_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Cosine similarity threshold below which a topic boundary is detected",
    )

    # --- Retrieval strategy ---
    retrieval_strategy: str = Field(
        default="passage_first",
        pattern="^(passage_first|entity_first|hybrid)$",
        description=(
            "Primary retrieval strategy: passage_first prioritises episodes/chunks, "
            "entity_first prioritises entity graph, hybrid balances both"
        ),
    )
    passage_first_entity_budget: int = Field(
        default=-1,
        ge=-1,
        le=100,
        description=(
            "Max entity slots in passage_first mode. "
            "-1 = use heuristic (min(3, top_n//3)), "
            "0 = no entities (all slots to episodes). "
            "Useful for benchmarks where entity summaries add noise."
        ),
    )

    cue_recall_enabled: bool = Field(
        default=False,
        description="Allow cue-backed latent episodes to participate in recall",
    )
    cue_recall_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    cue_recall_max: int = Field(default=2, ge=0, le=20)
    cue_recall_hit_threshold: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Promote cue-backed episodes to scheduled projection after this many hits",
    )
    cue_policy_learning_enabled: bool = Field(
        default=False,
        description="Adapt cue projection scheduling from recall feedback signals",
    )
    cue_policy_schedule_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Feedback policy score threshold for scheduling projection",
    )
    cue_policy_surface_weight: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Weight assigned to surfaced cue interactions",
    )
    cue_policy_select_weight: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Weight assigned to selected cue interactions",
    )
    cue_policy_use_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight assigned to used cue interactions",
    )
    cue_policy_near_miss_weight: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="Weight assigned to repeated cue near-miss interactions",
    )
    cue_policy_score_cap: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Upper bound for accumulated cue policy score",
    )
    cue_policy_source_boosts: dict[str, float] = Field(
        default_factory=lambda: {
            "remember": 0.20,
            "auto:bootstrap": 0.10,
        },
        description="Optional source-specific boosts for initial cue policy score",
    )
    cue_policy_discourse_boosts: dict[str, float] = Field(
        default_factory=lambda: {
            "world": 0.05,
            "hybrid": 0.02,
        },
        description="Optional discourse-class boosts for initial cue policy score",
    )

    # --- Typed edge weighting ---
    predicate_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "WORKS_AT": 0.8,
            "EXPERT_IN": 0.9,
            "USES": 0.6,
            "KNOWS": 0.5,
            "MENTIONED_WITH": 0.3,
            "RELATED_TO": 0.4,
            "CREATED": 0.7,
            "MENTORS": 0.7,
            "COLLABORATES_WITH": 0.6,
            "DEPENDS_ON": 0.7,
            "INTEGRATES_WITH": 0.5,
            "BUILT_WITH": 0.6,
            "LEADS": 0.8,
            "RESEARCHES": 0.8,
            "HEADQUARTERED_IN": 0.3,
            "DREAM_ASSOCIATED": 0.1,
            "RECOVERING_FROM": 0.8,
            "HAS_CONDITION": 0.85,
            "LIKES": 0.6,
            "DISLIKES": 0.6,
            "PREFERS": 0.7,
            "AIMS_FOR": 0.75,
            "REQUIRES": 0.8,
            "STUDYING": 0.65,
            "LED_TO": 0.7,
            "CAUSED_BY": 0.75,
            "HAS_PART": 0.65,
            "PARENT_OF": 0.8,
            "CHILD_OF": 0.8,
            "TREATS": 0.8,
            "TRIGGERED_BY": 0.9,
            "ENABLES": 0.75,
            "PREVENTS": 0.7,
        }
    )
    predicate_weight_default: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Memory consolidation ---
    consolidation_profile: str = Field(
        default="off",
        pattern="^(off|observe|conservative|standard)$",
    )
    integration_profile: str = Field(
        default="off",
        pattern="^(off|rework)$",
        description=(
            "Coherent preset for the reworked extraction/recall/consolidation loop. "
            "'rework' normalizes to consolidation_profile=standard, "
            "recall_profile=all, and enables the cue/projection rollout bundle."
        ),
    )
    consolidation_enabled: bool = Field(default=False)
    consolidation_interval_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0)
    consolidation_dry_run: bool = Field(default=True)

    # --- Three-tier scheduling ---
    consolidation_tiered_enabled: bool = Field(
        default=True,
        description="Run phases at different frequencies (hot/warm/cold tiers)",
    )
    consolidation_tier_hot_seconds: float = Field(
        default=900.0,
        ge=60.0,
        le=86400.0,
        description="Hot tier interval (triage): 15 min default",
    )
    consolidation_tier_warm_seconds: float = Field(
        default=7200.0,
        ge=300.0,
        le=86400.0,
        description=(
            "Warm tier interval (merge, infer, compact, mature, semanticize, "
            "reindex): 2 hours default"
        ),
    )
    consolidation_tier_cold_seconds: float = Field(
        default=21600.0,
        ge=1800.0,
        le=86400.0,
        description=(
            "Cold tier interval (replay, prune, schema, graph_embed, dream): 6 hours default"
        ),
    )
    consolidation_merge_threshold: float = Field(default=0.88, ge=0.5, le=1.0)
    consolidation_merge_max_per_cycle: int = Field(default=50, ge=1, le=500)
    consolidation_merge_require_same_type: bool = Field(default=True)
    consolidation_merge_block_size: int = Field(default=500, ge=50, le=5000)

    # --- Embedding-based merge (ANN candidate pre-filtering) ---
    consolidation_merge_use_embeddings: bool = Field(
        default=True,
        description="Use embedding cosine similarity for merge pre-filtering",
    )
    consolidation_merge_embedding_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Min cosine similarity to consider entities as merge candidates",
    )
    consolidation_merge_embedding_min_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Min fraction of entities with embeddings to use ANN approach",
    )

    # --- LLM-assisted merge (borderline candidates) ---
    consolidation_merge_llm_enabled: bool = Field(default=False)
    consolidation_merge_soft_threshold: float = Field(default=0.80, ge=0.5, le=1.0)
    consolidation_merge_llm_model: str = Field(default="claude-haiku-4-5-20251001")
    consolidation_merge_escalation_enabled: bool = Field(default=False)
    consolidation_merge_escalation_model: str = Field(default="claude-sonnet-4-6-20250514")
    consolidation_merge_ann_llm_max: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Max ANN semantic candidates to send to LLM merge judge per cycle",
    )

    # --- Multi-signal merge scorer (replaces LLM judge) ---
    consolidation_merge_multi_signal_enabled: bool = Field(
        default=True,
        description="Use multi-signal deterministic scorer instead of LLM for merge decisions",
    )
    consolidation_merge_auto_threshold: float = Field(
        default=0.82,
        ge=0.5,
        le=1.0,
        description="Confidence threshold for auto-merge in multi-signal scorer",
    )
    consolidation_merge_reject_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Confidence below this = auto-reject in multi-signal scorer",
    )
    consolidation_merge_structural_min_neighbors: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Min shared neighbors for structural merge candidate discovery",
    )
    consolidation_identifier_review_enabled: bool = Field(
        default=True,
        description=(
            "Persist quarantined review records for suspicious identifier-like merge "
            "candidates that were blocked by policy"
        ),
    )
    consolidation_identifier_review_min_similarity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Min raw name similarity before a blocked identifier pair is queued for review",
    )
    consolidation_identifier_review_max_per_cycle: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max suspicious identifier review records to persist per cycle",
    )

    consolidation_prune_activation_floor: float = Field(default=0.05, ge=0.0, le=0.5)
    consolidation_prune_min_age_days: int = Field(default=14, ge=1, le=365)
    consolidation_prune_min_access_count: int = Field(default=2, ge=0, le=100)
    consolidation_prune_max_per_cycle: int = Field(default=100, ge=1, le=1000)
    consolidation_infer_cooccurrence_min: int = Field(default=3, ge=2, le=20)
    consolidation_infer_confidence_floor: float = Field(default=0.6, ge=0.1, le=1.0)
    consolidation_infer_max_per_cycle: int = Field(default=50, ge=1, le=500)
    consolidation_infer_transitivity_enabled: bool = Field(default=False)
    consolidation_infer_transitive_predicates: list[str] = Field(
        default_factory=lambda: ["LOCATED_IN", "PART_OF"],
    )
    consolidation_infer_transitivity_decay: float = Field(default=0.8, ge=0.1, le=1.0)

    # --- Statistical scoring (Tier 2: PMI + tf-idf) ---
    consolidation_infer_pmi_enabled: bool = Field(default=False)
    consolidation_infer_pmi_min: float = Field(default=1.0, ge=0.0, le=10.0)
    consolidation_infer_tfidf_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- LLM validation (Tier 3) ---
    consolidation_infer_llm_enabled: bool = Field(default=False)
    consolidation_infer_llm_confidence_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    consolidation_infer_llm_max_per_cycle: int = Field(default=20, ge=1, le=100)
    consolidation_infer_llm_model: str = Field(default="claude-haiku-4-5-20251001")

    # --- LLM escalation (Sonnet re-validation of uncertain verdicts) ---
    consolidation_infer_escalation_enabled: bool = Field(default=False)
    consolidation_infer_escalation_model: str = Field(default="claude-sonnet-4-6-20250514")
    consolidation_infer_escalation_max_per_cycle: int = Field(default=5, ge=1, le=50)

    # --- Multi-signal infer validation (replaces LLM judge) ---
    consolidation_infer_auto_validation_enabled: bool = Field(
        default=True,
        description="Use multi-signal deterministic scorer instead of LLM for edge validation",
    )
    consolidation_infer_auto_approve_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Score threshold for auto-approving inferred edges",
    )
    consolidation_infer_auto_reject_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Score below this = auto-reject inferred edges",
    )

    # --- Distillation and calibration ---
    consolidation_distillation_enabled: bool = Field(
        default=True,
        description="Persist teacher/student examples from consolidation audit history",
    )
    consolidation_calibration_enabled: bool = Field(
        default=True,
        description="Compute rolling calibration snapshots from recent consolidation cycles",
    )
    consolidation_calibration_window_cycles: int = Field(
        default=25,
        ge=1,
        le=500,
        description="How many recent cycles to include in calibration summaries",
    )
    consolidation_calibration_min_examples: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Minimum labeled examples required before reporting calibration metrics",
    )
    consolidation_calibration_bins: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of confidence buckets for calibration summaries",
    )

    # --- Cross-encoder refinement (Tier 1) ---
    consolidation_cross_encoder_enabled: bool = Field(
        default=True,
        description="Use cross-encoder to refine uncertain merge/infer decisions",
    )

    consolidation_compaction_horizon_days: int = Field(default=90, ge=30, le=365)
    consolidation_compaction_keep_min: int = Field(default=10, ge=5, le=50)
    consolidation_compaction_logarithmic: bool = Field(default=True)

    # --- Episode replay ---
    consolidation_replay_enabled: bool = Field(default=False)
    consolidation_replay_max_per_cycle: int = Field(default=50, ge=1, le=500)
    consolidation_replay_window_hours: float = Field(default=24.0, ge=1.0, le=720.0)
    consolidation_replay_min_age_hours: float = Field(default=1.0, ge=0.0, le=48.0)
    consolidation_replay_vocab_linking_enabled: bool = Field(
        default=True,
        description="Scan episodes for exact entity name matches after deferred extraction",
    )

    # --- Proactive notifications ---
    notification_surfacing_enabled: bool = Field(default=False)
    notification_max_per_group: int = Field(default=200, ge=10, le=1000)
    notification_mcp_max_per_response: int = Field(default=3, ge=0, le=10)
    notification_mcp_max_surfaces: int = Field(default=2, ge=1, le=5)
    notification_temporal_enabled: bool = Field(default=False)
    notification_temporal_horizon_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0)
    notification_activation_spike_threshold: float = Field(default=0.3, ge=0.1, le=1.0)
    notification_dream_enabled: bool = Field(default=True)
    notification_merge_enabled: bool = Field(default=True)
    notification_schema_enabled: bool = Field(default=True)
    notification_maturation_enabled: bool = Field(default=True)

    # --- Reindex after consolidation ---
    consolidation_reindex_max_per_cycle: int = Field(default=200, ge=1, le=5000)

    # --- Dream spreading (offline consolidation) ---
    consolidation_dream_enabled: bool = Field(default=False)
    consolidation_dream_max_seeds: int = Field(default=20, ge=1, le=200)
    consolidation_dream_activation_floor: float = Field(default=0.15, ge=0.0, le=1.0)
    consolidation_dream_activation_ceiling: float = Field(default=0.75, ge=0.0, le=1.0)
    consolidation_dream_activation_midpoint: float = Field(default=0.40, ge=0.0, le=1.0)
    consolidation_dream_weight_increment: float = Field(default=0.05, ge=0.001, le=0.5)
    consolidation_dream_max_boost_per_edge: float = Field(default=0.15, ge=0.01, le=1.0)
    consolidation_dream_max_edge_weight: float = Field(default=3.0, ge=1.0, le=10.0)
    consolidation_dream_min_boost: float = Field(default=0.005, ge=0.0, le=0.1)

    # --- Dream LTD (Long-Term Depression) ---
    consolidation_dream_ltd_enabled: bool = Field(
        default=True,
        description="Decay unboosted edges during dream (LTD analog)",
    )
    consolidation_dream_ltd_decay: float = Field(
        default=0.005,
        ge=0.001,
        le=0.1,
        description="Weight subtracted per cycle from unboosted seed-connected edges",
    )
    consolidation_dream_ltd_min_weight: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Floor weight below which edges are not decayed",
    )

    # Dream LTD sweep for low-activation entities
    consolidation_dream_ltd_sweep_enabled: bool = False
    consolidation_dream_ltd_sweep_size: int = 50
    consolidation_dream_ltd_sweep_decay: float = 0.002

    # --- Dream associations (cross-domain creative connections) ---
    consolidation_dream_associations_enabled: bool = Field(default=False)
    consolidation_dream_assoc_max_per_cycle: int = Field(default=10, ge=1, le=100)
    consolidation_dream_assoc_min_surprise: float = Field(default=0.25, ge=0.0, le=1.0)
    consolidation_dream_assoc_ttl_days: int = Field(default=30, ge=1, le=365)
    consolidation_dream_assoc_weight: float = Field(default=0.1, ge=0.01, le=0.5)
    consolidation_dream_assoc_max_per_domain_pair: int = Field(default=3, ge=1, le=20)
    consolidation_dream_assoc_min_summary_len: int = Field(default=20, ge=0, le=200)
    consolidation_dream_assoc_structural_max_hops: int = Field(default=3, ge=1, le=6)
    consolidation_dream_assoc_max_duration_ms: int = Field(default=5000, ge=500, le=30000)
    consolidation_dream_assoc_top_n_per_domain: int = Field(default=20, ge=5, le=200)
    consolidation_dream_assoc_min_graph_embeddings: int = Field(
        default=10,
        ge=0,
        description="Min graph embeddings required for structural blending in dream associations",
    )

    # --- Pressure-based triggering ---
    consolidation_pressure_enabled: bool = Field(default=False)
    consolidation_pressure_threshold: float = Field(default=100.0, gt=0.0, le=10000.0)
    consolidation_pressure_weight_episode: float = Field(default=1.0, ge=0.0, le=100.0)
    consolidation_pressure_weight_entity: float = Field(default=0.5, ge=0.0, le=100.0)
    consolidation_pressure_weight_near_miss: float = Field(default=2.0, ge=0.0, le=100.0)
    consolidation_pressure_time_factor: float = Field(default=0.01, ge=0.0, le=1.0)
    consolidation_pressure_cooldown_seconds: float = Field(default=300.0, ge=30.0, le=86400.0)

    # --- Triage (phase 0) ---
    triage_enabled: bool = Field(default=False, description="Enable triage phase in consolidation")
    triage_extract_ratio: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Fraction of QUEUED episodes to extract (0.0-1.0)",
    )
    triage_min_score: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum score to consider for extraction",
    )

    # --- Triage personal narrative boost ---
    triage_personal_boost_enabled: bool = Field(
        default=True,
        description="Boost personal/emotional content in triage scoring",
    )
    triage_personal_boost: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Score boost for personal narrative content",
    )
    triage_personal_min_matches: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum personal keyword matches to trigger boost",
    )

    # --- Triage LLM judge ---
    triage_llm_judge_enabled: bool = Field(
        default=False,
        description="Use Haiku as triage judge (replaces heuristics)",
    )
    triage_llm_judge_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model for LLM triage judge",
    )

    # --- Multi-signal triage scorer (replaces LLM judge) ---
    triage_multi_signal_enabled: bool = Field(
        default=True,
        description="Use multi-signal scorer instead of LLM for triage",
    )
    triage_scorer_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "embedding_surprise": 0.25,
            "structural_extractability": 0.20,
            "entity_candidate_count": 0.15,
            "knowledge_gap": 0.10,
            "yield_prediction": 0.10,
            "emotional_salience": 0.10,
            "novelty": 0.05,
            "goal_boost": 0.05,
        },
        description="Signal weights for multi-signal triage scorer (must sum to ~1.0)",
    )
    triage_llm_escalation_enabled: bool = Field(
        default=True,
        description="Escalate borderline scores to LLM judge (only when multi-signal enabled)",
    )
    triage_llm_escalation_low: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Lower bound of borderline band for LLM escalation",
    )
    triage_llm_escalation_high: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Upper bound of borderline band for LLM escalation",
    )
    triage_llm_escalation_max_per_cycle: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Max episodes per triage cycle to escalate to LLM",
    )

    # --- Worker confidence routing ---
    worker_extract_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Worker extracts immediately above this score (no triage needed)",
    )
    worker_skip_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Worker skips below this score (no triage needed)",
    )

    # --- Progressive projection / cue layer ---
    cue_layer_enabled: bool = Field(
        default=False,
        description="Generate deterministic episode cues on store_episode",
    )
    cue_vector_index_enabled: bool = Field(
        default=True,
        description="Index cue text for vector search when cue layer is enabled",
    )
    targeted_projection_enabled: bool = Field(
        default=True,
        description="Allow long episodes to use targeted span selection before extraction",
    )
    projector_v2_enabled: bool = Field(
        default=True,
        description="Enable the progressive planner/projector pipeline for episode projection",
    )
    projection_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Max retryable projection attempts before dead-letter state",
    )
    projection_planner_enabled: bool = Field(
        default=True,
        description="Use deterministic span planning before extractor calls",
    )
    projection_span_target_chars: int = Field(
        default=1400,
        ge=200,
        le=8000,
        description="Target span size for projection planning",
    )
    projection_span_max_chars: int = Field(
        default=2200,
        ge=400,
        le=8000,
        description="Hard cap for an individual planned span",
    )
    projection_min_span_chars: int = Field(
        default=300,
        ge=50,
        le=4000,
        description="Minimum planned span size before emitting a chunk",
    )
    projection_span_budget: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of primary spans to select per projection",
    )
    projection_neighbor_span_radius: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of adjacent spans to include for context expansion",
    )

    # --- Identity core ---
    identity_core_enabled: bool = Field(
        default=True,
        description="Enable identity core protected entity subgraph",
    )
    identity_predicates: list[str] = Field(
        default_factory=lambda: [
            "FAMILY_OF",
            "PARENT_OF",
            "CHILD_OF",
            "SIBLING_OF",
            "MARRIED_TO",
            "PARTNER_OF",
            "LIVES_IN",
            "WORKS_AT",
        ],
        description="Relationship predicates that trigger identity core auto-detection",
    )

    # --- Cross-domain penalty (topic-aware spreading) ---
    cross_domain_penalty_enabled: bool = Field(
        default=False,
        description="Penalize spreading activation across topic domains",
    )
    cross_domain_penalty_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Multiplier applied when spreading crosses topic domains",
    )
    domain_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "personal": ["Person", "Event", "Emotion", "Goal", "Preference", "Habit", "Intention"],
            "technical": ["Technology", "Software", "Project", "Identifier"],
            "creative": ["CreativeWork", "Article"],
            "knowledge": ["Concept"],
            "health": ["HealthCondition", "BodyPart"],
            "spatial": ["Organization", "Location"],
        },
        description="Entity type to topic domain mapping for cross-domain penalty",
    )

    # --- Inhibitory spreading (Brain Architecture) ---
    inhibitory_spreading_enabled: bool = Field(
        default=False,
        description="Enable inhibitory spreading activation",
    )
    inhibit_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Strength of lateral inhibition",
    )
    inhibit_similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Min cosine similarity for lateral inhibition",
    )
    inhibit_max_seed_anchors: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max seed entities for inhibition anchoring",
    )
    inhibition_predicate_suppression: bool = Field(
        default=True,
        description="Suppress contradictory predicate groups",
    )

    # --- Goal-relevance gating (Brain Architecture) ---
    goal_priming_enabled: bool = Field(
        default=False,
        description="Enable goal-priming in retrieval and triage",
    )
    goal_priming_boost: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Spreading activation boost for goal neighbors",
    )
    goal_priming_activation_floor: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Min activation for a goal to be considered active",
    )
    goal_priming_max_goals: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max active goals to consider",
    )
    goal_priming_max_neighbors: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max 1-hop neighbors per goal",
    )
    goal_priming_cache_ttl_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="TTL for goal priming cache",
    )
    goal_triage_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Triage score boost for goal-relevant content",
    )
    goal_prune_protection: bool = Field(
        default=True,
        description="Protect goal-related entities from pruning",
    )

    # --- Emotional salience (Brain Architecture) ---
    emotional_salience_enabled: bool = Field(
        default=True,
        description="Enable emotional salience scoring (pure regex, no LLM)",
    )
    emotional_triage_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for emotional composite in triage scoring",
    )
    emotional_prune_resistance: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Activation floor boost for emotional entities",
    )
    emotional_retrieval_boost: float = Field(
        default=0.08,
        ge=0.0,
        le=0.5,
        description="Retrieval score boost scaled by emo_composite",
    )
    triage_personal_floor: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum triage score for personal content",
    )
    triage_personal_floor_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Emotional composite threshold to activate personal floor",
    )

    # --- State-dependent retrieval (Brain Architecture) ---
    state_dependent_retrieval_enabled: bool = Field(
        default=False,
        description="Enable cognitive state biasing in retrieval",
    )
    state_domain_weight: float = Field(
        default=0.06,
        ge=0.0,
        le=0.3,
        description="Weight for domain affinity bias",
    )
    state_arousal_match_weight: float = Field(
        default=0.04,
        ge=0.0,
        le=0.3,
        description="Weight for arousal level matching",
    )
    state_arousal_ema_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="EMA decay for session arousal tracking",
    )

    # --- Context tiers ---
    context_identity_budget: int = Field(
        default=200,
        description="Token budget for identity core tier in get_context",
    )
    context_project_budget: int = Field(
        default=400,
        description="Token budget for project context tier in get_context",
    )
    context_recency_budget: int = Field(
        default=400,
        description="Token budget for recent activity tier in get_context",
    )

    # --- Extraction provider ---
    extraction_provider: str = Field(
        default="narrow",
        pattern="^(auto|anthropic|ollama|narrow)$",
        description=(
            "Extraction backend: 'narrow' (default) is zero-cost deterministic. "
            "'auto' tries anthropic→ollama→narrow. "
            "'anthropic' uses Claude Haiku for richer extraction (requires API key)."
        ),
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model for extraction",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )

    # --- Briefing format ---
    briefing_enabled: bool = Field(
        default=True,
        description="Enable LLM briefing format for get_context",
    )
    briefing_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model for briefing synthesis",
    )
    briefing_cache_ttl_seconds: float = Field(
        default=300.0,
        description="TTL for briefing cache entries",
    )
    briefing_max_tokens: int = Field(
        default=300,
        description="Max tokens for briefing response",
    )

    # --- Recall profile (enables Wave 1-4 features) ---
    recall_profile: str = Field(
        default="off",
        pattern="^(off|wave1|wave2|wave3|wave4|all)$",
    )

    # --- Background episode worker ---
    worker_enabled: bool = Field(default=False, description="Enable background episode worker")

    # --- AutoRecall (piggyback on observe/remember) ---
    auto_recall_enabled: bool = Field(
        default=False,
        description="Enable auto-recall on observe/remember",
    )
    auto_recall_limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max entities returned",
    )
    auto_recall_min_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Min composite score",
    )
    auto_recall_cooldown_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Topic dedup cooldown",
    )
    auto_recall_max_per_minute: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Rate limit per minute",
    )
    auto_recall_on_observe: bool = Field(
        default=True,
        description="Run auto-recall on observe",
    )
    auto_recall_on_remember: bool = Field(
        default=True,
        description="Run auto-recall on remember",
    )
    auto_recall_on_tool_call: bool = Field(
        default=False,
        description="Piggyback auto-recall on read-oriented tool calls",
    )
    auto_recall_session_prime: bool = Field(
        default=True,
        description="Auto-prime context on first call",
    )
    auto_recall_session_prime_max_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Session prime token budget",
    )
    # --- recall_lite (fast entity-probe on every observe/remember) ---
    auto_recall_token_budget: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Approximate token budget for recall_lite context per turn (~4 chars/token)",
    )
    auto_recall_cache_ttl_seconds: float = Field(
        default=300.0,
        ge=30.0,
        le=1800.0,
        description="TTL for session entity cache in recall_lite",
    )
    auto_recall_level: str = Field(
        default="lite",
        description="Recall level: 'lite' (FTS5) or 'medium' (FTS5+embedding)",
    )
    recall_need_analyzer_enabled: bool = Field(
        default=False,
        description="Enable heuristic memory-need analysis before recall",
    )
    recall_need_graph_probe_enabled: bool = Field(
        default=False,
        description="Enable graph-grounded recall-need resonance probing",
    )
    recall_need_structural_enabled: bool = Field(
        default=False,
        description="Enable structural recall-need patterns in live gating",
    )
    recall_need_shift_enabled: bool = Field(
        default=False,
        description="Enable shift-aware recall-need scoring",
    )
    recall_need_impoverishment_enabled: bool = Field(
        default=False,
        description="Enable impoverishment-aware recall-need scoring",
    )
    recall_need_shift_shadow_only: bool = Field(
        default=True,
        description="Emit shift recall-need telemetry without affecting recall decisions",
    )
    recall_need_impoverishment_shadow_only: bool = Field(
        default=True,
        description="Emit impoverishment recall-need telemetry without affecting recall decisions",
    )
    recall_need_adaptive_thresholds_enabled: bool = Field(
        default=False,
        description="Adjust recall-need thresholds from rolling runtime outcomes",
    )
    recall_need_target_use_rate: float = Field(
        default=0.55,
        ge=0.1,
        le=1.0,
        description="Target fraction of recall triggers that lead to useful usage",
    )
    recall_need_threshold_window: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Rolling window size for recall-need runtime metrics",
    )
    recall_need_adaptive_min_samples: int = Field(
        default=30,
        ge=5,
        le=500,
        description="Minimum trigger samples before adaptive threshold changes apply",
    )
    recall_need_graph_override_enabled: bool = Field(
        default=False,
        description="Allow high-confidence graph resonance to trigger bounded recall override",
    )
    recall_need_graph_override_resonance_threshold: float = Field(
        default=0.72,
        ge=0.4,
        le=1.0,
        description="Minimum graph resonance required for graph-only override",
    )
    recall_need_post_response_safety_net_enabled: bool = Field(
        default=False,
        description=(
            "Retry knowledge-chat replies once when memory-needed turns get a generic response"
        ),
    )
    epistemic_routing_enabled: bool = Field(
        default=False,
        description="Enable epistemic routing across memory, artifacts, and runtime sources",
    )
    artifact_bootstrap_enabled: bool = Field(
        default=False,
        description="Enable project artifact bootstrapping for parity across surfaces",
    )
    artifact_recall_enabled: bool = Field(
        default=False,
        description="Allow artifact substrate participation in routed answers",
    )
    epistemic_runtime_executor_enabled: bool = Field(
        default=False,
        description="Enable runtime/config evidence executor for inspect/reconcile questions",
    )
    decision_graph_enabled: bool = Field(
        default=False,
        description="Materialize decision and artifact externalization semantics in the graph",
    )
    epistemic_reconcile_enabled: bool = Field(
        default=False,
        description="Enable multi-source reconciliation between memory and artifacts",
    )
    answer_contract_enabled: bool = Field(
        default=False,
        description="Enable answer-contract resolution on top of epistemic routing",
    )
    claim_state_modeling_enabled: bool = Field(
        default=False,
        description="Annotate routed evidence with deterministic claim-state labels",
    )
    artifact_bootstrap_stale_seconds: int = Field(
        default=86400,
        ge=300,
        le=604800,
        description="How long bootstrapped project artifacts stay fresh before refresh",
    )
    recall_telemetry_enabled: bool = Field(
        default=False,
        description="Emit structured recall telemetry events",
    )
    recall_usage_feedback_enabled: bool = Field(
        default=False,
        description="Enable surfaced/selected/used feedback semantics for recall",
    )
    recall_planner_enabled: bool = Field(
        default=False,
        description="Enable planner-driven multi-intent recall",
    )
    recall_planner_max_intents: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Max intents emitted by the recall planner",
    )
    recall_planner_subquery_limit: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Per-intent semantic search budget",
    )
    recall_packets_enabled: bool = Field(
        default=True,
        description="Return packetized memory alongside raw recall results",
    )
    recall_packet_auto_limit: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max packets for auto-surfaced recall",
    )
    recall_packet_explicit_limit: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Max packets for explicit recall surfaces",
    )
    recall_packet_chat_limit: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max packets for chat tool recall output",
    )

    # --- Relevance Confidence ---
    relevance_confidence_enabled: bool = Field(
        default=True,
        description="Compute embedding-based relevance confidence per result",
    )
    relevance_confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Min relevance to include in results (0 = return all)",
    )
    relevance_abstention_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Below this relevance → 'I don't remember'",
    )

    # --- Conversation Awareness (Wave 2) ---
    conv_context_enabled: bool = Field(
        default=False,
        description="Enable conversation-context-aware retrieval",
    )
    conv_fingerprint_enabled: bool = Field(
        default=False,
        description="Enable rolling EMA fingerprint",
    )
    conv_fingerprint_alpha: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="EMA decay per turn",
    )
    conv_multi_query_enabled: bool = Field(
        default=False,
        description="Enable multi-query decomposition",
    )
    conv_multi_query_turns: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Recent turns for topic sub-query",
    )
    conv_multi_query_top_entities: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Top session entities for entity sub-query",
    )

    # --- Query decomposition (temporal / multi-hop) ---
    query_decomposition_enabled: bool = Field(
        default=True,
        description="Decompose complex temporal/multi-hop queries into atomic sub-queries",
    )
    query_decomposition_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model used for LLM-based query decomposition",
    )

    conv_session_entity_seeds_enabled: bool = Field(
        default=False,
        description="Inject session entities as spreading seeds",
    )
    conv_session_entity_seed_energy: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Base seed energy for session entities",
    )
    conv_near_miss_enabled: bool = Field(
        default=False,
        description="Enable near-miss detection",
    )
    conv_near_miss_window: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of near-miss candidates to track",
    )
    conv_context_rerank_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=0.3,
        description="Weight for fingerprint context boost",
    )

    # --- Topic Shift Detection (Wave 3) ---
    conv_topic_shift_enabled: bool = Field(
        default=False,
        description="Enable topic shift detection in conversation",
    )
    conv_topic_shift_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold below which a topic shift is detected",
    )
    conv_topic_shift_recall_boost: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Number of recall hints on topic shift",
    )

    # --- Surprise Detection (Wave 3) ---
    surprise_detection_enabled: bool = Field(
        default=False,
        description="Enable surprise connection detection",
    )
    surprise_activation_floor: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Max activation for a neighbor to be considered a surprise",
    )
    surprise_dormancy_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Min days since last access for dormancy check",
    )
    surprise_edge_weight_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Min edge weight for surprise detection",
    )
    surprise_max_per_episode: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max surprise connections to cache per episode",
    )
    surprise_cache_ttl_seconds: float = Field(
        default=300.0,
        ge=30.0,
        le=3600.0,
        description="TTL for surprise cache entries",
    )

    # --- Retrieval Priming (Wave 3) ---
    retrieval_priming_enabled: bool = Field(
        default=False,
        description="Enable retrieval priming (boost 1-hop neighbors)",
    )
    retrieval_priming_top_n: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top results to prime from",
    )
    retrieval_priming_boost: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Base boost for primed entities",
    )
    retrieval_priming_ttl_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="TTL for priming boosts",
    )
    retrieval_priming_max_neighbors: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max neighbors to prime per result entity",
    )

    # --- Prospective Memory (Wave 4) ---
    prospective_memory_enabled: bool = Field(
        default=False,
        description="Enable prospective memory trigger matching",
    )
    prospective_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default cosine similarity threshold",
    )
    prospective_max_fires: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default max fires per intention",
    )
    prospective_ttl_days: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Default TTL for intentions",
    )
    prospective_max_per_episode: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max triggered intentions per episode",
    )
    prospective_activation_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default activation threshold for v2 intentions",
    )
    prospective_graph_embedded: bool = Field(
        default=True,
        description="Use v2 graph-embedded intentions (vs v1 flat table)",
    )
    prospective_cooldown_seconds: float = Field(
        default=300.0,
        ge=0.0,
        le=86400.0,
        description="Min seconds between fires for an intention",
    )
    prospective_warmth_levels: list[float] = Field(
        default_factory=lambda: [0.3, 0.6, 0.8],
        description="Warmth thresholds: [cool, warming, warm] (>=1.0 is hot)",
    )

    # --- Graph structural embeddings ---
    weight_graph_structural: float = Field(default=0.1, ge=0.0, le=1.0)

    # --- Graph embedding training control ---
    graph_embedding_retrain_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Fraction of entities that must be affected to trigger full retrain",
    )
    graph_embedding_stagger_transe: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Run TransE every Nth consolidation cycle",
    )
    graph_embedding_stagger_gnn: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Run GNN every Nth consolidation cycle",
    )

    # Node2Vec
    graph_embedding_node2vec_enabled: bool = Field(default=True)
    graph_embedding_node2vec_dimensions: int = Field(default=64, ge=16, le=256)
    graph_embedding_node2vec_walk_length: int = Field(default=20, ge=5, le=100)
    graph_embedding_node2vec_num_walks: int = Field(default=10, ge=1, le=50)
    graph_embedding_node2vec_p: float = Field(default=1.0, ge=0.1, le=10.0)
    graph_embedding_node2vec_q: float = Field(default=1.0, ge=0.1, le=10.0)
    graph_embedding_node2vec_window: int = Field(default=5, ge=2, le=15)
    graph_embedding_node2vec_epochs: int = Field(default=5, ge=1, le=20)
    graph_embedding_node2vec_min_entities: int = Field(default=50, ge=10)

    # TransE
    graph_embedding_transe_enabled: bool = Field(default=True)
    graph_embedding_transe_dimensions: int = Field(default=64, ge=16, le=256)
    graph_embedding_transe_margin: float = Field(default=1.0, ge=0.1, le=10.0)
    graph_embedding_transe_lr: float = Field(default=0.01, ge=0.001, le=0.1)
    graph_embedding_transe_epochs: int = Field(default=100, ge=10, le=1000)
    graph_embedding_transe_negative_samples: int = Field(default=10, ge=1, le=50)
    graph_embedding_transe_batch_size: int = Field(default=128, ge=32, le=1024)
    graph_embedding_transe_min_triples: int = Field(default=100, ge=20)

    # GNN (GraphSAGE)
    graph_embedding_gnn_enabled: bool = Field(default=True)
    graph_embedding_gnn_hidden_dim: int = Field(default=128, ge=32, le=512)
    graph_embedding_gnn_output_dim: int = Field(default=64, ge=16, le=256)
    graph_embedding_gnn_layers: int = Field(default=2, ge=1, le=4)
    graph_embedding_gnn_lr: float = Field(default=0.001, ge=0.0001, le=0.01)
    graph_embedding_gnn_epochs: int = Field(default=50, ge=5, le=500)
    graph_embedding_gnn_min_entities: int = Field(default=200, ge=50)

    # --- Memory maturation (Brain Architecture Phase 2A) ---
    memory_maturation_enabled: bool = False
    maturation_transitional_threshold: float = 0.42
    maturation_semantic_threshold: float = 0.70
    maturation_min_age_days: int = 7
    maturation_min_cycles: int = 5
    maturation_max_per_cycle: int = 50
    maturation_source_weight: float = 0.30
    maturation_temporal_weight: float = 0.25
    maturation_richness_weight: float = 0.25
    maturation_regularity_weight: float = 0.20
    # Differential decay
    decay_exponent_episodic: float = 0.5
    decay_exponent_semantic: float = 0.3
    # Episode transition
    episode_transition_enabled: bool = False
    episode_transitional_coverage: float = 0.50
    episode_semantic_coverage: float = 0.85
    episode_transitional_min_cycles: int = 2
    episode_semantic_min_cycles: int = 5
    episode_transition_max_per_cycle: int = 20
    # Prune interaction
    episodic_prune_age_days: int = 14
    semantic_prune_age_days: int = 180

    # --- Reconsolidation (Brain Architecture Phase 2B) ---
    reconsolidation_enabled: bool = False
    reconsolidation_window_seconds: float = 300.0
    reconsolidation_max_modifications: int = 3
    reconsolidation_max_entries: int = 50
    reconsolidation_overlap_threshold: float = 0.10

    # --- Schema Formation (Brain Architecture Phase 3) ---
    schema_formation_enabled: bool = False
    schema_min_instances: int = 5
    schema_min_edges: int = 2
    schema_max_per_cycle: int = 5
    schema_max_entities_scan: int = 500

    # --- Microglia (graph immune surveillance) ---
    microglia_enabled: bool = False
    microglia_tag_threshold: float = 0.5
    microglia_confirm_threshold: float = 0.4
    microglia_min_cycles_to_demote: int = 2
    microglia_max_demotions_per_cycle: int = 20
    microglia_scan_edges_per_cycle: int = 500
    microglia_scan_entities_per_cycle: int = 200

    # --- Evidence-based extraction (v2) ---
    evidence_extraction_enabled: bool = Field(
        default=True,
        description="Enable evidence-based extraction pipeline (v2) instead of LLM extraction",
    )
    evidence_commit_entity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for committing entity evidence",
    )
    evidence_commit_relationship_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for committing relationship evidence",
    )
    evidence_commit_attribute_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for committing attribute evidence",
    )
    evidence_commit_temporal_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for committing temporal evidence",
    )
    evidence_adaptive_thresholds: bool = Field(
        default=True,
        description="Adapt commit thresholds based on graph density",
    )
    evidence_store_deferred: bool = Field(
        default=True,
        description="Store deferred evidence for later adjudication",
    )
    evidence_client_proposals_enabled: bool = Field(
        default=False,
        description="Accept client-supplied entity/relationship proposals",
    )
    evidence_forced_commit_cycles: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Force-commit deferred evidence after N consolidation cycles",
    )
    edge_adjudication_enabled: bool = Field(
        default=True,
        description="Enable v3 edge adjudication for ambiguous evidence",
    )
    edge_adjudication_client_enabled: bool = Field(
        default=True,
        description="Expose client-assisted adjudication requests and tools",
    )
    edge_adjudication_server_enabled: bool = Field(
        default=False,
        description="Allow offline server-side LLM adjudication for unresolved requests",
    )
    edge_adjudication_server_model: str = Field(
        default="claude-sonnet-4-6-20250514",
        description="Anthropic model used for offline server adjudication",
    )
    edge_adjudication_server_max_per_cycle: int = Field(
        default=10,
        ge=0,
        le=500,
        description="Maximum server adjudication requests per consolidation cycle",
    )
    edge_adjudication_server_daily_budget: int = Field(
        default=50,
        ge=0,
        le=5000,
        description="Daily cap on server adjudication requests",
    )
    edge_adjudication_server_min_age_minutes: int = Field(
        default=10,
        ge=0,
        le=10080,
        description="Minimum age before a pending request is eligible for server adjudication",
    )
    edge_adjudication_request_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="How long adjudication requests remain open before expiring",
    )
    edge_adjudication_max_requests_per_episode: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum adjudication work items created from a single episode",
    )

    # --- GC-MMR (Wave 3) ---
    gc_mmr_enabled: bool = Field(
        default=False,
        description="Enable graph-connected MMR re-ranking",
    )
    gc_mmr_lambda_relevance: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Relevance weight in GC-MMR",
    )
    gc_mmr_lambda_diversity: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Diversity penalty weight in GC-MMR",
    )
    gc_mmr_lambda_connectivity: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Graph connectivity bonus weight in GC-MMR",
    )

    def model_post_init(self, __context: object) -> None:
        """Apply consolidation, recall, and integration profile presets."""
        profile = self.consolidation_profile
        recall_profile = self.recall_profile
        integration_profile = self.integration_profile

        def _set(field: str, value: object) -> None:
            object.__setattr__(self, field, value)

        if integration_profile == "rework":
            profile = "standard"
            recall_profile = "all"
            _set("consolidation_profile", profile)
            _set("recall_profile", recall_profile)

        if profile == "observe":
            _set("consolidation_enabled", True)
            _set("consolidation_dry_run", True)
            _set("consolidation_replay_enabled", True)
            _set("consolidation_dream_enabled", True)
            _set("consolidation_infer_pmi_enabled", True)
            _set("consolidation_dream_associations_enabled", True)
            _set("triage_enabled", True)
            _set("worker_enabled", True)
            _set("auto_recall_enabled", True)
        elif profile == "conservative":
            _set("consolidation_enabled", True)
            _set("consolidation_dry_run", False)
            _set("consolidation_merge_threshold", 0.92)
            _set("consolidation_prune_min_age_days", 30)
            _set("consolidation_replay_enabled", True)
            _set("consolidation_dream_enabled", True)
            _set("triage_enabled", True)
            _set("triage_extract_ratio", 0.25)
            _set("worker_enabled", True)
            _set("inhibitory_spreading_enabled", True)
            _set("goal_priming_enabled", True)
            _set("state_dependent_retrieval_enabled", True)
            _set("memory_maturation_enabled", True)
            _set("episode_transition_enabled", True)
            _set("notification_surfacing_enabled", True)
            _set("notification_temporal_enabled", True)
        elif profile == "standard":
            _set("consolidation_enabled", True)
            _set("consolidation_dry_run", False)
            _set("consolidation_replay_enabled", True)
            _set("consolidation_dream_enabled", True)
            _set("consolidation_infer_pmi_enabled", True)
            _set("consolidation_infer_transitivity_enabled", True)
            _set("consolidation_pressure_enabled", True)
            _set("triage_enabled", True)
            _set("triage_extract_ratio", 0.35)
            _set("worker_enabled", True)
            _set("auto_recall_enabled", True)
            _set("inhibitory_spreading_enabled", True)
            _set("goal_priming_enabled", True)
            _set("state_dependent_retrieval_enabled", True)
            # Multi-signal triage replaces LLM judge (zero API cost)
            _set("triage_multi_signal_enabled", True)
            _set("triage_llm_judge_enabled", False)
            _set("triage_llm_escalation_enabled", False)
            # Multi-signal scorers replace LLM judges (zero API cost)
            _set("consolidation_merge_multi_signal_enabled", True)
            _set("consolidation_infer_auto_validation_enabled", True)
            # LLM judges disabled when multi-signal is active (kept as opt-in fallback)
            _set("consolidation_merge_llm_enabled", False)
            _set("consolidation_merge_escalation_enabled", False)
            _set("consolidation_infer_llm_enabled", False)
            _set("consolidation_infer_escalation_enabled", False)
            _set("consolidation_dream_associations_enabled", True)
            _set("graph_embedding_node2vec_enabled", True)
            _set("weight_graph_structural", 0.1)
            _set("memory_maturation_enabled", True)
            _set("episode_transition_enabled", True)
            _set("reconsolidation_enabled", True)
            _set("schema_formation_enabled", True)
            _set("cross_domain_penalty_enabled", True)
            _set("consolidation_dream_ltd_sweep_enabled", True)
            _set("microglia_enabled", True)
            _set("notification_surfacing_enabled", True)
            _set("notification_temporal_enabled", True)

        # --- Recall profile presets (cumulative) ---
        rp = recall_profile
        if rp != "off":
            # Wave 1: AutoRecall
            _set("auto_recall_enabled", True)
            _set("auto_recall_on_observe", True)
            _set("auto_recall_on_remember", True)
            _set("auto_recall_session_prime", True)
            _set("recall_need_analyzer_enabled", True)
            _set("recall_need_structural_enabled", True)
            _set("recall_telemetry_enabled", True)
            _set("recall_usage_feedback_enabled", True)
            _set("auto_recall_on_tool_call", True)

        if rp in ("wave2", "wave3", "wave4", "all"):
            # Wave 2: Conversation Awareness
            _set("auto_recall_level", "medium")
            _set("recall_need_graph_probe_enabled", True)
            _set("conv_context_enabled", True)
            _set("conv_fingerprint_enabled", True)
            _set("conv_multi_query_enabled", True)
            _set("conv_session_entity_seeds_enabled", True)
            _set("conv_near_miss_enabled", True)
            _set("recall_planner_enabled", True)

        if rp in ("wave3", "wave4", "all"):
            # Wave 3: Proactive Intelligence
            _set("recall_need_shift_enabled", True)
            _set("recall_need_impoverishment_enabled", True)
            _set("recall_need_shift_shadow_only", False)
            _set("recall_need_impoverishment_shadow_only", False)
            _set("conv_topic_shift_enabled", True)
            _set("surprise_detection_enabled", True)
            _set("retrieval_priming_enabled", True)
            _set("gc_mmr_enabled", True)

        if rp in ("wave4", "all"):
            # Wave 4: Prospective Memory
            _set("prospective_memory_enabled", True)

        if integration_profile == "rework":
            _set("cue_layer_enabled", True)
            _set("cue_vector_index_enabled", True)
            _set("cue_recall_enabled", True)
            _set("cue_policy_learning_enabled", True)
            _set("targeted_projection_enabled", True)
            _set("projector_v2_enabled", True)
            _set("projection_planner_enabled", True)
            _set("recall_need_analyzer_enabled", True)
            _set("recall_need_graph_probe_enabled", True)
            _set("recall_need_structural_enabled", True)
            _set("recall_need_shift_enabled", True)
            _set("recall_need_impoverishment_enabled", True)
            _set("recall_need_shift_shadow_only", False)
            _set("recall_need_impoverishment_shadow_only", False)
            _set("recall_usage_feedback_enabled", True)
            _set("recall_planner_enabled", True)
            _set("epistemic_routing_enabled", True)
            _set("artifact_bootstrap_enabled", True)
            _set("artifact_recall_enabled", True)
            _set("epistemic_runtime_executor_enabled", True)
            _set("decision_graph_enabled", True)
            _set("epistemic_reconcile_enabled", True)
            _set("answer_contract_enabled", True)
            _set("claim_state_modeling_enabled", True)
            _set("auto_recall_on_tool_call", True)
            _set("memory_maturation_enabled", True)
            _set("episode_transition_enabled", True)

        # --- Guard: disable LLM features if no API key (only for profile-set) ---
        if profile in ("standard",) and not os.environ.get("ANTHROPIC_API_KEY"):
            _set("triage_llm_judge_enabled", False)
            _set("consolidation_infer_llm_enabled", False)
            _set("consolidation_infer_escalation_enabled", False)
            _set("consolidation_merge_llm_enabled", False)
            _set("consolidation_merge_escalation_enabled", False)


class EngramConfig(BaseSettings):
    """Root configuration for Engram. Supports env vars, .env files, and YAML."""

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_nested_delimiter="__",
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: Literal["lite", "full", "helix", "auto"] = "auto"
    default_group_id: str = "default"

    # Sub-configs
    server: ServerConfig = Field(default_factory=ServerConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    falkordb: FalkorDBConfig = Field(default_factory=FalkorDBConfig)
    helix: HelixDBConfig = Field(default_factory=HelixDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgres: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)

    def get_sqlite_path(self) -> Path:
        """Return expanded SQLite database path, creating parent dirs."""
        path = Path(self.sqlite.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
