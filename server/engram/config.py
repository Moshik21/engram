"""Pydantic Settings configuration for Engram."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6381/0"
    password: str = ""


class EmbeddingConfig(BaseModel):
    provider: str = "voyage"
    model: str = "voyage-3-lite"
    dimensions: int = 512
    api_key: str = ""
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


class EncryptionConfig(BaseModel):
    enabled: bool = False
    master_key: str = ""


class CORSConfig(BaseModel):
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )


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
    spread_energy_budget: float = Field(default=50.0, gt=0.0)
    weight_semantic: float = Field(default=0.40, ge=0.0, le=1.0)
    weight_activation: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_spreading: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_edge_proximity: float = Field(default=0.15, ge=0.0, le=1.0)
    seed_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    activation_ttl_days: int = Field(default=90, ge=1, le=365)
    exploration_threshold: int = Field(default=5, ge=1, le=100)
    exploration_weight: float = Field(default=0.05, ge=0.0, le=0.5)
    rediscovery_weight: float = Field(default=0.02, ge=0.0, le=0.5)
    rediscovery_halflife_days: float = Field(default=30.0, gt=0.0, le=365.0)
    retrieval_top_k: int = Field(default=50, ge=5, le=500)
    retrieval_top_n: int = Field(default=10, ge=1, le=100)

    # --- Spreading strategy ---
    spreading_strategy: str = Field(default="bfs", pattern="^(bfs|ppr)$")

    # --- PPR parameters ---
    ppr_alpha: float = Field(default=0.15, ge=0.01, le=0.5)
    ppr_max_iterations: int = Field(default=20, ge=5, le=100)
    ppr_epsilon: float = Field(default=1e-6, gt=0.0)
    ppr_expansion_hops: int = Field(default=3, ge=1, le=6)

    # --- Thompson Sampling ---
    ts_enabled: bool = Field(default=False)
    ts_weight: float = Field(default=0.08, ge=0.0, le=0.5)
    ts_positive_increment: float = Field(default=1.0, gt=0.0)
    ts_negative_increment: float = Field(default=1.0, gt=0.0)

    # --- Fan-based spreading (ACT-R S_ji) ---
    fan_s_max: float = Field(default=3.5, gt=0.0, le=5.0)
    fan_s_min: float = Field(default=0.3, ge=0.0, le=2.0)

    # --- RRF fusion ---
    rrf_k: int = Field(default=60, ge=1, le=200)
    use_rrf: bool = Field(default=True)

    # --- Community-aware spreading ---
    community_spreading_enabled: bool = Field(default=False)
    community_bridge_boost: float = Field(default=1.5, ge=1.0, le=3.0)
    community_intra_dampen: float = Field(default=0.7, ge=0.1, le=1.0)
    community_stale_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    community_max_iterations: int = Field(default=10, ge=1, le=50)

    # --- Multi-pool candidate generation ---
    multi_pool_enabled: bool = Field(default=False)
    pool_search_limit: int = Field(default=30, ge=5, le=200)
    pool_activation_limit: int = Field(default=20, ge=5, le=100)
    pool_graph_seed_count: int = Field(default=10, ge=1, le=50)
    pool_graph_max_neighbors: int = Field(default=10, ge=1, le=50)
    pool_graph_limit: int = Field(default=20, ge=5, le=100)
    pool_wm_max_neighbors: int = Field(default=5, ge=1, le=20)
    pool_wm_limit: int = Field(default=15, ge=5, le=50)
    pool_total_limit: int = Field(default=80, ge=20, le=500)

    # --- Re-ranker ---
    reranker_enabled: bool = Field(default=False)
    reranker_top_n: int = Field(default=10, ge=1, le=50)

    # --- MMR diversity ---
    mmr_enabled: bool = Field(default=False)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)

    # --- Implicit feedback ---
    feedback_enabled: bool = Field(default=False)
    feedback_ttl_days: int = Field(default=90, ge=1, le=365)

    # --- Context-gated spreading ---
    context_gating_enabled: bool = Field(default=False)
    context_gate_floor: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Structure-aware embeddings ---
    structure_aware_embeddings: bool = Field(default=True)
    predicate_natural_names: dict[str, str] = Field(default_factory=lambda: {
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
    })
    structure_max_relationships: int = Field(default=15, ge=1, le=50)

    # --- Working memory buffer ---
    working_memory_enabled: bool = Field(default=True)
    working_memory_capacity: int = Field(default=20, ge=5, le=100)
    working_memory_ttl_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    working_memory_seed_energy: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Episode retrieval ---
    episode_retrieval_enabled: bool = Field(default=True)
    episode_retrieval_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    episode_retrieval_max: int = Field(default=3, ge=0, le=20)

    # --- Typed edge weighting ---
    predicate_weights: dict[str, float] = Field(default_factory=lambda: {
        "WORKS_AT": 0.8, "EXPERT_IN": 0.9, "USES": 0.6, "KNOWS": 0.5,
        "MENTIONED_WITH": 0.3, "RELATED_TO": 0.4, "CREATED": 0.7,
        "MENTORS": 0.7, "COLLABORATES_WITH": 0.6, "DEPENDS_ON": 0.7,
        "INTEGRATES_WITH": 0.5, "BUILT_WITH": 0.6, "LEADS": 0.8,
        "RESEARCHES": 0.8, "HEADQUARTERED_IN": 0.3,
    })
    predicate_weight_default: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- Memory consolidation ---
    consolidation_enabled: bool = Field(default=False)
    consolidation_interval_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0)
    consolidation_dry_run: bool = Field(default=True)
    consolidation_merge_threshold: float = Field(default=0.88, ge=0.5, le=1.0)
    consolidation_merge_max_per_cycle: int = Field(default=50, ge=1, le=500)
    consolidation_merge_require_same_type: bool = Field(default=True)
    consolidation_merge_block_size: int = Field(default=500, ge=50, le=5000)
    consolidation_prune_activation_floor: float = Field(default=0.05, ge=0.0, le=0.5)
    consolidation_prune_min_age_days: int = Field(default=30, ge=1, le=365)
    consolidation_prune_min_access_count: int = Field(default=0, ge=0, le=100)
    consolidation_prune_max_per_cycle: int = Field(default=100, ge=1, le=1000)
    consolidation_infer_cooccurrence_min: int = Field(default=3, ge=2, le=20)
    consolidation_infer_confidence_floor: float = Field(default=0.6, ge=0.1, le=1.0)
    consolidation_infer_max_per_cycle: int = Field(default=50, ge=1, le=500)
    consolidation_infer_transitivity_enabled: bool = Field(default=False)
    consolidation_infer_transitive_predicates: list[str] = Field(
        default_factory=lambda: ["LOCATED_IN", "PART_OF"],
    )
    consolidation_infer_transitivity_decay: float = Field(default=0.8, ge=0.1, le=1.0)
    consolidation_compaction_horizon_days: int = Field(default=90, ge=30, le=365)
    consolidation_compaction_keep_min: int = Field(default=10, ge=5, le=50)
    consolidation_compaction_logarithmic: bool = Field(default=True)

    # --- Reindex after consolidation ---
    consolidation_reindex_max_per_cycle: int = Field(default=200, ge=1, le=5000)
    consolidation_reindex_batch_size: int = Field(default=32, ge=1, le=128)

    # --- Pressure-based triggering ---
    consolidation_pressure_enabled: bool = Field(default=False)
    consolidation_pressure_threshold: float = Field(default=100.0, gt=0.0, le=10000.0)
    consolidation_pressure_weight_episode: float = Field(default=1.0, ge=0.0, le=100.0)
    consolidation_pressure_weight_entity: float = Field(default=0.5, ge=0.0, le=100.0)
    consolidation_pressure_weight_near_miss: float = Field(default=2.0, ge=0.0, le=100.0)
    consolidation_pressure_time_factor: float = Field(default=0.01, ge=0.0, le=1.0)
    consolidation_pressure_cooldown_seconds: float = Field(default=300.0, ge=30.0, le=86400.0)


class EngramConfig(BaseSettings):
    """Root configuration for Engram. Supports env vars, .env files, and YAML."""

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: Literal["lite", "full", "auto"] = "auto"
    default_group_id: str = "default"

    # Sub-configs
    server: ServerConfig = Field(default_factory=ServerConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    falkordb: FalkorDBConfig = Field(default_factory=FalkorDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)

    def get_sqlite_path(self) -> Path:
        """Return expanded SQLite database path, creating parent dirs."""
        path = Path(self.sqlite.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
