"""Entity model for knowledge graph nodes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from engram.entity_dedup_policy import entity_identifier_facets
from engram.utils.dates import utc_now


class Entity(BaseModel):
    """A node in the knowledge graph representing a person, concept, project, etc."""

    id: str
    name: str
    entity_type: str
    summary: str | None = None
    attributes: dict | None = None
    group_id: str = "default"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None
    activation_current: float = 0.0
    access_count: int = 0
    last_accessed: datetime | None = None
    pii_detected: bool = False
    pii_categories: list[str] | None = None
    identity_core: bool = False
    # Maturation tier (episodic | transitional | semantic) and reconsolidation
    # count. Native Helix persists these as first-class columns (authoritative);
    # the SQLite lite backend has no columns for them, so attributes JSON
    # remains its store and the validator below resolves the field on read.
    mat_tier: str = "episodic"
    recon_count: int = 0
    lexical_regime: str | None = None
    canonical_identifier: str | None = None
    identifier_label: bool = False

    # Summary provenance — tracks which episodes shaped this entity's summary
    source_episode_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    evidence_span_start: datetime | None = None
    evidence_span_end: datetime | None = None

    @model_validator(mode="after")
    def _resolve_maturation_fields(self) -> Entity:
        """Fall back to attributes JSON for maturation fields.

        The field (native column) is authoritative when set; rows written
        before mat_tier was a first-class field — and all SQLite lite rows —
        carry the tier only in attributes JSON.
        """
        attrs = self.attributes or {}
        if self.mat_tier == "episodic" and attrs.get("mat_tier") in (
            "transitional",
            "semantic",
        ):
            self.mat_tier = attrs["mat_tier"]
        if self.recon_count == 0:
            raw = attrs.get("recon_count")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
                self.recon_count = int(raw)
        return self

    @model_validator(mode="after")
    def _derive_identifier_facets(self) -> Entity:
        facets = entity_identifier_facets(self.name)
        if self.lexical_regime is None:
            self.lexical_regime = str(facets["lexical_regime"])
        if self.canonical_identifier is None:
            self.canonical_identifier = facets["canonical_identifier"]  # type: ignore[assignment]
        if bool(facets["identifier_label"]):
            self.identifier_label = True
        return self
