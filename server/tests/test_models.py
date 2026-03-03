"""Tests for data models."""

import pytest

from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship
from engram.models.tenant import TenantContext


class TestEntity:
    def test_create_entity(self):
        e = Entity(id="ent_1", name="Python", entity_type="Technology")
        assert e.id == "ent_1"
        assert e.name == "Python"
        assert e.group_id == "default"
        assert e.deleted_at is None

    def test_entity_with_attributes(self):
        e = Entity(
            id="ent_2",
            name="ReadyCheck",
            entity_type="Project",
            attributes={"status": "active"},
        )
        assert e.attributes == {"status": "active"}

    def test_entity_pii_defaults(self):
        e = Entity(id="ent_3", name="Test", entity_type="Test")
        assert e.pii_detected is False
        assert e.pii_categories is None

    def test_entity_with_pii_flags(self):
        e = Entity(
            id="ent_4",
            name="Jake",
            entity_type="Person",
            pii_detected=True,
            pii_categories=["phone", "email"],
        )
        assert e.pii_detected is True
        assert "phone" in e.pii_categories


class TestEpisode:
    def test_create_episode(self):
        ep = Episode(id="ep_1", content="Working on the project")
        assert ep.status == EpisodeStatus.PENDING
        assert ep.group_id == "default"

    def test_episode_status(self):
        ep = Episode(id="ep_2", content="test", status=EpisodeStatus.COMPLETED)
        assert ep.status == EpisodeStatus.COMPLETED

    def test_episode_new_fields_defaults(self):
        """New Week 6 fields have correct defaults."""
        ep = Episode(id="ep_3", content="test")
        assert ep.updated_at is None
        assert ep.error is None
        assert ep.retry_count == 0
        assert ep.processing_duration_ms is None

    def test_episode_old_status_compat(self):
        """Legacy statuses (pending, processing, failed) still work."""
        ep1 = Episode(id="ep_a", content="x", status=EpisodeStatus.PENDING)
        assert ep1.status.value == "pending"
        ep2 = Episode(id="ep_b", content="x", status=EpisodeStatus.PROCESSING)
        assert ep2.status.value == "processing"
        ep3 = Episode(id="ep_c", content="x", status=EpisodeStatus.FAILED)
        assert ep3.status.value == "failed"

    def test_episode_new_status_values(self):
        """Granular pipeline statuses exist and are valid."""
        for name in [
            "QUEUED", "EXTRACTING", "RESOLVING", "WRITING",
            "EMBEDDING", "ACTIVATING", "COMPLETED",
            "RETRYING", "DEAD_LETTER",
        ]:
            status = EpisodeStatus[name]
            assert isinstance(status.value, str)

    def test_episode_status_from_string(self):
        """Status can be created from a string value."""
        ep = Episode(id="ep_d", content="test", status="queued")
        assert ep.status == EpisodeStatus.QUEUED


class TestRelationship:
    def test_create_relationship(self):
        r = Relationship(
            id="rel_1",
            source_id="ent_1",
            target_id="ent_2",
            predicate="USES",
        )
        assert r.weight == 1.0
        assert r.valid_to is None

    def test_confidence_default(self):
        r = Relationship(
            id="rel_2",
            source_id="ent_1",
            target_id="ent_2",
            predicate="USES",
        )
        assert r.confidence == 1.0

    def test_confidence_custom(self):
        r = Relationship(
            id="rel_3",
            source_id="ent_1",
            target_id="ent_2",
            predicate="USES",
            confidence=0.7,
        )
        assert r.confidence == 0.7


class TestActivationState:
    def test_create_default(self):
        state = ActivationState(node_id="ent_1")
        assert state.access_history == []
        assert state.spreading_bonus == 0.0
        assert state.access_count == 0

    def test_activation_state_no_stored_activation(self):
        """Verify current_activation and base_activation fields don't exist."""
        state = ActivationState(node_id="ent_1")
        assert not hasattr(state, "current_activation")
        assert not hasattr(state, "base_activation")

    def test_entity_activation_defaults(self):
        """Verify activation_current defaults to 0 and activation_base is absent."""
        e = Entity(id="ent_1", name="Test", entity_type="Test")
        assert not hasattr(e, "activation_base")
        assert e.activation_current == 0.0


class TestTenantContext:
    def test_default_tenant(self):
        t = TenantContext(group_id="default")
        assert t.role == "owner"
        assert t.auth_method == "none"

    def test_frozen(self):
        t = TenantContext(group_id="test")
        with pytest.raises(Exception):
            t.group_id = "other"

    def test_empty_group_id_rejected(self):
        with pytest.raises(ValueError):
            TenantContext(group_id="")

    def test_whitespace_group_id_rejected(self):
        with pytest.raises(ValueError):
            TenantContext(group_id="   ")
