"""Tests for the Post-Consolidation benchmark method."""

from engram.benchmark.methods import (
    ALL_METHODS,
    METHOD_POST_CONSOLIDATION,
)


class TestPostConsolidationMethod:
    def test_post_consolidation_config(self):
        """Method has requires_consolidation=True and spreading_enabled=True."""
        assert METHOD_POST_CONSOLIDATION.requires_consolidation is True
        assert METHOD_POST_CONSOLIDATION.spreading_enabled is True
        assert METHOD_POST_CONSOLIDATION.routing_enabled is True

    def test_all_methods_includes_post_consolidation(self):
        """Post-Consolidation is present in ALL_METHODS."""
        names = [m.name for m in ALL_METHODS]
        assert "Post-Consolidation" in names

    def test_all_methods_count(self):
        """ALL_METHODS has 16 total methods."""
        assert len(ALL_METHODS) == 16

    def test_regular_methods_exclude_consolidation(self):
        """No regular method has requires_consolidation=True."""
        regular = [m for m in ALL_METHODS if not m.requires_consolidation]
        assert len(regular) == 15
        for m in regular:
            assert m.requires_consolidation is False
