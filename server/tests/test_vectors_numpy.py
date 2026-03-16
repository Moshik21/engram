"""Tests for numpy-accelerated vector operations."""

from engram.utils.vectors import cosine_similarity


class TestCosineSimNumpy:
    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_zero_vector(self):
        """Zero vector returns 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_empty_vectors(self):
        """Empty vectors return 0.0."""
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        """Mismatched vector lengths return 0.0."""
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
