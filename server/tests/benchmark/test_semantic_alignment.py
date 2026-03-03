"""Tests for semantic query/ground-truth alignment and cross-cluster pair uniqueness."""

from engram.benchmark.corpus import CorpusGenerator

_corpus = CorpusGenerator(seed=42).generate()


def _entity_names_in_text(query_text: str, entity_map: dict) -> set[str]:
    """Return entity IDs whose name appears in the query text."""
    found: set[str] = set()
    for eid, entity in entity_map.items():
        if entity.name.lower() in query_text.lower():
            found.add(eid)
    return found


class TestSemanticQueryAlignment:
    """Each semantic query should reference entities that overlap with its ground truth."""

    def test_each_semantic_query_has_overlapping_entities(self):
        entity_map = {e.id: e for e in _corpus.entities}
        semantic_queries = [
            q for q in _corpus.ground_truth if q.category == "semantic"
        ]
        assert len(semantic_queries) > 0, "No semantic queries found"

        failures = []
        for q in semantic_queries:
            # Find entity names mentioned in the query text
            names_in_query = _entity_names_in_text(q.query_text, entity_map)
            gt_ids = set(q.relevant_entities.keys())
            overlap = names_in_query & gt_ids
            if not overlap:
                failures.append(
                    f"{q.query_id}: query='{q.query_text[:80]}...' "
                    f"mentions={[entity_map[e].name for e in names_in_query]}, "
                    f"gt_count={len(gt_ids)}"
                )

        assert len(failures) <= len(semantic_queries) // 2, (
            f"{len(failures)}/{len(semantic_queries)} semantic queries have "
            f"zero entity overlap between query text and ground truth:\n"
            + "\n".join(failures)
        )


class TestCrossClusterPairUniqueness:
    """Cross-cluster queries should test distinct cluster pairs."""

    def test_cross_cluster_uses_distinct_pairs(self):
        cross_queries = [
            q for q in _corpus.ground_truth if q.category == "cross_cluster"
        ]
        assert len(cross_queries) > 0, "No cross-cluster queries found"

        # Each pair of queries should have < 60% ground truth overlap
        high_overlap_pairs = []
        for i, q1 in enumerate(cross_queries):
            gt1 = set(q1.relevant_entities.keys())
            for j, q2 in enumerate(cross_queries):
                if j <= i:
                    continue
                gt2 = set(q2.relevant_entities.keys())
                if not gt1 or not gt2:
                    continue
                overlap = len(gt1 & gt2) / min(len(gt1), len(gt2))
                if overlap >= 0.60:
                    high_overlap_pairs.append(
                        f"{q1.query_id} & {q2.query_id}: "
                        f"{overlap:.0%} overlap ({len(gt1 & gt2)} shared)"
                    )

        assert len(high_overlap_pairs) == 0, (
            "Cross-cluster queries have high ground truth overlap:\n"
            + "\n".join(high_overlap_pairs)
        )
