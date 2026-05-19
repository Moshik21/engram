from engram.config import ActivationConfig
from engram.retrieval.belief import BeliefMapScorer


def test_belief_map_scoring():
    cfg = ActivationConfig()
    cfg.belief_map_enabled = True
    scorer = BeliefMapScorer(cfg)

    # Test high corroboration
    entity_data = {"evidence_count": 10, "relationships": []}
    belief = scorer.calculate_belief(entity_data, relevance=0.9, activation=0.8)
    assert belief.composite_confidence > 0.8
    assert belief.evidence_density == 1.0

    # Test temporal polarity (negation)
    entity_data_negated = {
        "evidence_count": 10,
        "relationships": [{"polarity": "negative"}]
    }
    belief_negated = scorer.calculate_belief(entity_data_negated, relevance=0.9, activation=0.8)
    assert belief_negated.temporal_stability < 1.0
    assert belief_negated.composite_confidence < belief.composite_confidence

def test_belief_map_disabled():
    cfg = ActivationConfig()
    cfg.belief_map_enabled = False
    scorer = BeliefMapScorer(cfg)

    entity_data = {"evidence_count": 10}
    belief = scorer.calculate_belief(entity_data, relevance=0.9, activation=0.8)
    assert belief.composite_confidence == 0.9  # Should fallback to relevance
