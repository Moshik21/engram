import asyncio
from engram.config import ActivationConfig
from engram.retrieval.belief import BeliefMapScorer
from engram.activation.neuroplasticity import NeuroplasticityEngine

async def test_synapse():
    cfg = ActivationConfig()
    cfg.belief_map_enabled = True
    cfg.neuroplasticity_enabled = True

    # Test Belief Map
    scorer = BeliefMapScorer(cfg)
    entity_data = {"evidence_count": 5}
    belief = scorer.calculate_belief(entity_data, relevance=0.8, activation=0.7)
    print(f"Belief Map: {belief.to_dict()}")

    # Test Neuroplasticity
    engine = NeuroplasticityEngine(cfg)
    engine.handle_positive_feedback("entity_123")
    print("Neuroplasticity test complete (check logs)")

if __name__ == "__main__":
    asyncio.run(test_synapse())
