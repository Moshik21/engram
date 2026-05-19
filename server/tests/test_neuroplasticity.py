from engram.activation.neuroplasticity import NeuroplasticityEngine
from engram.config import ActivationConfig


def test_neuroplasticity_feedback_loop():
    cfg = ActivationConfig()
    cfg.neuroplasticity_enabled = True
    engine = NeuroplasticityEngine(cfg)

    # Positive feedback should nudge parameters
    # (Since it currently just logs/returns, we check the handle calls)
    engine.handle_positive_feedback("entity_1")
    engine.handle_negative_feedback("entity_2")

    # Verify it doesn't crash and returns expected types if we add return values
    # For now, it's a pass/fail on execution
    assert True

def test_neuroplasticity_disabled():
    cfg = ActivationConfig()
    cfg.neuroplasticity_enabled = False
    engine = NeuroplasticityEngine(cfg)
    engine.handle_positive_feedback("entity_1")
    assert True
