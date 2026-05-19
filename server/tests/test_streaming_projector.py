from unittest.mock import MagicMock, patch

from engram.extraction.models import ClaimCandidate, EntityCandidate, ExtractionResult
from engram.extraction.streaming_projector import StreamingEvidenceProjector


def test_streaming_broadcast():
    with patch("engram.extraction.streaming_projector.get_event_bus") as mock_get_bus:
        mock_bus = MagicMock()
        mock_get_bus.return_value = mock_bus

        projector = StreamingEvidenceProjector(group_id="test_group")

        result = ExtractionResult(
            entities=[EntityCandidate(name="Konner", entity_type="person", summary="Dev")],
            relationships=[
                ClaimCandidate(
                    subject_text="Konner",
                    predicate="lives_in",
                    object_text="SF",
                )
            ],
        )

        projector.broadcast_result(result)

        # Verify both entity and relationship were published
        assert mock_bus.publish.call_count == 2

        # Check first call (entity)
        args1, kwargs1 = mock_bus.publish.call_args_list[0]
        assert kwargs1["event_type"] == "streaming.entity_discovered"
        assert kwargs1["payload"]["name"] == "Konner"

        # Check second call (relationship)
        args2, kwargs2 = mock_bus.publish.call_args_list[1]
        assert kwargs2["event_type"] == "streaming.relationship_discovered"
        assert kwargs2["payload"]["source"] == "Konner"
