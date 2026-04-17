import pytest
from unittest.mock import MagicMock
from pipeline.topic_writer import write_topic, write_all_topics


def test_write_topic_skips_if_checkpoint_exists(tmp_checkpoints):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("04_topics", "attention", {"name": "attention", "prose": "..."}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    group = {"name": "Attention", "slug": "attention", "video_ids": [],
             "dependency_order": 0, "prerequisites": [], "ref_urls": [], "ref_contents": {}}
    result = write_topic(group, [], {}, mock_client, base_dir=tmp_checkpoints)
    mock_client.complete.assert_not_called()
    assert result["name"] == "attention"


def test_write_topic_calls_llm_with_transcript(tmp_checkpoints):
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "Attention mechanisms allow the model to focus on relevant tokens. "
        "[Video: \"The Attention Mechanism\" @ 12:34]"
    )
    group = {
        "name": "The Attention Mechanism", "slug": "the-attention-mechanism",
        "video_ids": ["abc123"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [{"video_id": "abc123", "title": "The Attention Mechanism",
                    "segments": [{"start": 752.0, "end": 760.0,
                                  "text": "attention allows the model to focus"}],
                    "full_text": "attention allows the model to focus", "corrections": []}]
    result = write_topic(group, transcripts, {}, mock_client, base_dir=tmp_checkpoints)
    assert "attention" in result["prose"].lower()
    mock_client.complete.assert_called_once()
