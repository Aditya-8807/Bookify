import pytest
from unittest.mock import MagicMock
from pipeline.topic_writer import detect_overlaps, write_topic, write_all_topics


def test_detect_overlaps_finds_shared_concepts():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "overlaps": [
            {
                "concept": "softmax function",
                "primary_topic": "The Attention Mechanism",
                "secondary_topics": ["Training from Scratch"],
            }
        ]
    }
    groups = [
        {"name": "The Attention Mechanism", "slug": "attention", "video_ids": ["v1"],
         "dependency_order": 0, "prerequisites": [], "ref_urls": [], "ref_contents": {}},
        {"name": "Training from Scratch", "slug": "training", "video_ids": ["v2"],
         "dependency_order": 1, "prerequisites": [], "ref_urls": [], "ref_contents": {}},
    ]
    overlaps = detect_overlaps(groups, mock_client)
    assert len(overlaps) == 1
    assert overlaps[0]["concept"] == "softmax function"


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


def test_write_topic_retries_when_draft_too_short(tmp_checkpoints):
    short_draft = "Too short."
    full_draft = " ".join(["word"] * 2600)
    mock_client = MagicMock()
    mock_client.complete.side_effect = [short_draft, full_draft]

    # Build a transcript large enough that source_word_budget >= min_words
    long_text = " ".join(["token"] * 2600)
    group = {
        "name": "The Attention Mechanism", "slug": "attention-retry",
        "video_ids": ["abc123"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [{"video_id": "abc123", "title": "The Attention Mechanism",
                    "segments": [{"start": 0.0, "end": 10.0, "text": long_text}],
                    "full_text": long_text, "corrections": []}]
    result = write_topic(group, transcripts, {}, mock_client, min_words=2500, base_dir=tmp_checkpoints)
    assert mock_client.complete.call_count == 2
    assert result["prose"] == full_draft
