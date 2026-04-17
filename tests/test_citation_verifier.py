import pytest
from unittest.mock import MagicMock
from pipeline.citation_verifier import verify_topic, verify_all_topics


def _prose_response(prose, verified=1, rewritten=0, removed=0):
    return (
        f"PROSE:\n{prose}\n"
        f"---STATS---\n"
        f"verified={verified} rewritten={rewritten} removed={removed}"
    )


def test_verify_topic_skips_if_checkpoint_exists(tmp_checkpoints):
    from utils.checkpoint import save_checkpoint
    save_checkpoint(
        "04b_verified", "attention",
        {"name": "attention", "slug": "attention", "prose": "verified", "citations": [], "stats": {}},
        base_dir=tmp_checkpoints,
    )
    mock_client = MagicMock()
    topic = {"name": "attention", "slug": "attention", "prose": "original"}
    result = verify_topic(topic, [], mock_client, base_dir=tmp_checkpoints)
    mock_client.complete.assert_not_called()
    assert result["prose"] == "verified"


def test_verify_topic_single_pass(tmp_checkpoints):
    mock_client = MagicMock()
    mock_client.complete.return_value = _prose_response(
        "The transformer uses multi-head attention.", verified=1
    )
    topic = {"name": "Attention", "slug": "attention-test",
             "prose": "The transformer uses multi-head attention."}
    result = verify_topic(topic, ["transformer uses multi-head attention"], mock_client, base_dir=tmp_checkpoints)
    assert "transformer" in result["prose"].lower()
    assert result["stats"]["verified"] == 1
    mock_client.complete.assert_called_once()


def test_verify_topic_uses_sources(tmp_checkpoints):
    """Verify source texts are included in the LLM prompt."""
    mock_client = MagicMock()
    mock_client.complete.return_value = _prose_response("Fixed prose.", rewritten=1)
    topic = {"name": "Test", "slug": "test-slug", "prose": "Some claim here."}
    verify_topic(topic, ["source text 1", "source text 2"], mock_client, base_dir=tmp_checkpoints)
    call_args = mock_client.complete.call_args
    full_call = str(call_args)
    assert "source text 1" in full_call


def test_verify_all_topics_sequential(tmp_checkpoints):
    mock_client = MagicMock()
    mock_client.complete.return_value = _prose_response("Verified prose.", verified=1)
    topics = [
        {"name": "Topic A", "slug": "topic-a", "prose": "Prose A."},
        {"name": "Topic B", "slug": "topic-b", "prose": "Prose B."},
    ]
    groups_by_slug = {
        "topic-a": {"video_ids": ["v1"], "ref_contents": {}},
        "topic-b": {"video_ids": ["v2"], "ref_contents": {}},
    }
    transcripts_by_vid = {
        "v1": {"full_text": "source a"},
        "v2": {"full_text": "source b"},
    }
    results = verify_all_topics(
        topics, groups_by_slug, transcripts_by_vid,
        llm_client=mock_client, base_dir=tmp_checkpoints,
    )
    assert len(results) == 2
    assert mock_client.complete.call_count == 2
