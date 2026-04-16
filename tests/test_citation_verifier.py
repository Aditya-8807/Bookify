import pytest
from unittest.mock import MagicMock
from pipeline.citation_verifier import extract_claims, score_claim, verify_topic


def test_extract_claims_returns_list():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "claims": [
            "The transformer uses multi-head attention.",
            "Softmax normalizes attention scores to sum to 1.",
        ]
    }
    claims = extract_claims("Some prose about transformers.", mock_client)
    assert len(claims) == 2
    assert "transformer" in claims[0]


def test_score_claim_verified():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "score": 0.95,
        "passage": "The transformer uses multi-head attention.",
    }
    score, passage = score_claim(
        claim="The transformer uses multi-head attention.",
        source_texts=["The transformer uses multi-head attention with 8 heads."],
        llm_client=mock_client,
    )
    assert score >= 0.8
    assert "transformer" in passage


def test_score_claim_unverified():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {"score": 0.2, "passage": ""}
    score, passage = score_claim(
        claim="The model has exactly 175 billion parameters.",
        source_texts=["The model was trained on a large dataset."],
        llm_client=mock_client,
    )
    assert score < 0.5


def test_verify_topic_skips_if_checkpoint_exists(tmp_checkpoints):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("04b_verified", "attention", {"name": "attention", "prose": "verified"}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    topic = {"name": "attention", "slug": "attention", "prose": "original"}
    result = verify_topic(topic, [], mock_client, base_dir=tmp_checkpoints)
    mock_client.complete_json.assert_not_called()
    assert result["prose"] == "verified"


def test_verify_topic_keeps_verified_claims(tmp_checkpoints):
    mock_client = MagicMock()
    mock_client.complete_json.side_effect = [
        {"claims": ["The transformer uses multi-head attention."]},
        {"score": 0.95, "passage": "transformer uses multi-head attention"},
    ]
    topic = {"name": "attention", "slug": "attention-test",
             "prose": "The transformer uses multi-head attention."}
    result = verify_topic(topic, ["transformer uses multi-head attention"], mock_client, base_dir=tmp_checkpoints)
    assert "transformer" in result["prose"].lower()
    assert result["stats"]["verified"] == 1
