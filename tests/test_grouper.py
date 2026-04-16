import pytest
from unittest.mock import MagicMock
from pipeline.grouper import group_and_order, fetch_reference_content, slugify


def test_slugify():
    assert slugify("The Attention Mechanism") == "the-attention-mechanism"
    assert slugify("KV Cache & Embeddings") == "kv-cache-embeddings"


def test_group_and_order_returns_sorted_groups(tmp_checkpoints, mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "topics": [
            {"name": "Embeddings", "video_ids": ["vid2"], "prerequisites": ["Tokenization"]},
            {"name": "Tokenization", "video_ids": ["vid1"], "prerequisites": []},
        ]
    }
    mocker.patch("pipeline.grouper.fetch_reference_content", return_value={})

    transcripts = [
        {"video_id": "vid1", "title": "Tokenization", "full_text": "tokenization text",
         "segments": [], "corrections": []},
        {"video_id": "vid2", "title": "Embeddings", "full_text": "embeddings text",
         "segments": [], "corrections": []},
    ]
    video_metas = [
        {"video_id": "vid1", "ref_urls": [], "title": "T", "description": "",
         "playlist_index": 0, "audio_path": ""},
        {"video_id": "vid2", "ref_urls": [], "title": "E", "description": "",
         "playlist_index": 1, "audio_path": ""},
    ]

    groups = group_and_order(transcripts, video_metas, mock_client, base_dir=tmp_checkpoints)
    assert groups[0]["name"] == "Tokenization"
    assert groups[1]["name"] == "Embeddings"
    assert groups[1]["dependency_order"] == 1


def test_fetch_reference_content_skips_on_error(mocker):
    mocker.patch("requests.get", side_effect=Exception("network error"))
    result = fetch_reference_content(["https://arxiv.org/abs/1706.03762"])
    assert result == {}


def test_fetch_reference_content_returns_text(mocker):
    mock_resp = MagicMock()
    mock_resp.text = "<html><body><p>Attention is all you need.</p></body></html>"
    mock_resp.raise_for_status = MagicMock()
    mocker.patch("requests.get", return_value=mock_resp)
    result = fetch_reference_content(["https://arxiv.org/abs/1706.03762"])
    assert "https://arxiv.org/abs/1706.03762" in result
    assert "Attention is all you need." in result["https://arxiv.org/abs/1706.03762"]
