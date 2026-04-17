import pytest
from unittest.mock import MagicMock
from pipeline.grouper import group_and_order, fetch_and_checkpoint_ref_content, fetch_reference_content, slugify
from utils.checkpoint import save_checkpoint


def test_slugify():
    assert slugify("The Attention Mechanism") == "the-attention-mechanism"
    assert slugify("KV Cache & Embeddings") == "kv-cache-embeddings"


def _base_transcripts():
    return [
        {"video_id": "vid1", "title": "Tokenization", "full_text": "tokenization text",
         "segments": [], "corrections": []},
        {"video_id": "vid2", "title": "Embeddings", "full_text": "embeddings text",
         "segments": [], "corrections": []},
    ]


def _base_video_metas():
    return [
        {"video_id": "vid1", "ref_urls": [], "title": "T", "description": "",
         "playlist_index": 0, "audio_path": ""},
        {"video_id": "vid2", "ref_urls": [], "title": "E", "description": "",
         "playlist_index": 1, "audio_path": ""},
    ]


def test_group_and_order_returns_sorted_groups(tmp_checkpoints):
    mock_client = MagicMock()
    # LLM returns Embeddings first — but vid2 has playlist_index=1, vid1 has 0,
    # so Tokenization (vid1) must come first in the output.
    mock_client.complete_json.return_value = {
        "topics": [
            {"name": "Embeddings", "video_ids": ["vid2"]},
            {"name": "Tokenization", "video_ids": ["vid1"]},
        ]
    }
    save_checkpoint("01b_ref_content", "vid1", {}, base_dir=tmp_checkpoints)
    save_checkpoint("01b_ref_content", "vid2", {}, base_dir=tmp_checkpoints)

    groups = group_and_order(
        _base_transcripts(), _base_video_metas(), mock_client, base_dir=tmp_checkpoints
    )
    # Sorted by playlist_index, not LLM order
    assert groups[0]["name"] == "Tokenization"
    assert groups[1]["name"] == "Embeddings"
    assert groups[1]["dependency_order"] == 1


def test_group_and_order_loads_ref_content_from_checkpoint(tmp_checkpoints):
    """group_and_order must use pre-saved ref content, not re-fetch."""
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "topics": [{"name": "Attention", "video_ids": ["vid1"], "prerequisites": []}]
    }
    save_checkpoint(
        "01b_ref_content", "vid1",
        {"https://example.com/paper": "The attention mechanism explained."},
        base_dir=tmp_checkpoints,
    )

    groups = group_and_order(
        [{"video_id": "vid1", "title": "Attention", "full_text": "attn", "segments": [], "corrections": []}],
        [{"video_id": "vid1", "ref_urls": ["https://example.com/paper"], "title": "A",
          "description": "", "playlist_index": 0, "audio_path": ""}],
        mock_client,
        base_dir=tmp_checkpoints,
    )
    assert groups[0]["ref_contents"]["https://example.com/paper"] == "The attention mechanism explained."


def test_fetch_and_checkpoint_ref_content_saves_per_video(tmp_checkpoints, mocker):
    mocker.patch(
        "pipeline.grouper.fetch_reference_content",
        return_value={"https://arxiv.org/abs/1706.03762": "Attention is all you need."},
    )
    video_metas = [
        {"video_id": "vid1", "ref_urls": ["https://arxiv.org/abs/1706.03762"],
         "title": "T", "description": "", "playlist_index": 0, "audio_path": ""},
    ]
    fetch_and_checkpoint_ref_content(video_metas, base_dir=tmp_checkpoints)

    from utils.checkpoint import load_checkpoint, checkpoint_exists
    assert checkpoint_exists("01b_ref_content", "vid1", base_dir=tmp_checkpoints)
    content = load_checkpoint("01b_ref_content", "vid1", base_dir=tmp_checkpoints)
    assert "Attention is all you need." in content["https://arxiv.org/abs/1706.03762"]


def test_fetch_and_checkpoint_ref_content_skips_existing(tmp_checkpoints, mocker):
    save_checkpoint("01b_ref_content", "vid1", {"url": "cached"}, base_dir=tmp_checkpoints)
    mock_fetch = mocker.patch("pipeline.grouper.fetch_reference_content")
    video_metas = [
        {"video_id": "vid1", "ref_urls": ["url"], "title": "T",
         "description": "", "playlist_index": 0, "audio_path": ""},
    ]
    fetch_and_checkpoint_ref_content(video_metas, base_dir=tmp_checkpoints)
    mock_fetch.assert_not_called()


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
