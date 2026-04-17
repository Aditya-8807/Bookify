import pytest
from unittest.mock import patch, MagicMock
from pipeline.transcriber import transcribe_all, _merge_segments


def _snippet(start, duration, text):
    s = MagicMock()
    s.start = start
    s.duration = duration
    s.text = text
    return s


def _patch_api(snippets):
    mock_instance = MagicMock()
    mock_instance.fetch.return_value = snippets
    return patch("pipeline.transcriber._YTApi", return_value=mock_instance), mock_instance


def test_transcribe_all_skips_checkpointed_videos(tmp_checkpoints, sample_video_meta):
    from utils.checkpoint import save_checkpoint
    save_checkpoint(
        "02_transcripts", "abc123",
        {"video_id": "abc123", "title": "T", "segments": [], "full_text": ""},
        base_dir=tmp_checkpoints,
    )
    patcher, mock_instance = _patch_api([])
    with patcher:
        results = transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)
        mock_instance.fetch.assert_not_called()
    assert results[0]["video_id"] == "abc123"


def test_transcribe_all_produces_segments(tmp_checkpoints, sample_video_meta):
    snippets = [
        _snippet(0.0, 5.0, "Today we learn attention."),
        _snippet(5.0, 5.0, "The KV cache is important."),
    ]
    patcher, _ = _patch_api(snippets)
    with patcher:
        results = transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)

    t = results[0]
    assert t["video_id"] == "abc123"
    # Both snippets fit within the 45s window so they merge into one block.
    assert len(t["segments"]) == 1
    assert t["segments"][0]["start"] == 0.0
    assert t["segments"][0]["end"] == 10.0
    assert "Today we learn attention." in t["full_text"]


def test_transcribe_all_saves_checkpoint(tmp_checkpoints, sample_video_meta):
    patcher, _ = _patch_api([_snippet(0.0, 3.0, "Hello world.")])
    with patcher:
        transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)

    from utils.checkpoint import checkpoint_exists
    assert checkpoint_exists("02_transcripts", "abc123", base_dir=tmp_checkpoints)


def test_transcribe_all_full_text_joins_segments(tmp_checkpoints, sample_video_meta):
    snippets = [
        _snippet(0.0, 5.0, "First segment."),
        _snippet(5.0, 5.0, "Second segment."),
    ]
    patcher, _ = _patch_api(snippets)
    with patcher:
        results = transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)

    assert results[0]["full_text"] == "First segment. Second segment."


def test_merge_segments_groups_within_window():
    # A+B span 0–40 (40s ≤ 45s) → merged; C and D each exceed the window alone.
    segs = [
        {"start": 0.0, "end": 20.0, "text": "A"},
        {"start": 20.0, "end": 40.0, "text": "B"},
        {"start": 40.0, "end": 50.0, "text": "C"},
        {"start": 50.0, "end": 100.0, "text": "D"},
    ]
    merged = _merge_segments(segs, max_duration_s=45.0)
    assert len(merged) == 3
    assert merged[0]["text"] == "A B"
    assert merged[1]["text"] == "C"
    assert merged[2]["text"] == "D"


def test_transcribe_all_multiple_videos(tmp_checkpoints):
    videos = [
        {"video_id": f"vid{i}", "title": f"Video {i}", "description": "",
         "playlist_index": i, "ref_urls": []}
        for i in range(3)
    ]
    snippets = [_snippet(0.0, 5.0, "content")]
    patcher, mock_instance = _patch_api(snippets)
    with patcher:
        results = transcribe_all(videos, base_dir=tmp_checkpoints)

    assert mock_instance.fetch.call_count == 3
    assert len(results) == 3
