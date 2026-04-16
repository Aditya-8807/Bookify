import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pipeline.fetcher import extract_playlist_videos, fetch_video_audio, fetch_all


def test_extract_playlist_videos(mocker):
    mock_ydl_class = mocker.patch("pipeline.fetcher.YoutubeDL")
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {
        "entries": [
            {"id": "vid1", "title": "Video 1", "description": "desc1", "playlist_index": 1},
            {"id": "vid2", "title": "Video 2", "description": "desc2", "playlist_index": 2},
        ]
    }
    result = extract_playlist_videos("https://youtube.com/playlist?list=ABC")
    assert len(result) == 2
    assert result[0]["video_id"] == "vid1"
    assert result[1]["title"] == "Video 2"


def test_fetch_video_audio_skips_if_checkpoint_exists(tmp_checkpoints, mocker):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("01_fetch", "vid1", {"video_id": "vid1"}, base_dir=tmp_checkpoints)
    mock_ydl = mocker.patch("pipeline.fetcher.YoutubeDL")
    fetch_video_audio(
        {"video_id": "vid1", "title": "T", "description": "", "playlist_index": 0},
        llm_client=MagicMock(),
        base_dir=tmp_checkpoints,
    )
    mock_ydl.assert_not_called()


def test_fetch_video_audio_saves_checkpoint(tmp_checkpoints, mocker):
    mocker.patch("pipeline.fetcher.YoutubeDL")
    mocker.patch("pipeline.fetcher.filter_description_urls", return_value=["https://arxiv.org/abs/1706.03762"])
    fetch_video_audio(
        {"video_id": "vid2", "title": "Attention", "description": "Paper: https://arxiv.org/abs/1706.03762", "playlist_index": 1},
        llm_client=MagicMock(),
        base_dir=tmp_checkpoints,
    )
    from utils.checkpoint import checkpoint_exists
    assert checkpoint_exists("01_fetch", "vid2", base_dir=tmp_checkpoints)
