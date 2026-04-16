import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pipeline.transcriber import transcribe_video, transcribe_all


def test_transcribe_video_skips_if_checkpoint_exists(tmp_checkpoints, sample_video_meta):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("02_transcripts", "abc123", {"video_id": "abc123"}, base_dir=tmp_checkpoints)
    with patch("pipeline.transcriber.WhisperModel") as mock_model:
        result = transcribe_video(sample_video_meta, whisper_model="large-v3", base_dir=tmp_checkpoints)
    mock_model.assert_not_called()
    assert result["video_id"] == "abc123"


def test_transcribe_video_produces_segments(tmp_checkpoints, sample_video_meta, mocker, tmp_path):
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)

    mock_model_class = mocker.patch("pipeline.transcriber.WhisperModel")
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_seg1 = MagicMock(start=0.0, end=5.0, text=" Today we learn attention.")
    mock_seg2 = MagicMock(start=5.0, end=10.0, text=" The KV cache is important.")
    mock_model.transcribe.return_value = ([mock_seg1, mock_seg2], MagicMock())

    result = transcribe_video(sample_video_meta, whisper_model="large-v3", base_dir=tmp_checkpoints)

    assert result["video_id"] == "abc123"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 0.0
    assert "Today we learn attention." in result["full_text"]


def test_transcribe_video_deletes_audio(tmp_checkpoints, sample_video_meta, mocker, tmp_path):
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)

    mock_model_class = mocker.patch("pipeline.transcriber.WhisperModel")
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.transcribe.return_value = ([], MagicMock())

    transcribe_video(sample_video_meta, whisper_model="large-v3", base_dir=tmp_checkpoints)
    assert not audio_file.exists()
