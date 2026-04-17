import pytest
from unittest.mock import MagicMock, call
from pathlib import Path
from pipeline.transcriber import transcribe_all


def _make_mock_model(mocker, segments=None):
    mock_model_class = mocker.patch("pipeline.transcriber.WhisperModel")
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.transcribe.return_value = (segments or [], MagicMock())
    return mock_model_class, mock_model


def test_transcribe_all_skips_checkpointed_videos(tmp_checkpoints, sample_video_meta, mocker):
    from utils.checkpoint import save_checkpoint
    save_checkpoint(
        "02_transcripts", "abc123",
        {"video_id": "abc123", "title": "T", "segments": [], "full_text": ""},
        base_dir=tmp_checkpoints,
    )
    _, mock_model = _make_mock_model(mocker)
    results = transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)
    mock_model.transcribe.assert_not_called()
    assert results[0]["video_id"] == "abc123"


def test_transcribe_all_produces_segments(tmp_checkpoints, sample_video_meta, mocker, tmp_path):
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)

    mock_seg1 = MagicMock(start=0.0, end=5.0, text=" Today we learn attention.")
    mock_seg2 = MagicMock(start=5.0, end=10.0, text=" The KV cache is important.")
    _, mock_model = _make_mock_model(mocker, segments=[mock_seg1, mock_seg2])

    results = transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)

    assert len(results) == 1
    t = results[0]
    assert t["video_id"] == "abc123"
    assert len(t["segments"]) == 2
    assert t["segments"][0]["start"] == 0.0
    assert "Today we learn attention." in t["full_text"]


def test_transcribe_all_uses_quality_settings(tmp_checkpoints, sample_video_meta, mocker, tmp_path):
    """Verify model, compute_type, VAD, and no_speech_threshold are applied."""
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)

    mock_model_class, mock_model = _make_mock_model(mocker)

    transcribe_all([sample_video_meta], whisper_model="small", compute_type="int8", base_dir=tmp_checkpoints)

    mock_model_class.assert_called_once_with("small", device="cpu", compute_type="int8")

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs["vad_filter"] is True
    assert call_kwargs["no_speech_threshold"] == 0.3
    assert call_kwargs["condition_on_previous_text"] is True
    # Title is passed as initial_prompt so Whisper knows the domain vocabulary.
    assert call_kwargs["initial_prompt"] == sample_video_meta["title"]


def test_transcribe_all_checkpoint_saved_before_audio_deleted(
    tmp_checkpoints, sample_video_meta, mocker, tmp_path
):
    """Checkpoint must exist on disk before the audio file is removed."""
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)

    checkpoint_saved_before_delete = {}

    original_remove = __import__("os").remove

    def patched_remove(path):
        from utils.checkpoint import checkpoint_exists
        checkpoint_saved_before_delete["saved"] = checkpoint_exists(
            "02_transcripts", "abc123", base_dir=tmp_checkpoints
        )
        original_remove(path)

    mocker.patch("pipeline.transcriber.os.remove", side_effect=patched_remove)
    _make_mock_model(mocker)

    transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)

    assert checkpoint_saved_before_delete.get("saved") is True
    assert not audio_file.exists()


def test_transcribe_all_deletes_audio(tmp_checkpoints, sample_video_meta, mocker, tmp_path):
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)
    _make_mock_model(mocker)
    transcribe_all([sample_video_meta], base_dir=tmp_checkpoints)
    assert not audio_file.exists()


def test_transcribe_all_loads_model_once_for_multiple_videos(tmp_checkpoints, mocker, tmp_path):
    """Model must be instantiated exactly once regardless of video count."""
    videos = []
    for i in range(3):
        vid = f"vid{i}"
        audio_file = tmp_path / f"{vid}.mp3"
        audio_file.write_text("fake")
        videos.append({
            "video_id": vid,
            "title": f"Video {i}",
            "description": "",
            "playlist_index": i,
            "ref_urls": [],
            "audio_path": str(audio_file),
        })

    mock_model_class, mock_model = _make_mock_model(mocker)
    mock_model.transcribe.return_value = ([], MagicMock())

    transcribe_all(videos, base_dir=tmp_checkpoints)

    mock_model_class.assert_called_once()
    assert mock_model.transcribe.call_count == 3
