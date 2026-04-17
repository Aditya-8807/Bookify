import pytest
from unittest.mock import MagicMock
from pipeline.terminology_corrector import correct_transcript, correct_all, _apply_corrections


def test_correct_transcript_fixes_misheard_terms(tmp_checkpoints, sample_transcript):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "corrections": [
            {"original": "a tension mechanisms", "corrected": "attention mechanisms", "timestamp": "00:00"},
            {"original": "cave cache", "corrected": "KV cache", "timestamp": "00:05"},
        ]
    }
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    assert "attention mechanisms" in result["full_text"]
    assert "KV cache" in result["full_text"]
    assert len(result["corrections"]) == 2


def test_correct_transcript_skips_if_checkpoint_exists(tmp_checkpoints, sample_transcript):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("02b_corrected", "abc123", {"video_id": "abc123", "corrections": []}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    mock_client.complete_json.assert_not_called()
    assert result["video_id"] == "abc123"


def test_correct_transcript_logs_corrections(tmp_checkpoints, sample_transcript):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "corrections": [{"original": "a tension", "corrected": "attention", "timestamp": "00:00"}]
    }
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    assert result["corrections"][0]["original"] == "a tension"
    assert result["corrections"][0]["corrected"] == "attention"


def test_correct_transcript_no_corrections(tmp_checkpoints, sample_transcript):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {"corrections": []}
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    assert result["segments"] == sample_transcript["segments"]
    assert result["corrections"] == []


def test_apply_corrections_word_boundary():
    segs = [{"start": 0.0, "end": 5.0, "text": "a tension is key, not tension but a tension"}]
    corrections = [{"original": "a tension", "corrected": "attention", "timestamp": "00:00"}]
    result = _apply_corrections(segs, corrections)
    assert result[0]["text"] == "attention is key, not tension but attention"
