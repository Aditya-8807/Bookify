import pytest
from pathlib import Path


@pytest.fixture
def tmp_checkpoints(tmp_path):
    """Temporary checkpoints directory for tests."""
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    return cp


@pytest.fixture
def sample_video_meta():
    return {
        "video_id": "abc123",
        "title": "The Attention Mechanism",
        "description": "Paper: https://arxiv.org/abs/1706.03762\nSupport: https://patreon.com/xyz",
        "playlist_index": 0,
        "ref_urls": [],
        "audio_path": "checkpoints/audio/abc123.mp3",
    }


@pytest.fixture
def sample_transcript():
    return {
        "video_id": "abc123",
        "title": "The Attention Mechanism",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Today we look at a tension mechanisms."},
            {"start": 5.0, "end": 10.0, "text": "The cave cache stores key value pairs."},
        ],
        "full_text": "Today we look at a tension mechanisms. The cave cache stores key value pairs.",
    }
