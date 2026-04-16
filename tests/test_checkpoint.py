import pytest
from pathlib import Path
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists, list_checkpoints


def test_save_and_load(tmp_checkpoints):
    data = {"video_id": "abc", "title": "Test"}
    save_checkpoint("01_fetch", "abc", data, base_dir=tmp_checkpoints)
    result = load_checkpoint("01_fetch", "abc", base_dir=tmp_checkpoints)
    assert result == data


def test_checkpoint_exists_true(tmp_checkpoints):
    save_checkpoint("01_fetch", "abc", {}, base_dir=tmp_checkpoints)
    assert checkpoint_exists("01_fetch", "abc", base_dir=tmp_checkpoints)


def test_checkpoint_exists_false(tmp_checkpoints):
    assert not checkpoint_exists("01_fetch", "missing", base_dir=tmp_checkpoints)


def test_list_checkpoints(tmp_checkpoints):
    save_checkpoint("01_fetch", "vid1", {}, base_dir=tmp_checkpoints)
    save_checkpoint("01_fetch", "vid2", {}, base_dir=tmp_checkpoints)
    keys = list_checkpoints("01_fetch", base_dir=tmp_checkpoints)
    assert set(keys) == {"vid1", "vid2"}


def test_list_checkpoints_empty(tmp_checkpoints):
    assert list_checkpoints("01_fetch", base_dir=tmp_checkpoints) == []


def test_save_creates_nested_dirs(tmp_checkpoints):
    save_checkpoint("04_topics", "attention-mechanisms", {"prose": "..."}, base_dir=tmp_checkpoints)
    assert checkpoint_exists("04_topics", "attention-mechanisms", base_dir=tmp_checkpoints)
