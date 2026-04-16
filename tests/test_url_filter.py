import pytest
from unittest.mock import MagicMock
from utils.url_filter import (
    extract_urls_from_description,
    blacklist_filter,
    llm_classify_urls,
    filter_description_urls,
)


def test_extract_urls_basic():
    desc = "Check out https://arxiv.org/abs/1706.03762 and https://github.com/karpathy/llm.c"
    urls = extract_urls_from_description(desc)
    assert "https://arxiv.org/abs/1706.03762" in urls
    assert "https://github.com/karpathy/llm.c" in urls


def test_extract_urls_with_context():
    desc = "Paper link: https://arxiv.org/abs/1706.03762\nSupport me: https://patreon.com/x"
    result = extract_urls_from_description(desc)
    assert len(result) == 2


def test_blacklist_removes_patreon():
    urls = ["https://patreon.com/xyz", "https://arxiv.org/abs/1706.03762"]
    assert blacklist_filter(urls) == ["https://arxiv.org/abs/1706.03762"]


def test_blacklist_removes_twitter_and_x():
    urls = ["https://twitter.com/user", "https://x.com/user", "https://github.com/repo"]
    assert blacklist_filter(urls) == ["https://github.com/repo"]


def test_blacklist_removes_discord():
    urls = ["https://discord.gg/abc123"]
    assert blacklist_filter(urls) == []


def test_blacklist_removes_youtube_selflinks():
    urls = ["https://youtube.com/watch?v=abc", "https://arxiv.org/abs/1234.5678"]
    assert blacklist_filter(urls) == ["https://arxiv.org/abs/1234.5678"]


def test_blacklist_keeps_arxiv_github_huggingface():
    urls = [
        "https://arxiv.org/abs/1706.03762",
        "https://github.com/karpathy/nanoGPT",
        "https://huggingface.co/docs/transformers",
    ]
    assert blacklist_filter(urls) == urls


def test_llm_classify_keeps_educational():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "classifications": [
            {"url": "https://arxiv.org/abs/1706.03762", "label": "educational_reference"},
        ]
    }
    urls_with_context = [("https://arxiv.org/abs/1706.03762", "The paper I reference:")]
    result = llm_classify_urls(urls_with_context, mock_client)
    assert result == ["https://arxiv.org/abs/1706.03762"]


def test_llm_classify_drops_promotional():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "classifications": [
            {"url": "https://some-course.com/enroll", "label": "promotional"},
        ]
    }
    urls_with_context = [("https://some-course.com/enroll", "Join my course:")]
    result = llm_classify_urls(urls_with_context, mock_client)
    assert result == []


def test_filter_description_urls_end_to_end():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "classifications": [
            {"url": "https://arxiv.org/abs/1706.03762", "label": "educational_reference"},
        ]
    }
    desc = "Paper: https://arxiv.org/abs/1706.03762\nPatreon: https://patreon.com/xyz"
    result = filter_description_urls(desc, mock_client)
    assert result == ["https://arxiv.org/abs/1706.03762"]
