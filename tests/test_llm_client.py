import pytest
from unittest.mock import MagicMock
from llm.client import LLMClient


def test_init_openai(mocker):
    mocker.patch("llm.client.OpenAI")
    mocker.patch("llm.client.Anthropic")
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client.temperature == 0.3


def test_init_anthropic(mocker):
    mocker.patch("llm.client.OpenAI")
    mocker.patch("llm.client.Anthropic")
    client = LLMClient(provider="anthropic", model="claude-opus-4-6", temperature=0.3)
    assert client.provider == "anthropic"


def test_init_invalid_provider(mocker):
    mocker.patch("llm.client.OpenAI")
    mocker.patch("llm.client.Anthropic")
    with pytest.raises(ValueError, match="Unsupported provider: bogus"):
        LLMClient(provider="bogus", model="x", temperature=0.3)


def test_complete_openai(mocker):
    mock_openai = mocker.patch("llm.client.OpenAI")
    mocker.patch("llm.client.Anthropic")
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello world"))]
    )
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    result = client.complete(system="You are helpful.", user="Say hello")
    assert result == "Hello world"


def test_complete_json_returns_dict(mocker):
    mock_openai = mocker.patch("llm.client.OpenAI")
    mocker.patch("llm.client.Anthropic")
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"key": "value"}'))]
    )
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    result = client.complete_json(system="Return JSON.", user="Give me a dict")
    assert result == {"key": "value"}


def test_complete_json_strips_markdown_fences(mocker):
    mock_openai = mocker.patch("llm.client.OpenAI")
    mocker.patch("llm.client.Anthropic")
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='```json\n{"key": "value"}\n```'))]
    )
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    result = client.complete_json(system="Return JSON.", user="Give me a dict")
    assert result == {"key": "value"}
