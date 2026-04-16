import pytest
from unittest.mock import MagicMock
from pipeline.assembler import generate_glossary, assemble_book


def test_generate_glossary_returns_terms():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "terms": [
            {"term": "attention mechanism", "definition": "A mechanism that weights input tokens by relevance."},
            {"term": "softmax", "definition": "A function that converts logits to a probability distribution."},
        ]
    }
    glossary = generate_glossary("Some book prose about attention and softmax.", mock_client)
    assert len(glossary) == 2
    assert glossary[0]["term"] == "attention mechanism"


def test_assemble_book_includes_all_sections(tmp_checkpoints):
    mock_client = MagicMock()
    mock_client.complete.return_value = "Generated intro or conclusion text."
    mock_client.complete_json.return_value = {
        "terms": [{"term": "attention", "definition": "A weighting mechanism."}]
    }
    topics = [
        {"name": "Tokenization", "slug": "tokenization", "prose": "Tokenization prose.",
         "citations": [], "stats": {}},
        {"name": "Attention", "slug": "attention", "prose": "Attention prose.",
         "citations": [], "stats": {}},
    ]
    groups = [
        {"name": "Tokenization", "slug": "tokenization",
         "ref_urls": ["https://arxiv.org/abs/1"], "ref_contents": {},
         "video_ids": [], "dependency_order": 0, "prerequisites": []},
        {"name": "Attention", "slug": "attention",
         "ref_urls": [], "ref_contents": {},
         "video_ids": [], "dependency_order": 1, "prerequisites": []},
    ]
    result = assemble_book(topics, groups, mock_client, base_dir=tmp_checkpoints)
    assert "Tokenization prose." in result
    assert "Attention prose." in result
    assert "Generated intro or conclusion text." in result
    assert "https://arxiv.org/abs/1" in result
