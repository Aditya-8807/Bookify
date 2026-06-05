from unittest.mock import MagicMock
from pipeline.subtopic_splitter import split_topic_into_subtopics, split_all_topics


def _make_group(name="Attention Mechanism", slug="attention-mechanism"):
    return {
        "name": name, "slug": slug, "video_ids": ["v1", "v2"],
        "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {}, "subtopics": [],
    }


def test_split_topic_returns_list_of_subtopics(tmp_path):
    llm = MagicMock()
    llm.complete_json.return_value = {
        "subtopics": [
            {"name": "Scaled Dot-Product Attention", "description": "The core formula Q K^T / sqrt(d_k) V"},
            {"name": "Multi-Head Attention", "description": "Running attention in parallel across heads"},
            {"name": "Positional Encoding", "description": "Sine/cosine embeddings added to input"},
        ]
    }
    result = split_topic_into_subtopics(_make_group(), llm, n_subtopics=3, base_dir=tmp_path)
    assert len(result) == 3
    assert result[0]["name"] == "Scaled Dot-Product Attention"
    assert "slug" in result[0]
    assert "description" in result[0]


def test_split_topic_is_checkpointed(tmp_path):
    llm = MagicMock()
    llm.complete_json.return_value = {
        "subtopics": [{"name": "Sub A", "description": "desc A"}]
    }
    split_topic_into_subtopics(_make_group(), llm, base_dir=tmp_path)
    split_topic_into_subtopics(_make_group(), llm, base_dir=tmp_path)
    assert llm.complete_json.call_count == 1


def test_split_all_topics_returns_groups_with_subtopics(tmp_path):
    llm = MagicMock()
    llm.complete_json.return_value = {
        "subtopics": [{"name": "Sub", "description": "desc"}]
    }
    groups = [_make_group("Topic A", "topic-a"), _make_group("Topic B", "topic-b")]
    result = split_all_topics(groups, llm, n_subtopics=1, base_dir=tmp_path)
    assert len(result) == 2
    assert len(result[0]["subtopics"]) == 1
    assert len(result[1]["subtopics"]) == 1
