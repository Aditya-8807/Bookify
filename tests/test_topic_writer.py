import pytest
from unittest.mock import MagicMock
from pipeline.topic_writer import detect_overlaps, write_topic, write_all_topics


def test_detect_overlaps_finds_shared_concepts():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "overlaps": [
            {
                "concept": "softmax function",
                "primary_topic": "The Attention Mechanism",
                "secondary_topics": ["Training from Scratch"],
            }
        ]
    }
    groups = [
        {"name": "The Attention Mechanism", "slug": "attention", "video_ids": ["v1"],
         "dependency_order": 0, "prerequisites": [], "ref_urls": [], "ref_contents": {}},
        {"name": "Training from Scratch", "slug": "training", "video_ids": ["v2"],
         "dependency_order": 1, "prerequisites": [], "ref_urls": [], "ref_contents": {}},
    ]
    overlaps = detect_overlaps(groups, mock_client)
    assert len(overlaps) == 1
    assert overlaps[0]["concept"] == "softmax function"


def test_write_topic_skips_if_checkpoint_exists(tmp_checkpoints):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("04_topics", "attention", {"name": "attention", "prose": "..."}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    group = {"name": "Attention", "slug": "attention", "video_ids": [],
             "dependency_order": 0, "prerequisites": [], "ref_urls": [], "ref_contents": {}}
    result = write_topic(group, [], {}, mock_client, base_dir=tmp_checkpoints)
    mock_client.complete.assert_not_called()
    assert result["name"] == "attention"


def test_write_topic_calls_llm_with_transcript(tmp_checkpoints):
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "Attention mechanisms allow the model to focus on relevant tokens. "
        "[Video: \"The Attention Mechanism\" @ 12:34]"
    )
    group = {
        "name": "The Attention Mechanism", "slug": "the-attention-mechanism",
        "video_ids": ["abc123"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [{"video_id": "abc123", "title": "The Attention Mechanism",
                    "segments": [{"start": 752.0, "end": 760.0,
                                  "text": "attention allows the model to focus"}],
                    "full_text": "attention allows the model to focus", "corrections": []}]
    result = write_topic(group, transcripts, {}, mock_client, base_dir=tmp_checkpoints)
    assert "attention" in result["prose"].lower()
    mock_client.complete.assert_called_once()


def test_write_topic_retries_when_draft_too_short(tmp_checkpoints):
    short_draft = "Too short."
    full_draft = " ".join(["word"] * 2600)
    mock_client = MagicMock()
    mock_client.complete.side_effect = [short_draft, full_draft]

    # Build a transcript large enough that source_word_budget >= min_words
    long_text = " ".join(["token"] * 2600)
    group = {
        "name": "The Attention Mechanism", "slug": "attention-retry",
        "video_ids": ["abc123"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [{"video_id": "abc123", "title": "The Attention Mechanism",
                    "segments": [{"start": 0.0, "end": 10.0, "text": long_text}],
                    "full_text": long_text, "corrections": []}]
    result = write_topic(group, transcripts, {}, mock_client, min_words=2500, base_dir=tmp_checkpoints)
    assert mock_client.complete.call_count == 2
    assert result["prose"] == full_draft


from unittest.mock import MagicMock
from pipeline.topic_writer import write_subtopic, write_topic_with_subtopics


def _make_group_with_subtopics():
    return {
        "name": "Attention Mechanism", "slug": "attention-mechanism",
        "video_ids": ["v1"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
        "subtopics": [
            {"name": "Scaled Dot-Product", "slug": "scaled-dot-product", "description": "The core formula"},
            {"name": "Multi-Head Attention", "slug": "multi-head-attention", "description": "Parallel heads"},
        ],
    }


def _make_rag_col():
    rag_col = MagicMock()
    rag_col.query.return_value = {
        "documents": [["attention uses query key value", "softmax normalizes weights"]],
        "metadatas": [[
            {"video_id": "v1", "title": "T1", "start": 0.0},
            {"video_id": "v1", "title": "T1", "start": 5.0},
        ]],
    }
    return rag_col


def test_write_subtopic_uses_rag_chunks(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "This is a detailed explanation of scaled dot product attention " * 100
    rag_col = _make_rag_col()
    group = _make_group_with_subtopics()
    subtopic = group["subtopics"][0]
    result = write_subtopic(group, subtopic, rag_col, llm, base_dir=tmp_path)
    assert isinstance(result, str)
    assert len(result) > 0
    rag_col.query.assert_called_once()


def test_write_subtopic_checkpoints_result(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "content " * 2500  # > 2000 words, no retry triggered
    rag_col = _make_rag_col()
    group = _make_group_with_subtopics()
    subtopic = group["subtopics"][0]
    write_subtopic(group, subtopic, rag_col, llm, base_dir=tmp_path)
    write_subtopic(group, subtopic, rag_col, llm, base_dir=tmp_path)
    assert llm.complete.call_count == 1


def test_write_topic_with_subtopics_assembles_sections(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "detailed section content " * 200
    rag_col = _make_rag_col()
    group = _make_group_with_subtopics()
    result = write_topic_with_subtopics(group, rag_col, llm, base_dir=tmp_path)
    assert result["name"] == "Attention Mechanism"
    assert "Scaled Dot-Product" in result["prose"]
    assert "Multi-Head Attention" in result["prose"]


from pipeline.topic_writer import coverage_pass


def _make_corrected_transcript(video_id, segs):
    return {
        "video_id": video_id, "title": "T", "full_text": "",
        "segments": [{"start": float(i), "end": float(i+1), "text": s} for i, s in enumerate(segs)],
        "corrections": [],
    }


def test_coverage_pass_appends_when_uncovered_content_exists(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "Additional content about completely new material here for the chapter."
    rag_col = MagicMock()
    group = {
        "name": "Topic", "slug": "topic", "video_ids": ["v1"],
        "subtopics": [], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    # Segments with words that do NOT appear in the prose
    transcripts = [_make_corrected_transcript("v1", ["xylophone zymurgy quixotic"] * 20)]
    prose = "This is the written chapter content about attention transformers."
    result = coverage_pass(group, transcripts, prose, rag_col, llm, target_coverage=0.99, base_dir=tmp_path)
    assert len(result) > len(prose)
    llm.complete.assert_called_once()


def test_coverage_pass_skips_when_coverage_sufficient(tmp_path):
    llm = MagicMock()
    rag_col = MagicMock()
    group = {
        "name": "Topic", "slug": "topic", "video_ids": ["v1"],
        "subtopics": [], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    # Use words that ARE in the prose
    transcripts = [_make_corrected_transcript("v1", ["attention transformer architecture neural"] * 5)]
    prose = "attention transformer architecture neural network deep learning"
    result = coverage_pass(group, transcripts, prose, rag_col, llm, target_coverage=0.5, base_dir=tmp_path)
    llm.complete.assert_not_called()
    assert result == prose


def test_coverage_pass_returns_prose_unchanged_when_no_segments(tmp_path):
    llm = MagicMock()
    rag_col = MagicMock()
    group = {
        "name": "Topic", "slug": "topic", "video_ids": ["v1"],
        "subtopics": [], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [_make_corrected_transcript("v1", [])]  # empty segments
    prose = "Some prose content here."
    result = coverage_pass(group, transcripts, prose, rag_col, llm, base_dir=tmp_path)
    assert result == prose
    llm.complete.assert_not_called()
