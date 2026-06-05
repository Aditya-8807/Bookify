from utils.rag_index import build_index, query_chunks, index_book_chapters


def _make_transcripts():
    return [
        {
            "video_id": "v1", "title": "Attention Explained", "full_text": "",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "the attention mechanism computes a weighted sum of values"},
                {"start": 5.0, "end": 10.0, "text": "query key and value matrices are learned during training"},
            ],
            "corrections": [],
        },
        {
            "video_id": "v2", "title": "Backprop Basics", "full_text": "",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "backpropagation computes gradients using the chain rule"},
            ],
            "corrections": [],
        },
    ]


def test_build_index_creates_collection(tmp_path):
    col = build_index(_make_transcripts(), persist_dir=str(tmp_path / "idx"))
    assert col.count() == 3


def test_build_index_is_idempotent(tmp_path):
    persist = str(tmp_path / "idx")
    col1 = build_index(_make_transcripts(), persist_dir=persist)
    col2 = build_index(_make_transcripts(), persist_dir=persist)
    assert col2.count() == 3


def test_query_returns_relevant_chunks(tmp_path):
    col = build_index(_make_transcripts(), persist_dir=str(tmp_path / "idx"))
    chunks = query_chunks(col, "attention query key value", n_results=2)
    assert len(chunks) == 2
    texts = [c["text"] for c in chunks]
    assert any("attention" in t or "query" in t for t in texts)


def test_query_chunk_has_required_fields(tmp_path):
    col = build_index(_make_transcripts(), persist_dir=str(tmp_path / "idx"))
    chunks = query_chunks(col, "backpropagation gradients", n_results=1)
    assert "text" in chunks[0]
    assert "video_id" in chunks[0]
    assert "title" in chunks[0]
    assert "start" in chunks[0]


def test_index_book_chapters_adds_to_collection(tmp_path):
    persist = str(tmp_path / "idx")
    col = build_index(_make_transcripts(), persist_dir=persist)
    chapters = [{"name": "Ch1", "slug": "ch1", "prose": "attention is about weighting values by relevance"}]
    index_book_chapters(col, chapters)
    assert col.count() == 4
