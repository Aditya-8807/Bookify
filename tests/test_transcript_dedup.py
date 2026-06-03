from pipeline.transcript_dedup import dedup_segments, dedup_all


def _make_transcript(segs):
    return {"video_id": "v1", "title": "T", "full_text": "", "segments": segs, "corrections": []}


def test_dedup_removes_near_duplicate_segment():
    segs = [
        {"start": 0.0, "end": 3.0, "text": "attention is all you need the transformer architecture"},
        {"start": 5.0, "end": 8.0, "text": "attention is all you need the transformer architecture"},
        {"start": 10.0, "end": 13.0, "text": "backpropagation computes gradients via chain rule"},
    ]
    result = dedup_segments(_make_transcript(segs))
    assert len(result["segments"]) == 2


def test_dedup_keeps_similar_but_distinct_segments():
    segs = [
        {"start": 0.0, "end": 3.0, "text": "attention mechanism in transformers uses query key value"},
        {"start": 5.0, "end": 8.0, "text": "attention mechanism also applies in vision transformers"},
    ]
    result = dedup_segments(_make_transcript(segs), threshold=0.9)
    assert len(result["segments"]) == 2


def test_dedup_preserves_transcript_fields():
    segs = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    t = _make_transcript(segs)
    t["corrections"] = [{"old": "helo", "new": "hello"}]
    result = dedup_segments(t)
    assert result["video_id"] == "v1"
    assert result["corrections"] == [{"old": "helo", "new": "hello"}]


def test_dedup_all_processes_list():
    t1 = _make_transcript([
        {"start": 0.0, "end": 2.0, "text": "same text same text same text same"},
        {"start": 3.0, "end": 5.0, "text": "same text same text same text same"},
    ])
    t2 = _make_transcript([
        {"start": 0.0, "end": 2.0, "text": "different content about neural networks"},
    ])
    results = dedup_all([t1, t2])
    assert len(results) == 2
    assert len(results[0]["segments"]) == 1
    assert len(results[1]["segments"]) == 1
