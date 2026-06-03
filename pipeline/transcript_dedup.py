import re
from typing import List
from pipeline import CorrectedTranscript


def _tokenize(text: str) -> set:
    return set(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup_segments(transcript: CorrectedTranscript, threshold: float = 0.85) -> CorrectedTranscript:
    seen: List[set] = []
    kept = []
    for seg in transcript["segments"]:
        words = _tokenize(seg["text"])
        if any(_jaccard(words, s) >= threshold for s in seen):
            continue
        seen.append(words)
        kept.append(seg)
    return {**transcript, "segments": kept}


def dedup_all(transcripts: List[CorrectedTranscript], threshold: float = 0.85) -> List[CorrectedTranscript]:
    return [dedup_segments(t, threshold) for t in transcripts]
