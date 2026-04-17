from pathlib import Path
from typing import List, Dict

from pipeline import Transcript, CorrectedTranscript
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists

CHUNK_SIZE = 50  # segments per LLM call — keeps prompts well within token limits

SYSTEM_PROMPT = """You are a technical transcript corrector for AI/ML content.
Fix misheard or garbled domain-specific terms in these transcript segments.
The video title gives you domain context.
Common errors: "a tension" → "attention", "cave cache" → "KV cache",
"soft max" → "softmax", "gradient decent" → "gradient descent",
"layer nor" → "layer norm", "back prop" → "backprop", "norm" → "norm".
Return JSON:
{
  "corrected_segments": [{"start": float, "end": float, "text": "corrected text"}],
  "corrections": [{"original": "wrong", "corrected": "right", "timestamp": "MM:SS"}]
}
Only fix misheard technical terms. Do not rephrase or alter meaning."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _correct_chunk(title: str, segments: List[Dict], llm_client) -> tuple:
    """Correct one chunk of segments. Returns (corrected_segments, corrections)."""
    segments_text = "\n".join(
        f"[{_fmt_ts(s['start'])}] {s['text']}" for s in segments
    )
    user = f"Video title: {title}\n\nSegments:\n{segments_text}"
    result = llm_client.complete_json(system=SYSTEM_PROMPT, user=user)
    corrected_segs = result.get("corrected_segments", segments)
    corrections = result.get("corrections", [])
    # Ensure corrected_segs has same length as input — fall back per-segment if needed
    if len(corrected_segs) != len(segments):
        corrected_segs = segments
    return corrected_segs, corrections


def correct_transcript(
    transcript: Transcript,
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> CorrectedTranscript:
    vid = transcript["video_id"]
    if checkpoint_exists("02b_corrected", vid, base_dir=base_dir):
        return load_checkpoint("02b_corrected", vid, base_dir=base_dir)

    segments = transcript["segments"]
    all_corrected: List[Dict] = []
    all_corrections: List[Dict] = []

    # Process in chunks to stay within token limits for long lectures.
    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i: i + CHUNK_SIZE]
        corrected_chunk, corrections = _correct_chunk(transcript["title"], chunk, llm_client)
        all_corrected.extend(corrected_chunk)
        all_corrections.extend(corrections)

    corrected: CorrectedTranscript = {
        "video_id": vid,
        "title": transcript["title"],
        "segments": all_corrected,
        "full_text": " ".join(s["text"] for s in all_corrected),
        "corrections": all_corrections,
    }
    save_checkpoint("02b_corrected", vid, corrected, base_dir=base_dir)
    return corrected


def correct_all(
    transcripts: List[Transcript],
    llm_client,
    batch_size: int = 1,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[CorrectedTranscript]:
    if progress:
        progress.add_stage("Stage 3: Terminology", total=len(transcripts))

    results = []
    for transcript in transcripts:
        results.append(correct_transcript(transcript, llm_client, base_dir))
        if progress:
            progress.advance("Stage 3: Terminology")

    return results
