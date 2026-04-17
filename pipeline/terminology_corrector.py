import re
from pathlib import Path
from typing import List, Dict

from pipeline import Transcript, CorrectedTranscript
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists

CHUNK_SIZE = 50  # segments per LLM call — keeps prompts well within token limits

SYSTEM_PROMPT = """You are a technical transcript corrector for AI/ML content.
Identify misheard or garbled domain-specific terms in these transcript segments.
The video title gives you domain context.
Common errors: "a tension" → "attention", "cave cache" → "KV cache",
"soft max" → "softmax", "gradient decent" → "gradient descent",
"layer nor" → "layer norm", "back prop" → "backprop".
Return ONLY the corrections needed as JSON:
{
  "corrections": [{"original": "wrong phrase", "corrected": "right phrase", "timestamp": "MM:SS"}]
}
Return {"corrections": []} if nothing needs fixing. Max 30 corrections per chunk.
Do NOT return the full segments — only the corrections list."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _apply_corrections(segments: List[Dict], corrections: List[Dict]) -> List[Dict]:
    """Apply correction substitutions to segment text using word-boundary matching."""
    if not corrections:
        return segments
    result = []
    for seg in segments:
        text = seg["text"]
        for c in corrections:
            text = re.sub(
                r'\b' + re.escape(c["original"]) + r'\b',
                c["corrected"],
                text,
                flags=re.IGNORECASE,
            )
        result.append({**seg, "text": text})
    return result


def _correct_chunk(title: str, segments: List[Dict], llm_client) -> List[Dict]:
    """Return corrections list for one chunk of segments."""
    segments_text = "\n".join(
        f"[{_fmt_ts(s['start'])}] {s['text']}" for s in segments
    )
    user = f"Video title: {title}\n\nSegments:\n{segments_text}"
    result = llm_client.complete_json(system=SYSTEM_PROMPT, user=user)
    return result.get("corrections", [])


def correct_transcript(
    transcript: Transcript,
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> CorrectedTranscript:
    vid = transcript["video_id"]
    if checkpoint_exists("02b_corrected", vid, base_dir=base_dir):
        return load_checkpoint("02b_corrected", vid, base_dir=base_dir)

    segments = transcript["segments"]
    all_corrections: List[Dict] = []

    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i: i + CHUNK_SIZE]
        all_corrections.extend(_correct_chunk(transcript["title"], chunk, llm_client))

    corrected_segments = _apply_corrections(segments, all_corrections)

    corrected: CorrectedTranscript = {
        "video_id": vid,
        "title": transcript["title"],
        "segments": corrected_segments,
        "full_text": " ".join(s["text"] for s in corrected_segments),
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
