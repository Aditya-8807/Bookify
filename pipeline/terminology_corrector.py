from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from pipeline import Transcript, CorrectedTranscript
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


SYSTEM_PROMPT = """You are a technical transcript corrector for AI/ML content.
Fix misheard or garbled domain-specific terms in these transcript segments.
The video title gives you domain context.
Common errors: "a tension" → "attention", "cave cache" → "KV cache",
"soft max" → "softmax", "gradient decent" → "gradient descent",
"embedding" (already correct), "layer nor" → "layer norm".
Return JSON:
{
  "corrected_segments": [{"start": float, "end": float, "text": "corrected text"}],
  "corrections": [{"original": "wrong", "corrected": "right", "timestamp": "MM:SS"}]
}
Only fix misheard technical terms. Do not rephrase or alter meaning."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def correct_transcript(
    transcript: Transcript,
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> CorrectedTranscript:
    vid = transcript["video_id"]
    if checkpoint_exists("02b_corrected", vid, base_dir=base_dir):
        return load_checkpoint("02b_corrected", vid, base_dir=base_dir)

    segments_text = "\n".join(
        f"[{_fmt_ts(s['start'])}] {s['text']}" for s in transcript["segments"]
    )
    user = f"Video title: {transcript['title']}\n\nSegments:\n{segments_text}"
    result = llm_client.complete_json(system=SYSTEM_PROMPT, user=user)

    corrected_segs = result.get("corrected_segments", transcript["segments"])
    corrected: CorrectedTranscript = {
        "video_id": vid,
        "title": transcript["title"],
        "segments": corrected_segs,
        "full_text": " ".join(s["text"] for s in corrected_segs),
        "corrections": result.get("corrections", []),
    }
    save_checkpoint("02b_corrected", vid, corrected, base_dir=base_dir)
    return corrected


def correct_all(
    transcripts: List[Transcript],
    llm_client,
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[CorrectedTranscript]:
    if progress:
        progress.add_stage("Stage 2b: Terminology", total=len(transcripts))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(correct_transcript, t, llm_client, base_dir): t
            for t in transcripts
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 2b: Terminology")

    return results
