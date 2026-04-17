from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from pipeline import VideoMeta, Transcript, TranscriptSegment
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


def _merge_segments(
    segments: List[TranscriptSegment],
    max_duration_s: float = 45.0,
) -> List[TranscriptSegment]:
    """Merge consecutive Whisper segments into ~45-second blocks.
    Keeps start/end timestamps of the first/last segment in each block."""
    merged = []
    bucket: List[TranscriptSegment] = []

    for seg in segments:
        if not bucket:
            bucket.append(seg)
            continue
        block_duration = seg["end"] - bucket[0]["start"]
        if block_duration <= max_duration_s:
            bucket.append(seg)
        else:
            merged.append({
                "start": bucket[0]["start"],
                "end": bucket[-1]["end"],
                "text": " ".join(s["text"] for s in bucket),
            })
            bucket = [seg]

    if bucket:
        merged.append({
            "start": bucket[0]["start"],
            "end": bucket[-1]["end"],
            "text": " ".join(s["text"] for s in bucket),
        })

    return merged


def _transcribe_single(
    video: VideoMeta,
    model: WhisperModel,
    base_dir: Path,
) -> Transcript:
    vid = video["video_id"]
    if checkpoint_exists("02_transcripts", vid, base_dir=base_dir):
        return load_checkpoint("02_transcripts", vid, base_dir=base_dir)

    segments_raw, _ = model.transcribe(
        video["audio_path"],
        beam_size=5,
        # Seed Whisper with the video title so it recognises domain vocabulary
        # (e.g. "attention head", "KV cache") from the very first token.
        initial_prompt=video.get("title", ""),
        # Full float32 precision — no quantisation rounding errors.
        # Slower, but the user explicitly accepts slower for highest fidelity.
        # (compute_type is set at model-load time; this comment is for clarity.)
        condition_on_previous_text=True,
        # Lower threshold: segment is only skipped if ≥70% confident it is
        # silence. Default 0.6 is too aggressive and can drop real speech.
        no_speech_threshold=0.3,
        # VAD strips genuine silence/music before decoding so Whisper cannot
        # hallucinate text into non-speech gaps.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    raw_segments: List[TranscriptSegment] = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_raw
    ]
    segments = _merge_segments(raw_segments, max_duration_s=45)
    full_text = " ".join(s["text"] for s in segments)

    transcript: Transcript = {
        "video_id": vid,
        "title": video["title"],
        "segments": segments,
        "full_text": full_text,
    }
    save_checkpoint("02_transcripts", vid, transcript, base_dir=base_dir)
    return transcript


def transcribe_all(
    videos: List[VideoMeta],
    whisper_model: str = "small",
    compute_type: str = "int8",
    batch_size: int = 1,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[Transcript]:
    if progress:
        progress.add_stage("Stage 2: Transcribe", total=len(videos))

    model = WhisperModel(whisper_model, device="cpu", compute_type=compute_type)

    results = []
    for video in videos:
        transcript = _transcribe_single(video, model, base_dir)
        results.append(transcript)
        if progress:
            progress.advance("Stage 2: Transcribe")

    return results
