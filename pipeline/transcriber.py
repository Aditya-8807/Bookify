import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from pipeline import VideoMeta, Transcript, TranscriptSegment
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


def transcribe_video(
    video: VideoMeta,
    whisper_model: str = "large-v3",
    base_dir: Path = Path("checkpoints"),
) -> Transcript:
    vid = video["video_id"]
    if checkpoint_exists("02_transcripts", vid, base_dir=base_dir):
        return load_checkpoint("02_transcripts", vid, base_dir=base_dir)

    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    segments_raw, _ = model.transcribe(video["audio_path"], beam_size=5)

    segments: List[TranscriptSegment] = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_raw
    ]
    full_text = " ".join(s["text"] for s in segments)

    audio_path = video.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

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
    whisper_model: str = "large-v3",
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[Transcript]:
    if progress:
        progress.add_stage("Stage 2: Transcribe", total=len(videos))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(transcribe_video, v, whisper_model, base_dir): v
            for v in videos
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 2: Transcribe")

    return sorted(results, key=lambda x: x["video_id"])
