from pathlib import Path
from typing import List

import ssl
import certifi
import requests
from youtube_transcript_api import YouTubeTranscriptApi as _YTApi

from pipeline import VideoMeta, Transcript, TranscriptSegment
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


def _merge_segments(
    segments: List[TranscriptSegment],
    max_duration_s: float = 45.0,
) -> List[TranscriptSegment]:
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


def _fetch_transcript(video: VideoMeta, base_dir: Path) -> Transcript:
    vid = video["video_id"]
    if checkpoint_exists("02_transcripts", vid, base_dir=base_dir):
        return load_checkpoint("02_transcripts", vid, base_dir=base_dir)

    session = requests.Session()
    session.verify = certifi.where()
    fetched = _YTApi(http_client=session).fetch(vid)
    segments: List[TranscriptSegment] = [
        {
            "start": snippet.start,
            "end": snippet.start + snippet.duration,
            "text": snippet.text.strip(),
        }
        for snippet in fetched
    ]
    segments = _merge_segments(segments, max_duration_s=45)
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
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[Transcript]:
    if progress:
        progress.add_stage("Stage 2: Transcribe", total=len(videos))

    results = []
    for video in videos:
        transcript = _fetch_transcript(video, base_dir)
        results.append(transcript)
        if progress:
            progress.advance("Stage 2: Transcribe")

    return results
