from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from yt_dlp import YoutubeDL

from pipeline import VideoMeta
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from utils.url_filter import filter_description_urls


def extract_playlist_videos(playlist_url: str) -> List[Dict]:
    """Fetch all video metadata from a playlist without downloading."""
    ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    return [
        {
            "video_id": entry["id"],
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "playlist_index": entry.get("playlist_index", i),
        }
        for i, entry in enumerate(info.get("entries", []))
    ]


def fetch_video_audio(
    video: Dict,
    llm_client,
    base_dir: Path = Path("checkpoints"),
    audio_dir: Path = Path("checkpoints/audio"),
) -> VideoMeta:
    vid = video["video_id"]
    if checkpoint_exists("01_fetch", vid, base_dir=base_dir):
        return load_checkpoint("01_fetch", vid, base_dir=base_dir)

    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(audio_dir / f"{vid}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / f"{vid}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={vid}"])

    ref_urls = filter_description_urls(video.get("description", ""), llm_client)

    meta: VideoMeta = {
        "video_id": vid,
        "title": video["title"],
        "description": video.get("description", ""),
        "playlist_index": video["playlist_index"],
        "ref_urls": ref_urls,
        "audio_path": audio_path,
    }
    save_checkpoint("01_fetch", vid, meta, base_dir=base_dir)
    return meta


def fetch_all(
    playlist_url: str,
    llm_client,
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    audio_dir: Path = Path("checkpoints/audio"),
    progress=None,
) -> List[VideoMeta]:
    videos = extract_playlist_videos(playlist_url)
    if progress:
        progress.add_stage("Stage 1: Fetch", total=len(videos))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(fetch_video_audio, v, llm_client, base_dir, audio_dir): v
            for v in videos
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 1: Fetch")

    return sorted(results, key=lambda x: x["playlist_index"])
