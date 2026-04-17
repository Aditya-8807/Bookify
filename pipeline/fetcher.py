from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse, parse_qs

from yt_dlp import YoutubeDL

from pipeline import VideoMeta
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from utils.url_filter import filter_description_urls


def _playlist_url(url: str) -> str:
    """Extract pure playlist URL from any YouTube URL format."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "list" in params:
        return f"https://www.youtube.com/playlist?list={params['list'][0]}"
    return url


def extract_playlist_videos(playlist_url: str) -> List[Dict]:
    """Fetch all video metadata from a playlist without downloading."""
    clean_url = _playlist_url(playlist_url)
    ydl_opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=False)
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
) -> VideoMeta:
    vid = video["video_id"]
    if checkpoint_exists("01_fetch", vid, base_dir=base_dir):
        return load_checkpoint("01_fetch", vid, base_dir=base_dir)

    video_url = f"https://www.youtube.com/watch?v={vid}"
    meta_opts = {"quiet": True, "skip_download": True}
    with YoutubeDL(meta_opts) as ydl:
        full_info = ydl.extract_info(video_url, download=False) or {}
    description = full_info.get("description", video.get("description", ""))

    ref_urls = filter_description_urls(description, llm_client)

    meta: VideoMeta = {
        "video_id": vid,
        "title": video.get("title", full_info.get("title", "")),
        "description": description,
        "playlist_index": video["playlist_index"],
        "ref_urls": ref_urls,
    }
    save_checkpoint("01_fetch", vid, meta, base_dir=base_dir)
    return meta


def fetch_all(
    playlist_url: str,
    llm_client,
    batch_size: int = 1,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[VideoMeta]:
    videos = extract_playlist_videos(playlist_url)
    if progress:
        progress.add_stage("Stage 1: Fetch", total=len(videos))

    results = []
    for video in videos:
        results.append(fetch_video_audio(video, llm_client, base_dir))
        if progress:
            progress.advance("Stage 1: Fetch")

    return results
