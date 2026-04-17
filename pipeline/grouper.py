import re
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

from pipeline import CorrectedTranscript, VideoMeta, TopicGroup
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


GROUP_SYSTEM = """You are organizing transcripts from a YouTube playlist into a book.
Cluster the videos into thematic topic groups. Each group becomes one section of the book.
Return JSON:
{
  "topics": [
    {
      "name": "Human-readable topic name",
      "video_ids": ["vid1", "vid2"]
    }
  ]
}
Name topics concisely (e.g. "Tokenization & Vocabulary", "The Attention Mechanism").
Ensure all video_ids appear exactly once."""


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s.strip())
    return s


def fetch_reference_content(urls: List[str]) -> Dict[str, str]:
    content = {}
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Bookify/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            content[url] = text[:8000]
        except Exception:
            pass
    return content


def fetch_and_checkpoint_ref_content(
    video_metas: List[VideoMeta],
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> None:
    """
    Fetch reference URL content for every video and save to
    checkpoints/01b_ref_content/<video_id>.json BEFORE any LLM stage runs.
    This gives a permanent, auditable record of what raw source material
    was available to the LLM — independent of any LLM output.
    """
    if progress:
        progress.add_stage("Stage 1b: Ref Content", total=len(video_metas))

    for meta in video_metas:
        vid = meta["video_id"]
        if not checkpoint_exists("01b_ref_content", vid, base_dir=base_dir):
            content = fetch_reference_content(meta.get("ref_urls", []))
            save_checkpoint("01b_ref_content", vid, content, base_dir=base_dir)
        if progress:
            progress.advance("Stage 1b: Ref Content")


def group_and_order(
    transcripts: List[CorrectedTranscript],
    video_metas: List[VideoMeta],
    llm_client,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[TopicGroup]:
    if checkpoint_exists("03_groups", "groups", base_dir=base_dir):
        return load_checkpoint("03_groups", "groups", base_dir=base_dir)

    if progress:
        progress.add_stage("Stage 3: Group + Order", total=1)

    meta_by_id = {m["video_id"]: m for m in video_metas}
    # Pass playlist_index alongside each video so the LLM knows the original order.
    summaries = "\n\n".join(
        f"video_id={t['video_id']} playlist_index={meta_by_id.get(t['video_id'], {}).get('playlist_index', 0)} title={t['title']}\n{t['full_text'][:500]}"
        for t in transcripts
    )
    result = llm_client.complete_json(system=GROUP_SYSTEM, user=summaries)
    raw_topics = result.get("topics", [])

    # Sort topic groups by the earliest playlist_index among their videos.
    # The instructor's lecture order is the intentional pedagogical flow —
    # trusting it is safer than having the LLM invent its own dependency graph.
    def _min_playlist_index(topic: Dict) -> int:
        indices = [
            meta_by_id.get(vid, {}).get("playlist_index", 0)
            for vid in topic.get("video_ids", [])
        ]
        return min(indices) if indices else 0

    ordered = sorted(raw_topics, key=_min_playlist_index)

    groups: List[TopicGroup] = []
    for i, topic in enumerate(ordered):
        # Sort videos within the group by playlist_index so transcript chunks
        # are fed to the writer in the same order the instructor taught them.
        vids = sorted(
            topic["video_ids"],
            key=lambda v: meta_by_id.get(v, {}).get("playlist_index", 0),
        )
        all_ref_urls = list({url for vid in vids for url in meta_by_id.get(vid, {}).get("ref_urls", [])})

        # Load ref content from pre-LLM checkpoint instead of fetching here.
        ref_contents: Dict[str, str] = {}
        for vid in vids:
            if checkpoint_exists("01b_ref_content", vid, base_dir=base_dir):
                ref_contents.update(load_checkpoint("01b_ref_content", vid, base_dir=base_dir))

        groups.append({
            "name": topic["name"],
            "slug": slugify(topic["name"]),
            "video_ids": vids,
            "dependency_order": i,
            "prerequisites": [],
            "ref_urls": all_ref_urls,
            "ref_contents": ref_contents,
        })

    save_checkpoint("03_groups", "groups", groups, base_dir=base_dir)
    if progress:
        progress.advance("Stage 3: Group + Order")
    return groups
