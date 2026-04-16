import re
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

from pipeline import CorrectedTranscript, VideoMeta, TopicGroup
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


GROUP_SYSTEM = """You are organizing transcripts from a YouTube playlist into a book.
Cluster the videos into thematic topic groups. Each group becomes one section of the book.
Build a dependency graph: if topic B requires understanding topic A first, list A as a prerequisite of B.
Return JSON:
{
  "topics": [
    {
      "name": "Human-readable topic name",
      "video_ids": ["vid1", "vid2"],
      "prerequisites": ["Topic Name A"]
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


def _topological_sort(topics: List[Dict]) -> List[Dict]:
    name_to_topic = {t["name"]: t for t in topics}
    order = []
    visited = set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        for prereq in name_to_topic.get(name, {}).get("prerequisites", []):
            if prereq in name_to_topic:
                visit(prereq)
        order.append(name)

    for t in topics:
        visit(t["name"])

    return [name_to_topic[n] for n in order if n in name_to_topic]


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
    summaries = "\n\n".join(
        f"video_id={t['video_id']} title={t['title']}\n{t['full_text'][:500]}"
        for t in transcripts
    )
    result = llm_client.complete_json(system=GROUP_SYSTEM, user=summaries)
    raw_topics = result.get("topics", [])
    ordered = _topological_sort(raw_topics)

    groups: List[TopicGroup] = []
    for i, topic in enumerate(ordered):
        vids = topic["video_ids"]
        all_ref_urls = list({url for vid in vids for url in meta_by_id.get(vid, {}).get("ref_urls", [])})
        ref_contents = fetch_reference_content(all_ref_urls)
        groups.append({
            "name": topic["name"],
            "slug": slugify(topic["name"]),
            "video_ids": vids,
            "dependency_order": i,
            "prerequisites": topic.get("prerequisites", []),
            "ref_urls": all_ref_urls,
            "ref_contents": ref_contents,
        })

    save_checkpoint("03_groups", "groups", groups, base_dir=base_dir)
    if progress:
        progress.advance("Stage 3: Group + Order")
    return groups
