from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from pipeline import CorrectedTranscript, TopicGroup
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


OVERLAP_SYSTEM = """You are analysing topic groups for a technical book.
Identify concepts that appear in MORE THAN ONE topic group.
Return JSON:
{
  "overlaps": [
    {
      "concept": "concept name",
      "primary_topic": "topic where it should be fully explained",
      "secondary_topics": ["topics where it should only be cross-referenced"]
    }
  ]
}
Return an empty overlaps list if no overlaps found."""


WRITE_SYSTEM = """You are writing a section of a technical book about building LLMs from scratch.
Write clear, educational prose — not bullet points, not a transcript summary.
Structure: opening context → concept explanation → worked examples → section summary.
Strip first-person instructor voice ("In this video I will...").
Use consistent terminology throughout.
When citing something from the transcript, add: [Video: "<title>" @ MM:SS]
When citing a reference, add: [<URL>]
Where a concept was introduced in a prior section, add: "As introduced in <Topic>..."
Do NOT invent facts. Only write what is supported by the provided transcript and references."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def detect_overlaps(groups: List[TopicGroup], llm_client) -> List[Dict]:
    topics_summary = "\n".join(
        f"Topic: {g['name']} — videos: {g['video_ids']}" for g in groups
    )
    result = llm_client.complete_json(system=OVERLAP_SYSTEM, user=topics_summary)
    return result.get("overlaps", [])


def write_topic(
    group: TopicGroup,
    transcripts: List[CorrectedTranscript],
    overlaps_map: Dict[str, Dict],
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> Dict[str, Any]:
    slug = group["slug"]
    if checkpoint_exists("04_topics", slug, base_dir=base_dir):
        return load_checkpoint("04_topics", slug, base_dir=base_dir)

    trans_by_id = {t["video_id"]: t for t in transcripts}
    group_transcripts = [trans_by_id[vid] for vid in group["video_ids"] if vid in trans_by_id]

    transcript_text = "\n\n".join(
        f"=== {t['title']} ===\n" +
        "\n".join(f"[{_fmt_ts(s['start'])}] {s['text']}" for s in t["segments"])
        for t in group_transcripts
    )

    refs_text = ""
    for url, content in group["ref_contents"].items():
        refs_text += f"\n\n--- Reference: {url} ---\n{content[:3000]}"

    cross_refs = overlaps_map.get(group["name"], {})
    cross_ref_note = ""
    if cross_refs:
        concepts = ", ".join(cross_refs.keys())
        cross_ref_note = (
            f"\nNote: these concepts were introduced in prior sections and should be "
            f"cross-referenced, not re-explained: {concepts}"
        )

    user = (
        f"Topic: {group['name']}\n"
        f"Prerequisite topics already covered: {group['prerequisites']}\n"
        f"{cross_ref_note}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n"
        f"{refs_text}"
    )

    prose = llm_client.complete(system=WRITE_SYSTEM, user=user)

    result = {"name": slug, "slug": slug, "prose": prose}
    save_checkpoint("04_topics", slug, result, base_dir=base_dir)
    return result


def write_all_topics(
    groups: List[TopicGroup],
    transcripts: List[CorrectedTranscript],
    llm_client,
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[Dict]:
    overlaps = detect_overlaps(groups, llm_client)
    overlaps_map: Dict[str, Dict] = {}
    for ov in overlaps:
        for sec in ov.get("secondary_topics", []):
            overlaps_map.setdefault(sec, {})[ov["concept"]] = ov["primary_topic"]

    if progress:
        progress.add_stage("Stage 4: Write Topics", total=len(groups))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(write_topic, g, transcripts, overlaps_map, llm_client, base_dir): g
            for g in groups
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 4: Write Topics")

    slug_order = {g["slug"]: g["dependency_order"] for g in groups}
    return sorted(results, key=lambda x: slug_order.get(x["slug"], 0))
