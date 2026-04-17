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
Heading levels: use ## for section headings, ### for subsections. Never use # (h1) — those are reserved for chapter titles added by the book assembler.
Use consistent terminology throughout.
When citing something from the transcript, add: [Video: "<title>" @ MM:SS]
When citing a reference, add: [<URL>]
Where a concept was introduced in a prior section, add: "As introduced in <Topic>..."
Do NOT invent facts. Only write what is supported by the provided transcript and references.

Only add the following when they genuinely improve understanding — omit them if the prose is clearer on its own.

TABLES — use Markdown pipe tables (| col | col |) only when the content is naturally comparative or grid-structured:
- Multiple models/variants with numeric differences (parameter counts, sizes)
- Hyperparameter sets that would be harder to read as prose
- Layer-by-layer dimension breakdowns
REQUIRED: Every table MUST be preceded by exactly this line: [TABLE: Descriptive title here]
Never write a table without this marker — if you don't have a good title, write prose instead.

DIAGRAMS — use a fenced ```mermaid block only when a visual adds something prose cannot:
- A multi-step architecture or data flow that is hard to follow in words
- An algorithm loop or decision structure where order matters visually
Keep diagrams concise (≤ 12 nodes). Prefer `flowchart TD` for top-down flows, `graph LR` for relationships.
Always quote node labels that contain parentheses or special characters: `D["Transformer Blocks (12 layers)"]` not `D[Transformer Blocks (12 layers)]`.
Always add a caption on the line immediately before the mermaid block: [FIGURE: Descriptive title here]
Do not add a diagram just to have one — if the prose already explains it clearly, skip it."""


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

    result = {"name": group["name"], "slug": slug, "prose": prose}
    save_checkpoint("04_topics", slug, result, base_dir=base_dir)
    return result


def write_all_topics(
    groups: List[TopicGroup],
    transcripts: List[CorrectedTranscript],
    llm_client,
    batch_size: int = 1,
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
    for group in groups:
        results.append(write_topic(group, transcripts, overlaps_map, llm_client, base_dir))
        if progress:
            progress.advance("Stage 4: Write Topics")

    return results
