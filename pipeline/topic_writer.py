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


WRITE_SYSTEM = """You are writing a chapter of a comprehensive technical book on building LLMs from scratch.

CRITICAL LENGTH REQUIREMENT: Write a minimum of 4000 words. Every concept, algorithm, and
implementation detail present in the transcripts must be covered fully and deeply.
Do NOT summarise — this chapter must replace the need to watch the source videos.
If you run out of obvious structure, go deeper: add worked examples, explore edge cases,
explain the mathematical intuition, show the code step by step.

STRUCTURE:
- A ## opening section (descriptive subtitle, NOT the chapter name) giving context and motivation
- A dedicated ### subsection for every major concept in the transcripts:
    * What it is and why it matters
    * How it works mechanically or mathematically, with concrete numbers where the transcript provides them
    * Python / PyTorch implementation details from the transcript — reproduce actual code snippets in fenced ```python blocks
    * Common pitfalls or design choices explained in the lectures
- A ## Summary section (1–2 paragraphs, key takeaways only)

HEADING RULES:
- Use ## for major sections, ### for subsections. NEVER use # (reserved for chapter title added externally).
- The very first line of your response must be a ## heading — a descriptive subtitle, not the chapter name.

STYLE:
- Clear, educational prose. No bullet-point summaries.
- Strip first-person instructor voice ("In this video I will…", "Today we look at…").
- Use consistent terminology. Cross-reference earlier chapters: "As introduced in <Topic Name>…"

CITATIONS:
- Transcript: [Video: "<exact title>" @ MM:SS]
- Reference URL: [<URL>]

TABLES — use only for genuinely comparative or grid-structured data:
- Model variants with numeric differences, hyperparameter sets, dimension breakdowns
- Preceding line MUST be exactly: [TABLE: Descriptive title here]

DIAGRAMS — use only when a visual adds something prose cannot:
- Architecture flows, data pipelines, algorithm loops (≤ 12 nodes)
- Prefer `flowchart TD` for top-down, `graph LR` for relationships
- Quote node labels containing parentheses: `D["Transformer Blocks (12 layers)"]`
- Preceding line MUST be exactly: [FIGURE: Descriptive title here]

Do NOT invent facts. Every claim must be grounded in the provided transcripts or references."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


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


def detect_overlaps(groups: List[TopicGroup], llm_client) -> List[Dict]:
    topics_summary = "\n".join(
        f"Topic: {g['name']} — videos: {g['video_ids']}" for g in groups
    )
    result = llm_client.complete_json(system=OVERLAP_SYSTEM, user=topics_summary)
    return result.get("overlaps", [])


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
