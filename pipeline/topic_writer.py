import re
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


WRITE_SYSTEM = """You are writing a chapter of a deep technical book on building LLMs from scratch.

CRITICAL DEPTH REQUIREMENT:
- Target 2500+ words for this chapter.
- This is NOT a summary. Cover all major concepts, mechanisms, and implementation details from the provided sources.
- If content is dense, go step-by-step with concrete examples rather than compressing.

STYLE AND STRUCTURE:
- Educational prose, not bullet-point summaries.
- Strip first-person instructor voice ("In this video I will...").
- Use ## for major sections and ### for subsections.
- Never use # (h1); chapter title is injected by the assembler.
- Include an opening context section and a closing summary section.
- Maintain clear transitions and avoid abrupt topic jumps.

SOURCE DISCIPLINE:
- Do NOT invent facts.
- Every technical claim must be grounded in provided transcripts/references.
- Transcript citations: [Video: "<title>" @ MM:SS]
- Reference citations: [<URL>]
- Keep citations close to the specific claim they support.

FORMAT RULES:
- Avoid raw markdown artifacts in output. Do not output unmatched code fences.
- Use fenced ```python code blocks only when real code significantly improves clarity.
- Use [TABLE: ...] + markdown tables only for truly comparative data.
- Use [FIGURE: ...] + mermaid only when needed for non-trivial flows.
"""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def write_topic(
    group: TopicGroup,
    transcripts: List[CorrectedTranscript],
    overlaps_map: Dict[str, Dict],
    llm_client,
    min_words: int = 2500,
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
    source_word_budget = _word_count(transcript_text) + _word_count(refs_text)
    if source_word_budget >= min_words and _word_count(prose) < min_words:
        prose = llm_client.complete(
            system=WRITE_SYSTEM,
            user=(
                f"{user}\n\n"
                f"The previous draft was too short ({_word_count(prose)} words). "
                f"Rewrite this full chapter to at least {min_words} words with deeper technical detail, "
                f"more worked examples, and full coverage of all key source concepts."
            ),
        )

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
    min_words_per_topic: int = 2500,
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
        results.append(
            write_topic(
                group,
                transcripts,
                overlaps_map,
                llm_client,
                min_words=min_words_per_topic,
                base_dir=base_dir,
            )
        )
        if progress:
            progress.advance("Stage 4: Write Topics")

    return results
