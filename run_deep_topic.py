#!/usr/bin/env python3
"""
Deep single-topic book generator.

Targets ~12,000 words by splitting one topic into subtopics and writing
each at depth, then verifying, polishing, and rendering a standalone PDF.

Usage:
    python run_deep_topic.py
    python run_deep_topic.py --slug llm-architecture-components
    python run_deep_topic.py --slug llm-architecture-components --fresh
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from llm.client import client_from_config, get_cost_summary
from pipeline.pdf_renderer import markdown_to_html, render_pdf
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists

# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_SLUG = "llm-architecture-components"
DEEP_BASE    = Path("checkpoints/deep")
TARGET_WORDS = 12_000
WORDS_PER_SUBTOPIC = 2_000   # 6 subtopics × 2000 ≈ 12 000

# ── prompts ───────────────────────────────────────────────────────────────────

SPLIT_SYSTEM = """You are planning a deep technical book chapter.
Given a topic name and its source transcripts, identify 6 subtopics that together
cover the material exhaustively. Each subtopic should be independently writable
(has its own distinct concepts) but clearly part of the same chapter.

Return JSON only:
{
  "subtopics": [
    {"title": "...", "focus": "one-sentence description of what this subtopic covers"}
  ]
}
Exactly 6 subtopics. Order them so that foundational concepts come before advanced ones."""


WRITE_SYSTEM = """You are writing a subtopic section for a deep technical book on building LLMs from scratch.

DEPTH REQUIREMENT:
- Target {words} words for this section.
- This is NOT a summary. Go deep: cover mechanisms, intuitions, implementation details,
  worked examples, and edge cases.
- Use concrete examples (e.g. actual tensor shapes, pseudocode, step-by-step walkthroughs).

STYLE:
- Educational prose. Strip all first-person instructor voice ("In this video I...").
- Use ## for the section title and ### for subsections.
- Never use # (h1) — the assembler injects the top-level heading.
- Include a brief opening that motivates WHY this subtopic matters, and a short
  closing sentence that transitions to the next concept.

SOURCE DISCIPLINE:
- Ground every technical claim in the provided transcripts/references.
- Transcript citations: [Video: "<title>" @ MM:SS]
- Reference citations: [<URL>]
- AMBIGUOUS claims you cannot verify: mark (unverified) and flag conservatively.

FORMAT:
- Fenced ```python``` blocks only for real illustrative code.
- Markdown tables only for genuinely comparative data.
- No unmatched fences, no stray markdown artifacts."""


VERIFY_SYSTEM = """You are a technical book editor verifying a section against its source material.

Your job in ONE pass:
1. Read every factual/technical claim.
2. Check each against the provided sources.
3. score ≥ 0.8 (well-supported): keep as-is.
4. score 0.5–0.8 (partial): rewrite conservatively to match the source.
5. score < 0.5 (unsupported): remove and close the gap smoothly.

Rules:
- Do NOT add new facts.
- Preserve all citation markers [Video: "..." @ MM:SS] and [URL] exactly.

Respond in EXACTLY this format:
PROSE:
<full corrected prose>
---STATS---
verified=<N> rewritten=<N> removed=<N>"""


POLISH_SYSTEM = """You are a technical book editor doing a final prose pass on a single section.
Fix ONLY:
- Broken or incomplete sentences
- Awkward transitions where content was removed
- Orphaned citation markers with no surrounding sentence
- Inconsistent formatting (double blank lines, missing spacing after headers)

Rules:
- Do NOT add new facts or explanations.
- Do NOT change citation markers.
- Return only the corrected prose, no commentary."""


INTRO_SYSTEM = """Write an introduction for a deep technical chapter on "{topic}" in a book about
building LLMs from scratch.
Explain what the chapter covers, why it matters, and what the reader will be able to
understand after reading it. Write 3 paragraphs of clear, engaging prose. No bullet points."""


CONCLUSION_SYSTEM = """Write a conclusion for the chapter on "{topic}" in a technical book about
building LLMs from scratch.
Summarise the key concepts covered, reinforce the most important insights, and briefly
point to what the reader might explore next. Write 2 paragraphs. No bullet points."""


GLOSSARY_SYSTEM = """Extract all technical terms explicitly defined or explained in this text.
For each term write a one-sentence definition based solely on how the text explains it.
Return JSON: {{"terms": [{{"term": "...", "definition": "..."}}]}}
Sort alphabetically."""


# ── helpers ───────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _load_corrected_transcripts(video_ids: list, base_dir: Path = Path("checkpoints")) -> list:
    transcripts = []
    corrected_dir = base_dir / "02b_corrected"
    for vid in video_ids:
        path = corrected_dir / f"{vid}.json"
        if path.exists():
            transcripts.append(json.loads(path.read_text()))
    return transcripts


def _build_transcript_block(transcripts: list) -> str:
    parts = []
    for t in transcripts:
        segments = t.get("segments") or []
        body = "\n".join(f"[{_fmt_ts(s['start'])}] {s['text']}" for s in segments)
        parts.append(f"=== {t.get('title', t.get('video_id', ''))} ===\n{body}")
    return "\n\n".join(parts)


def _build_refs_block(ref_contents: dict) -> str:
    parts = []
    for url, content in ref_contents.items():
        parts.append(f"--- Reference: {url} ---\n{content[:4000]}")
    return "\n\n".join(parts)


def _parse_verify_response(raw: str, fallback_prose: str) -> tuple[str, dict]:
    stats = {"verified": 0, "rewritten": 0, "removed": 0}
    prose = fallback_prose
    if "PROSE:" in raw and "---STATS---" in raw:
        prose_part, stats_part = raw.split("---STATS---", 1)
        prose = prose_part.split("PROSE:", 1)[-1].strip()
        for key in ("verified", "rewritten", "removed"):
            m = re.search(rf"{key}=(\d+)", stats_part)
            if m:
                stats[key] = int(m.group(1))
    elif "PROSE:" in raw:
        prose = raw.split("PROSE:", 1)[-1].strip()
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()
    return prose, stats


# ── pipeline stages ───────────────────────────────────────────────────────────

def _ck_base(slug_prefix: str) -> Path:
    return DEEP_BASE / slug_prefix


def split_into_subtopics(group: dict, transcript_block: str, llm_client) -> list[dict]:
    """Ask the LLM to identify 6 focused subtopics from the source material."""
    base = _ck_base(group["slug"])
    if checkpoint_exists("subtopics", "data", base_dir=base):
        return load_checkpoint("subtopics", "data", base_dir=base)

    user = (
        f"Topic: {group['name']}\n\n"
        f"Source transcript excerpt (first 8000 chars):\n"
        f"{transcript_block[:8000]}"
    )
    result = llm_client.complete_json(system=SPLIT_SYSTEM, user=user)
    subtopics = result.get("subtopics", [])

    save_checkpoint("subtopics", "data", subtopics, base_dir=base)
    print(f"  Identified {len(subtopics)} subtopics")
    for i, s in enumerate(subtopics, 1):
        print(f"    {i}. {s['title']}")
    return subtopics


def write_subtopic(
    subtopic: dict,
    idx: int,
    transcript_block: str,
    refs_block: str,
    group_name: str,
    llm_client,
    slug_prefix: str,
) -> dict:
    """Write one subtopic section targeting WORDS_PER_SUBTOPIC words."""
    base = _ck_base(slug_prefix)
    key  = f"write_{idx:02d}"

    if checkpoint_exists("write", key, base_dir=base):
        return load_checkpoint("write", key, base_dir=base)

    system = WRITE_SYSTEM.format(words=WORDS_PER_SUBTOPIC)
    user = (
        f"Chapter: {group_name}\n"
        f"Subtopic {idx}: {subtopic['title']}\n"
        f"Focus: {subtopic['focus']}\n\n"
        f"TRANSCRIPTS:\n{transcript_block}\n\n"
        f"REFERENCES:\n{refs_block}"
    )

    prose = llm_client.complete(system=system, user=user)

    # Retry if too short
    if _word_count(prose) < WORDS_PER_SUBTOPIC * 0.8:
        prose = llm_client.complete(
            system=system,
            user=(
                f"{user}\n\n"
                f"The previous draft was only {_word_count(prose)} words. "
                f"Rewrite to at least {WORDS_PER_SUBTOPIC} words with deeper technical "
                f"detail, more worked examples, and full coverage of the source material."
            ),
        )

    result = {"title": subtopic["title"], "idx": idx, "prose": prose}
    save_checkpoint("write", key, result, base_dir=base)
    print(f"  [{idx}] {subtopic['title']} — {_word_count(prose):,} words")
    return result


def verify_subtopic(
    subtopic_result: dict,
    transcript_block: str,
    refs_block: str,
    llm_client,
    slug_prefix: str,
) -> dict:
    idx  = subtopic_result["idx"]
    base = _ck_base(slug_prefix)
    key  = f"verify_{idx:02d}"

    if checkpoint_exists("verify", key, base_dir=base):
        return load_checkpoint("verify", key, base_dir=base)

    sources_block = f"{transcript_block[:5000]}\n\n{refs_block[:3000]}"
    user = f"PROSE:\n{subtopic_result['prose']}\n\n===SOURCES===\n{sources_block}"
    raw = llm_client.complete(system=VERIFY_SYSTEM, user=user)
    prose, stats = _parse_verify_response(raw, subtopic_result["prose"])

    result = {**subtopic_result, "prose": prose, "stats": stats}
    save_checkpoint("verify", key, result, base_dir=base)
    print(f"  [{idx}] verified  verified={stats['verified']} rewritten={stats['rewritten']} removed={stats['removed']}")
    return result


def polish_subtopic(
    subtopic_result: dict,
    llm_client,
    slug_prefix: str,
) -> dict:
    idx  = subtopic_result["idx"]
    base = _ck_base(slug_prefix)
    key  = f"polish_{idx:02d}"

    if checkpoint_exists("polish", key, base_dir=base):
        return load_checkpoint("polish", key, base_dir=base)

    polished = llm_client.complete(system=POLISH_SYSTEM, user=subtopic_result["prose"])
    result = {**subtopic_result, "prose": polished}
    save_checkpoint("polish", key, result, base_dir=base)
    print(f"  [{idx}] polished — {_word_count(polished):,} words")
    return result


def assemble_deep_book(
    group: dict,
    polished_subtopics: list[dict],
    llm_client,
    slug_prefix: str,
) -> str:
    base = _ck_base(slug_prefix)
    if checkpoint_exists("assembled", "book", base_dir=base):
        return load_checkpoint("assembled", "book", base_dir=base)

    topic_name = group["name"]

    intro = llm_client.complete(
        system=INTRO_SYSTEM.format(topic=topic_name),
        user=f"Subtopics covered: {[s['title'] for s in polished_subtopics]}",
    )
    conclusion = llm_client.complete(
        system=CONCLUSION_SYSTEM.format(topic=topic_name),
        user=f"Subtopics covered: {[s['title'] for s in polished_subtopics]}",
    )

    all_prose = "\n\n".join(s["prose"] for s in polished_subtopics)
    glossary_raw = llm_client.complete_json(system=GLOSSARY_SYSTEM, user=all_prose[:20000])
    glossary_terms = sorted(glossary_raw.get("terms", []), key=lambda x: x["term"].lower())

    # Build references section from group ref_urls
    ref_urls = group.get("ref_urls", [])
    refs_section = "# References & Resources\n\n"
    for i, url in enumerate(ref_urls, 1):
        refs_section += f"{i}. [{url}]({url})\n"

    # Glossary section
    glossary_section = "# Glossary\n\n"
    for entry in glossary_terms:
        glossary_section += f"**{entry['term']}** — {entry['definition']}\n\n"

    # Assemble full markdown
    parts = [f"# {topic_name}\n\n## Introduction\n\n{intro}"]
    for s in polished_subtopics:
        parts.append(s["prose"])
    parts.append(f"## Conclusion\n\n{conclusion}")
    parts.append(glossary_section)
    parts.append(refs_section)

    full_md = "\n\n---\n\n".join(parts)
    save_checkpoint("assembled", "book", full_md, base_dir=base)
    return full_md


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a deep single-topic book (~12 000 words).")
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="Topic slug from checkpoints/03_groups/groups.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fresh", action="store_true", help="Delete existing deep checkpoints and start over")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    llm_client = client_from_config(config)

    # ── Load source data ──────────────────────────────────────────────────────
    groups = json.loads(Path("checkpoints/03_groups/groups.json").read_text())
    group  = next((g for g in groups if g["slug"] == args.slug), None)
    if group is None:
        slugs = [g["slug"] for g in groups]
        raise SystemExit(f"Slug '{args.slug}' not found. Available: {slugs}")

    slug_prefix = args.slug

    if args.fresh:
        target = DEEP_BASE / slug_prefix
        if target.exists():
            shutil.rmtree(target)
            print(f"Cleared {target}")

    DEEP_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Deep topic book: {group['name']} ({len(group['video_ids'])} videos) ===")
    print(f"    Target: {TARGET_WORDS:,} words  |  {WORDS_PER_SUBTOPIC:,} words/subtopic\n")

    transcripts    = _load_corrected_transcripts(group["video_ids"])
    transcript_blk = _build_transcript_block(transcripts)
    refs_blk       = _build_refs_block(group.get("ref_contents", {}))

    print(f"Loaded {len(transcripts)} transcripts, {_word_count(transcript_blk):,} source words\n")

    # ── Stage 1: Split into subtopics ─────────────────────────────────────────
    print("Stage 1: Identifying subtopics...")
    subtopics = split_into_subtopics(group, transcript_blk, llm_client)

    # ── Stage 2: Write each subtopic ─────────────────────────────────────────
    print("\nStage 2: Writing subtopics...")
    written = [
        write_subtopic(st, i + 1, transcript_blk, refs_blk, group["name"], llm_client, slug_prefix)
        for i, st in enumerate(subtopics)
    ]
    total_written = sum(_word_count(s["prose"]) for s in written)
    print(f"  Total after writing: {total_written:,} words")

    # ── Stage 3: Verify citations ─────────────────────────────────────────────
    print("\nStage 3: Verifying citations...")
    verified = [
        verify_subtopic(w, transcript_blk, refs_blk, llm_client, slug_prefix)
        for w in written
    ]

    # ── Stage 4: Polish prose ─────────────────────────────────────────────────
    print("\nStage 4: Polishing prose...")
    polished = [
        polish_subtopic(v, llm_client, slug_prefix)
        for v in verified
    ]
    total_polished = sum(_word_count(s["prose"]) for s in polished)
    print(f"  Total after polishing: {total_polished:,} words")

    # ── Stage 5: Assemble ─────────────────────────────────────────────────────
    print("\nStage 5: Assembling book...")
    full_md = assemble_deep_book(group, polished, llm_client, slug_prefix)
    print(f"  Assembled: {_word_count(full_md):,} total words")

    # ── Stage 6: Render PDF ───────────────────────────────────────────────────
    print("\nStage 6: Rendering PDF...")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / f"deep_{slug_prefix}.pdf"

    html = markdown_to_html(full_md, title=group["name"])
    render_pdf(html, str(out_pdf))
    print(f"\n  PDF written: {out_pdf}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Topic   : {group['name']}")
    print(f"  Videos  : {len(group['video_ids'])}")
    print(f"  Words   : {total_polished:,}")
    print(f"  Output  : {out_pdf}")
    print(f"  {get_cost_summary()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
