import re
from pathlib import Path
from typing import List, Dict, Any

from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


VERIFY_SYSTEM = """You are a technical book editor verifying prose against source material.
You will receive a section of book prose and the source transcripts/references it was written from.

Your job — in a SINGLE pass:
1. Read every factual/technical claim in the prose.
2. Check each claim against the provided sources.
3. For claims that are well-supported (score ≥ 0.8): keep as-is.
4. For claims that are partially supported (0.5–0.8): rewrite conservatively to match what the source actually says.
5. For claims with no source support (<0.5): remove them and close the gap smoothly.

Rules:
- Do NOT invent new facts.
- Do NOT remove supported claims.
- Preserve all citation markers [Video: "..." @ MM:SS] and [URL] exactly.

Respond in EXACTLY this format (no JSON, no fences):
PROSE:
<the full corrected prose text>
---STATS---
verified=<N> rewritten=<N> removed=<N>"""


def verify_topic(
    topic: Dict[str, Any],
    source_texts: List[str],
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> Dict[str, Any]:
    slug = topic["slug"]
    if checkpoint_exists("04b_verified", slug, base_dir=base_dir):
        return load_checkpoint("04b_verified", slug, base_dir=base_dir)

    # Truncate each source to keep total prompt manageable.
    sources_block = "\n\n---\n\n".join(s[:4000] for s in source_texts[:6])
    user = f"PROSE:\n{topic['prose']}\n\n===SOURCES===\n{sources_block}"

    raw = llm_client.complete(system=VERIFY_SYSTEM, user=user)

    # Parse PROSE: ... ---STATS--- format (avoids JSON escaping issues in prose)
    prose = topic["prose"]
    stats = {"verified": 0, "rewritten": 0, "removed": 0}
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

    output = {
        "name": topic["name"],
        "slug": slug,
        "prose": prose,
        "citations": [],
        "stats": stats,
    }
    save_checkpoint("04b_verified", slug, output, base_dir=base_dir)
    return output


def verify_all_topics(
    topics: List[Dict],
    groups_by_slug: Dict[str, Any],
    transcripts_by_vid: Dict[str, Any],
    batch_size: int = 1,
    base_dir: Path = Path("checkpoints"),
    progress=None,
    llm_client=None,
) -> List[Dict]:
    if progress:
        progress.add_stage("Stage 5b: Verify", total=len(topics))

    results = []
    for topic in topics:
        slug = topic["slug"]
        group = groups_by_slug.get(slug, {})
        source_texts = []
        for vid in group.get("video_ids", []):
            t = transcripts_by_vid.get(vid)
            if t:
                source_texts.append(t["full_text"])
        source_texts += list(group.get("ref_contents", {}).values())
        results.append(verify_topic(topic, source_texts, llm_client, base_dir=base_dir))
        if progress:
            progress.advance("Stage 5b: Verify")

    return results
