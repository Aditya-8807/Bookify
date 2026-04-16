import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


EXTRACT_SYSTEM = """Extract all factual and technical claims from this prose.
A claim is any sentence that asserts something that could be verified or disproved.
Return JSON: {"claims": ["claim 1", "claim 2", ...]}
Return an empty list if there are no checkable claims."""

SCORE_SYSTEM = """Given a claim and source passages, score how well the sources support the claim.
Score 0.0 (no support) to 1.0 (fully supported).
Return JSON: {"score": 0.0-1.0, "passage": "most relevant source passage, or empty string"}"""

REWRITE_SYSTEM = """Rewrite the following paragraph to be grounded only in the provided source material.
Remove or rephrase any claims not supported by the sources.
Maintain the educational narrative flow — do not leave gaps or broken sentences.
Return only the rewritten paragraph, no commentary."""


@dataclass
class VerificationStats:
    verified: int = 0
    rewritten: int = 0
    removed: int = 0


def extract_claims(prose: str, llm_client) -> List[str]:
    result = llm_client.complete_json(system=EXTRACT_SYSTEM, user=prose)
    return result.get("claims", [])


def score_claim(claim: str, source_texts: List[str], llm_client) -> Tuple[float, str]:
    sources = "\n---\n".join(source_texts[:5])
    user = f"Claim: {claim}\n\nSources:\n{sources}"
    result = llm_client.complete_json(system=SCORE_SYSTEM, user=user)
    return float(result.get("score", 0.0)), result.get("passage", "")


def _rewrite_paragraph(paragraph: str, sources: List[str], llm_client) -> str:
    sources_text = "\n---\n".join(sources[:5])
    user = f"Paragraph:\n{paragraph}\n\nSources:\n{sources_text}"
    return llm_client.complete(system=REWRITE_SYSTEM, user=user)


def verify_topic(
    topic: Dict[str, Any],
    source_texts: List[str],
    llm_client,
    max_retries: int = 2,
    base_dir: Path = Path("checkpoints"),
) -> Dict[str, Any]:
    slug = topic["slug"]
    if checkpoint_exists("04b_verified", slug, base_dir=base_dir):
        return load_checkpoint("04b_verified", slug, base_dir=base_dir)

    prose = topic["prose"]
    stats = VerificationStats()

    claims = extract_claims(prose, llm_client)
    for claim in claims:
        score, _ = score_claim(claim, source_texts, llm_client)
        if score >= 0.8:
            stats.verified += 1
        elif score >= 0.5:
            prose = prose.replace(claim, _rewrite_paragraph(claim, source_texts, llm_client))
            stats.rewritten += 1
        else:
            passed = False
            for _ in range(max_retries):
                rewritten_claim = _rewrite_paragraph(claim, source_texts, llm_client)
                retry_score, _ = score_claim(rewritten_claim, source_texts, llm_client)
                if retry_score >= 0.8:
                    prose = prose.replace(claim, rewritten_claim)
                    stats.rewritten += 1
                    passed = True
                    break
            if not passed:
                prose = prose.replace(claim, "")
                stats.removed += 1

    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()

    result = {
        "name": topic["name"],
        "slug": slug,
        "prose": prose,
        "citations": [],
        "stats": {
            "verified": stats.verified,
            "rewritten": stats.rewritten,
            "removed": stats.removed,
        },
    }
    save_checkpoint("04b_verified", slug, result, base_dir=base_dir)
    return result


def verify_all_topics(
    topics: List[Dict],
    groups_by_slug: Dict[str, Any],
    transcripts_by_vid: Dict[str, Any],
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    progress=None,
    llm_client=None,
) -> List[Dict]:
    if progress:
        progress.add_stage("Stage 4b: Verify Citations", total=len(topics))

    results = []

    def _verify(topic):
        slug = topic["slug"]
        group = groups_by_slug.get(slug, {})
        source_texts = []
        for vid in group.get("video_ids", []):
            t = transcripts_by_vid.get(vid)
            if t:
                source_texts.append(t["full_text"])
        source_texts += list(group.get("ref_contents", {}).values())
        return verify_topic(topic, source_texts, llm_client, base_dir=base_dir)

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(_verify, t): t for t in topics}
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 4b: Verify Citations")

    slug_order = {t["slug"]: i for i, t in enumerate(topics)}
    return sorted(results, key=lambda x: slug_order.get(x["slug"], 0))
