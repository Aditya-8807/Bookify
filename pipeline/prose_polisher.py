from pathlib import Path
from typing import List, Dict, Any

from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


POLISH_SYSTEM = """You are a technical book editor doing a final prose pass.
The text you receive has had some sentences removed or rewritten during a citation verification step.
Your job is to fix any damage that left behind:
- Broken or incomplete sentences
- Awkward transitions where content was removed
- Orphaned citation markers with no surrounding sentence
- Inconsistent formatting (double blank lines, missing spacing after headers)
- Sentences that start with a conjunction because the preceding sentence was cut

Rules:
- Do NOT add new facts, explanations, or content
- Do NOT remove any correctly-formed sentences
- Do NOT change citation markers like [Video: "..." @ MM:SS] or [URL]
- Only fix flow, transitions, and formatting
- Return only the corrected prose, no commentary"""


def polish_topic(
    topic: Dict[str, Any],
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> Dict[str, Any]:
    slug = topic["slug"]
    if checkpoint_exists("04c_polished", slug, base_dir=base_dir):
        return load_checkpoint("04c_polished", slug, base_dir=base_dir)

    polished_prose = llm_client.complete(system=POLISH_SYSTEM, user=topic["prose"])

    result = {
        "name": topic["name"],
        "slug": slug,
        "prose": polished_prose,
        "citations": topic.get("citations", []),
        "stats": topic.get("stats", {}),
    }
    save_checkpoint("04c_polished", slug, result, base_dir=base_dir)
    return result


def polish_all_topics(
    topics: List[Dict[str, Any]],
    llm_client,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[Dict[str, Any]]:
    if progress:
        progress.add_stage("Stage 5c: Polish", total=len(topics))

    results = []
    for topic in topics:
        results.append(polish_topic(topic, llm_client, base_dir))
        if progress:
            progress.advance("Stage 4c: Polish")

    return results
