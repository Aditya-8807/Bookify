import re
from pathlib import Path
from typing import List

from pipeline import TopicGroup, SubTopic
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


SPLIT_SYSTEM = """You are structuring a chapter of a deep technical book.
Given a chapter topic and its source video IDs, identify distinct subtopics that fully cover the chapter's content.
Return JSON:
{
  "subtopics": [
    {
      "name": "Specific subtopic name",
      "description": "1-2 sentence description of exactly what this subtopic covers"
    }
  ]
}
Rules:
- Names must be specific (e.g. "Scaled Dot-Product Attention Formula", not "Attention").
- Together, all subtopics must cover the full scope of the chapter — no gaps.
- Do not overlap subtopics."""


def _slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    return re.sub(r"\s+", "-", s.strip())


def split_topic_into_subtopics(
    group: TopicGroup,
    llm_client,
    n_subtopics: int = 5,
    base_dir: Path = Path("checkpoints"),
) -> List[SubTopic]:
    ck_key = f"{group['slug']}_subtopics"
    if checkpoint_exists("03b_subtopics", ck_key, base_dir=base_dir):
        return load_checkpoint("03b_subtopics", ck_key, base_dir=base_dir)

    user = (
        f"Chapter: {group['name']}\n"
        f"Source videos: {', '.join(group['video_ids'])}\n"
        f"Target number of subtopics: {n_subtopics}\n"
    )
    result = llm_client.complete_json(system=SPLIT_SYSTEM, user=user)
    subtopics: List[SubTopic] = [
        {"name": s["name"], "slug": _slugify(s["name"]), "description": s["description"]}
        for s in result.get("subtopics", [])
    ]
    save_checkpoint("03b_subtopics", ck_key, subtopics, base_dir=base_dir)
    return subtopics


def split_all_topics(
    groups: List[TopicGroup],
    llm_client,
    n_subtopics: int = 5,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[TopicGroup]:
    if progress:
        progress.add_stage("Stage 3b: Split Subtopics", total=len(groups))
    expanded = []
    for group in groups:
        subtopics = split_topic_into_subtopics(group, llm_client, n_subtopics, base_dir=base_dir)
        expanded.append({**group, "subtopics": subtopics})
        if progress:
            progress.advance("Stage 3b: Split Subtopics")
    return expanded
