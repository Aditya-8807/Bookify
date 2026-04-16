from pathlib import Path
from typing import List, Dict, Any

from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


INTRO_SYSTEM = """Write an introduction for a technical book about building LLMs from scratch.
Explain what the book covers, who it's for, how to use it, and what the reader will learn.
Write 3-4 paragraphs of clear, engaging prose. Do not use bullet points."""

CONCLUSION_SYSTEM = """Write a conclusion for a technical book about building LLMs from scratch.
Summarize the key journey the reader has taken, what they've learned, and what they might explore next.
Write 2-3 paragraphs. Do not use bullet points."""

GLOSSARY_SYSTEM = """Extract all technical terms that are explicitly defined or explained in this book text.
For each term, write a one-sentence definition based solely on how the book explains it.
Return JSON: {"terms": [{"term": "...", "definition": "..."}]}
Sort terms alphabetically."""


def generate_glossary(full_prose: str, llm_client) -> List[Dict[str, str]]:
    result = llm_client.complete_json(system=GLOSSARY_SYSTEM, user=full_prose[:20000])
    return sorted(result.get("terms", []), key=lambda x: x["term"].lower())


def assemble_book(
    verified_topics: List[Dict[str, Any]],
    groups: List[Dict],
    llm_client,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> str:
    if checkpoint_exists("05_book", "book", base_dir=base_dir):
        return load_checkpoint("05_book", "book", base_dir=base_dir)

    if progress:
        progress.add_stage("Stage 5: Assemble", total=1)

    topic_names = [t["name"] for t in verified_topics]
    intro = llm_client.complete(
        system=INTRO_SYSTEM,
        user=f"Topics covered in order: {topic_names}",
    )
    conclusion = llm_client.complete(
        system=CONCLUSION_SYSTEM,
        user=f"Topics covered: {topic_names}",
    )

    all_prose = "\n\n".join(t["prose"] for t in verified_topics)
    glossary_terms = generate_glossary(all_prose, llm_client)

    group_by_slug = {g["slug"]: g for g in groups}
    references_section = "## References & Resources\n\n"
    for topic in verified_topics:
        group = group_by_slug.get(topic["slug"], {})
        urls = group.get("ref_urls", [])
        if urls:
            references_section += f"### {topic['name']}\n"
            for url in urls:
                references_section += f"- {url}\n"
            references_section += "\n"

    glossary_section = "## Glossary\n\n"
    for entry in glossary_terms:
        glossary_section += f"**{entry['term']}** — {entry['definition']}\n\n"

    parts = ["## Introduction\n\n" + intro]
    for topic in verified_topics:
        parts.append(f"## {topic['name']}\n\n{topic['prose']}")
    parts.append("## Conclusion\n\n" + conclusion)
    parts.append(glossary_section)
    parts.append(references_section)

    full_book = "\n\n---\n\n".join(parts)
    save_checkpoint("05_book", "book", full_book, base_dir=base_dir)

    if progress:
        progress.advance("Stage 5: Assemble")
    return full_book
