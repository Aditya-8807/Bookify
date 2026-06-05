from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from pipeline import CorrectedTranscript

_EMBED_MODEL = "all-MiniLM-L6-v2"
_TRANSCRIPT_COLLECTION = "transcripts"
_BATCH_SIZE = 500


def _get_ef() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)


def build_index(
    transcripts: List[CorrectedTranscript],
    persist_dir: str = "checkpoints/rag_index",
) -> chromadb.Collection:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(_TRANSCRIPT_COLLECTION, embedding_function=_get_ef())

    if col.count() > 0:
        return col

    docs, ids, metas = [], [], []
    for t in transcripts:
        for i, seg in enumerate(t["segments"]):
            docs.append(seg["text"])
            ids.append(f"{t['video_id']}_{i}")
            metas.append({
                "video_id": t["video_id"],
                "title": t["title"],
                "start": seg["start"],
                "source": "transcript",
            })

    for i in range(0, len(docs), _BATCH_SIZE):
        col.add(
            documents=docs[i:i + _BATCH_SIZE],
            ids=ids[i:i + _BATCH_SIZE],
            metadatas=metas[i:i + _BATCH_SIZE],
        )
    return col


def index_book_chapters(col: chromadb.Collection, chapters: List[Dict[str, Any]]) -> None:
    docs, ids, metas = [], [], []
    for ch in chapters:
        chunk_id = f"chapter__{ch['slug']}"
        if col.get(ids=[chunk_id])["ids"]:
            continue
        docs.append(ch["prose"][:2000])
        ids.append(chunk_id)
        metas.append({"slug": ch["slug"], "name": ch["name"], "source": "book"})
    if docs:
        col.add(documents=docs, ids=ids, metadatas=metas)


def query_chunks(
    col: chromadb.Collection,
    query: str,
    n_results: int = 8,
    source_filter: str = None,
) -> List[Dict[str, Any]]:
    where = {"source": source_filter} if source_filter else None
    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "video_id": meta.get("video_id", ""),
            "title": meta.get("title", meta.get("name", "")),
            "start": meta.get("start", 0.0),
            "source": meta.get("source", "transcript"),
        })
    return chunks
