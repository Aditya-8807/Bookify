# Pipeline Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix book depth (43-video playlist → 400-600 pages), cut token cost 90% via RAG-during-writing, and add auto-detect multilingual support (Hinglish/Hindi/etc → English output).

**Architecture:** Add three pre-writing stages: transcript deduplication (no API cost), language detection (no API cost), and sub-topic splitting (cheap LLM call per chapter). Replace full-transcript dumps in the writer with ChromaDB semantic retrieval (top-K chunks per sub-topic). Each sub-topic is written and checkpointed independently so a budget-exceeded or crash never loses completed work.

**Tech Stack:** `langdetect`, `chromadb>=0.5`, `sentence-transformers` (all-MiniLM-L6-v2), existing `LLMClient`, existing checkpoint system.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `pipeline/transcript_dedup.py` | Jaccard deduplication of transcript segments |
| Create | `pipeline/subtopic_splitter.py` | LLM splits each TopicGroup into 4-6 SubTopics |
| Create | `utils/language_detect.py` | langdetect + Hinglish heuristic, returns prompt instruction |
| Create | `utils/rag_index.py` | Build/query ChromaDB index over transcript segments |
| Create | `tests/test_transcript_dedup.py` | |
| Create | `tests/test_subtopic_splitter.py` | |
| Create | `tests/test_language_detect.py` | |
| Create | `tests/test_rag_index.py` | |
| Modify | `pipeline/__init__.py` | Add `SubTopic` TypedDict; add `subtopics` field to `TopicGroup` |
| Modify | `llm/client.py` | Add `api_key` param, `TokenBudgetExceeded`, `get_spend_usd()` |
| Modify | `pipeline/topic_writer.py` | Add `write_subtopic()`, `write_topic_with_subtopics()`, `coverage_pass()` |
| Modify | `run.py` | Add stages 2b (dedup), 3b (subtopic split), 3c (RAG index); update stage 5 |
| Modify | `config.yaml` | Add `token_budget_usd`, `subtopics_per_topic`, `coverage_target` |
| Modify | `requirements.txt` | Add `langdetect`, `chromadb`, `sentence-transformers` |

---

## Task 1: Add dependencies and SubTopic type

**Files:**
- Modify: `requirements.txt`
- Modify: `pipeline/__init__.py`

- [ ] **Step 1: Add new dependencies to requirements.txt**

Append these three lines to `requirements.txt`:
```
langdetect>=1.0.9
chromadb>=0.5.0
sentence-transformers>=3.0.0
```

- [ ] **Step 2: Add SubTopic TypedDict and update TopicGroup in pipeline/__init__.py**

Add after the `TopicGroup` class definition:

```python
class SubTopic(TypedDict):
    name: str
    slug: str
    description: str
```

Update `TopicGroup` to include an optional `subtopics` field:

```python
class TopicGroup(TypedDict):
    name: str
    slug: str
    video_ids: List[str]
    dependency_order: int
    prerequisites: List[str]
    ref_urls: List[str]
    ref_contents: Dict[str, str]
    subtopics: List["SubTopic"]  # populated by subtopic_splitter; empty list if not yet split
```

- [ ] **Step 3: Install dependencies**

Run: `pip install langdetect chromadb sentence-transformers`
Expected: all three packages install without error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pipeline/__init__.py
git commit -m "feat: add SubTopic type and pipeline dependencies (langdetect, chromadb, sentence-transformers)"
```

---

## Task 2: Transcript deduplication

**Files:**
- Create: `pipeline/transcript_dedup.py`
- Create: `tests/test_transcript_dedup.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transcript_dedup.py`:

```python
from pipeline.transcript_dedup import dedup_segments, dedup_all


def _make_transcript(segs):
    return {"video_id": "v1", "title": "T", "full_text": "", "segments": segs, "corrections": []}


def test_dedup_removes_near_duplicate_segment():
    segs = [
        {"start": 0.0, "end": 3.0, "text": "attention is all you need the transformer architecture"},
        {"start": 5.0, "end": 8.0, "text": "attention is all you need the transformer architecture"},
        {"start": 10.0, "end": 13.0, "text": "backpropagation computes gradients via chain rule"},
    ]
    result = dedup_segments(_make_transcript(segs))
    assert len(result["segments"]) == 2


def test_dedup_keeps_similar_but_distinct_segments():
    segs = [
        {"start": 0.0, "end": 3.0, "text": "attention mechanism in transformers uses query key value"},
        {"start": 5.0, "end": 8.0, "text": "attention mechanism also applies in vision transformers"},
    ]
    result = dedup_segments(_make_transcript(segs), threshold=0.9)
    assert len(result["segments"]) == 2


def test_dedup_preserves_transcript_fields():
    segs = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    t = _make_transcript(segs)
    t["corrections"] = [{"old": "helo", "new": "hello"}]
    result = dedup_segments(t)
    assert result["video_id"] == "v1"
    assert result["corrections"] == [{"old": "helo", "new": "hello"}]


def test_dedup_all_processes_list():
    t1 = _make_transcript([
        {"start": 0.0, "end": 2.0, "text": "same text same text same text same"},
        {"start": 3.0, "end": 5.0, "text": "same text same text same text same"},
    ])
    t2 = _make_transcript([
        {"start": 0.0, "end": 2.0, "text": "different content about neural networks"},
    ])
    results = dedup_all([t1, t2])
    assert len(results) == 2
    assert len(results[0]["segments"]) == 1
    assert len(results[1]["segments"]) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_transcript_dedup.py -v`
Expected: `ModuleNotFoundError: No module named 'pipeline.transcript_dedup'`

- [ ] **Step 3: Implement transcript_dedup.py**

Create `pipeline/transcript_dedup.py`:

```python
import re
from typing import List
from pipeline import CorrectedTranscript


def _tokenize(text: str) -> set:
    return set(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup_segments(transcript: CorrectedTranscript, threshold: float = 0.85) -> CorrectedTranscript:
    seen: List[set] = []
    kept = []
    for seg in transcript["segments"]:
        words = _tokenize(seg["text"])
        if any(_jaccard(words, s) >= threshold for s in seen):
            continue
        seen.append(words)
        kept.append(seg)
    return {**transcript, "segments": kept}


def dedup_all(transcripts: List[CorrectedTranscript], threshold: float = 0.85) -> List[CorrectedTranscript]:
    return [dedup_segments(t, threshold) for t in transcripts]
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_transcript_dedup.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/transcript_dedup.py tests/test_transcript_dedup.py
git commit -m "feat: add transcript segment deduplication (Jaccard similarity)"
```

---

## Task 3: Language detection

**Files:**
- Create: `utils/language_detect.py`
- Create: `tests/test_language_detect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_language_detect.py`:

```python
from utils.language_detect import detect_language, language_instruction


def test_detects_hinglish_by_marker_count():
    text = "aur yaar matlab kya hai toh bhi nahi samjha yeh concept"
    assert detect_language(text) == "hinglish"


def test_detects_english():
    text = "attention mechanism uses query key and value matrices to compute weighted sums"
    lang = detect_language(text)
    assert lang == "en"


def test_language_instruction_hindi_mentions_english_output():
    instr = language_instruction("hi")
    assert "English" in instr
    assert "Hindi" in instr


def test_language_instruction_hinglish_mentions_english_output():
    instr = language_instruction("hinglish")
    assert "English" in instr
    assert "Hinglish" in instr


def test_language_instruction_english_is_empty():
    assert language_instruction("en") == ""


def test_language_instruction_other_language_mentions_english():
    instr = language_instruction("fr")
    assert "English" in instr
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_language_detect.py -v`
Expected: `ModuleNotFoundError: No module named 'utils.language_detect'`

- [ ] **Step 3: Implement language_detect.py**

Create `utils/language_detect.py`:

```python
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # deterministic results

_HINGLISH_MARKERS = ["kya", "hai", "aur", "nahi", "bhi", "toh", "matlab", "yaar", "woh", "iska"]
_HINGLISH_THRESHOLD = 3


def detect_language(text: str) -> str:
    sample = text[:3000].lower()
    hits = sum(1 for marker in _HINGLISH_MARKERS if f" {marker} " in sample)
    if hits >= _HINGLISH_THRESHOLD:
        return "hinglish"
    try:
        return detect(sample)
    except Exception:
        return "en"


def language_instruction(lang_code: str) -> str:
    if lang_code == "hinglish":
        return (
            "The source material is in Hinglish (Hindi-English code-switching). "
            "Understand the content fully and write all output in fluent English."
        )
    if lang_code == "hi":
        return (
            "The source material is in Hindi. "
            "Understand the content fully and write all output in fluent English."
        )
    if lang_code != "en":
        return (
            f"The source material may be in language code '{lang_code}'. "
            "Write all output in fluent English."
        )
    return ""
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_language_detect.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/language_detect.py tests/test_language_detect.py
git commit -m "feat: add language detection with Hinglish heuristic"
```

---

## Task 4: LLMClient API key injection and token budget

**Files:**
- Modify: `llm/client.py`
- Modify: `tests/test_llm_client.py`

- [ ] **Step 1: Read existing test file**

Read `tests/test_llm_client.py` to understand existing test patterns before adding new ones.

- [ ] **Step 2: Write failing tests for new LLMClient features**

Append to `tests/test_llm_client.py`:

```python
from llm.client import TokenBudgetExceeded


def test_get_spend_usd_returns_float(mock_llm):
    # mock_llm is the existing fixture from conftest
    assert isinstance(mock_llm.get_spend_usd(), float)


def test_token_budget_exceeded_raised_when_over_budget(monkeypatch):
    import llm.client as lc
    lc._total_input_tokens = 0
    lc._total_output_tokens = 0

    client = lc.LLMClient(
        provider="gemini",
        model="gemini-flash-latest",
        temperature=0.3,
        token_budget_usd=0.000001,  # effectively zero
    )
    # Simulate having already spent over budget
    lc._total_input_tokens = 1_000_000
    lc._total_output_tokens = 1_000_000

    with pytest.raises(TokenBudgetExceeded):
        client.complete(system="s", user="u")


def test_client_from_config_accepts_api_key():
    config = {
        "llm": {"provider": "gemini", "model": "gemini-flash-latest", "temperature": 0.3},
        "pipeline": {"rate_limit_rpm": 0, "token_budget_usd": 5.0},
    }
    client = client_from_config(config, api_key="test-key-123")
    assert client._api_key == "test-key-123"
    assert client._token_budget_usd == 5.0
```

- [ ] **Step 3: Run failing tests**

Run: `pytest tests/test_llm_client.py -v -k "spend or budget or api_key"`
Expected: FAIL — `TokenBudgetExceeded` not defined, `get_spend_usd` not found.

- [ ] **Step 4: Add TokenBudgetExceeded and budget tracking to llm/client.py**

At the top of `llm/client.py`, after the imports, add:

```python
class TokenBudgetExceeded(Exception):
    pass
```

Update `LLMClient.__init__` signature:

```python
def __init__(self, provider: str, model: str, temperature: float,
             rate_limit_rpm: int = 0, api_key: str = None,
             token_budget_usd: float = None):
    global _active_provider, _min_request_interval
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    self.provider = provider
    self.model = model
    self.temperature = temperature
    self._api_key = api_key
    self._token_budget_usd = token_budget_usd
    _active_provider = provider
    _min_request_interval = (60.0 / rate_limit_rpm) if rate_limit_rpm > 0 else 0.0
    self._client = self._init_client()
```

Update `_init_client` to use `self._api_key`:

```python
def _init_client(self):
    if self.provider == "openai":
        return OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
    elif self.provider == "anthropic":
        return Anthropic(api_key=self._api_key) if self._api_key else Anthropic()
    else:  # gemini
        key = self._api_key or os.environ["GEMINI_API_KEY"]
        return genai.Client(api_key=key)
```

Add `get_spend_usd()` and `_check_budget()` methods to `LLMClient`:

```python
def get_spend_usd(self) -> float:
    pricing = _PRICING.get(self.provider, {"input": 0.0, "output": 0.0})
    return (
        _total_input_tokens / 1_000_000 * pricing["input"] +
        _total_output_tokens / 1_000_000 * pricing["output"]
    )

def _check_budget(self) -> None:
    if self._token_budget_usd and self.get_spend_usd() >= self._token_budget_usd:
        raise TokenBudgetExceeded(
            f"Token budget ${self._token_budget_usd:.2f} exceeded "
            f"(spent ${self.get_spend_usd():.4f})"
        )
```

Call `self._check_budget()` as the first line of `complete()`:

```python
def complete(self, system: str, user: str) -> str:
    self._check_budget()
    last_exc = None
    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        ...
```

Update `client_from_config` to accept and pass `api_key`:

```python
def client_from_config(config: dict, api_key: str = None) -> LLMClient:
    llm = config["llm"]
    return LLMClient(
        provider=llm["provider"],
        model=llm["model"],
        temperature=llm["temperature"],
        rate_limit_rpm=config.get("pipeline", {}).get("rate_limit_rpm", 0),
        api_key=api_key,
        token_budget_usd=config.get("pipeline", {}).get("token_budget_usd"),
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_llm_client.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add llm/client.py tests/test_llm_client.py
git commit -m "feat: add api_key injection, TokenBudgetExceeded, and get_spend_usd to LLMClient"
```

---

## Task 5: Sub-topic splitter

**Files:**
- Create: `pipeline/subtopic_splitter.py`
- Create: `tests/test_subtopic_splitter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_subtopic_splitter.py`:

```python
from unittest.mock import MagicMock
from pipeline.subtopic_splitter import split_topic_into_subtopics, split_all_topics
from pathlib import Path
import tempfile


def _make_group(name="Attention Mechanism", slug="attention-mechanism"):
    return {
        "name": name, "slug": slug, "video_ids": ["v1", "v2"],
        "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {}, "subtopics": [],
    }


def test_split_topic_returns_list_of_subtopics(tmp_path):
    llm = MagicMock()
    llm.complete_json.return_value = {
        "subtopics": [
            {"name": "Scaled Dot-Product Attention", "description": "The core formula Q K^T / sqrt(d_k) V"},
            {"name": "Multi-Head Attention", "description": "Running attention in parallel across heads"},
            {"name": "Positional Encoding", "description": "Sine/cosine embeddings added to input"},
        ]
    }
    result = split_topic_into_subtopics(_make_group(), llm, n_subtopics=3, base_dir=tmp_path)
    assert len(result) == 3
    assert result[0]["name"] == "Scaled Dot-Product Attention"
    assert "slug" in result[0]
    assert "description" in result[0]


def test_split_topic_is_checkpointed(tmp_path):
    llm = MagicMock()
    llm.complete_json.return_value = {
        "subtopics": [{"name": "Sub A", "description": "desc A"}]
    }
    split_topic_into_subtopics(_make_group(), llm, base_dir=tmp_path)
    split_topic_into_subtopics(_make_group(), llm, base_dir=tmp_path)
    assert llm.complete_json.call_count == 1  # second call hits checkpoint


def test_split_all_topics_returns_groups_with_subtopics(tmp_path):
    llm = MagicMock()
    llm.complete_json.return_value = {
        "subtopics": [{"name": "Sub", "description": "desc"}]
    }
    groups = [_make_group("Topic A", "topic-a"), _make_group("Topic B", "topic-b")]
    result = split_all_topics(groups, llm, n_subtopics=1, base_dir=tmp_path)
    assert len(result) == 2
    assert len(result[0]["subtopics"]) == 1
    assert len(result[1]["subtopics"]) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_subtopic_splitter.py -v`
Expected: `ModuleNotFoundError: No module named 'pipeline.subtopic_splitter'`

- [ ] **Step 3: Implement subtopic_splitter.py**

Create `pipeline/subtopic_splitter.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_subtopic_splitter.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/subtopic_splitter.py tests/test_subtopic_splitter.py
git commit -m "feat: add sub-topic splitter (LLM splits each chapter into 4-6 subtopics)"
```

---

## Task 6: RAG index (ChromaDB + sentence-transformers)

**Files:**
- Create: `utils/rag_index.py`
- Create: `tests/test_rag_index.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_index.py`:

```python
import pytest
from utils.rag_index import build_index, query_chunks, index_book_chapters


def _make_transcripts():
    return [
        {
            "video_id": "v1", "title": "Attention Explained", "full_text": "",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "the attention mechanism computes a weighted sum of values"},
                {"start": 5.0, "end": 10.0, "text": "query key and value matrices are learned during training"},
            ],
            "corrections": [],
        },
        {
            "video_id": "v2", "title": "Backprop Basics", "full_text": "",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "backpropagation computes gradients using the chain rule"},
            ],
            "corrections": [],
        },
    ]


def test_build_index_creates_collection(tmp_path):
    col = build_index(_make_transcripts(), persist_dir=str(tmp_path / "idx"))
    assert col.count() == 3


def test_build_index_is_idempotent(tmp_path):
    persist = str(tmp_path / "idx")
    col1 = build_index(_make_transcripts(), persist_dir=persist)
    col2 = build_index(_make_transcripts(), persist_dir=persist)
    assert col2.count() == 3  # no duplicate insertions


def test_query_returns_relevant_chunks(tmp_path):
    col = build_index(_make_transcripts(), persist_dir=str(tmp_path / "idx"))
    chunks = query_chunks(col, "attention query key value", n_results=2)
    assert len(chunks) == 2
    texts = [c["text"] for c in chunks]
    assert any("attention" in t or "query" in t for t in texts)


def test_query_chunk_has_required_fields(tmp_path):
    col = build_index(_make_transcripts(), persist_dir=str(tmp_path / "idx"))
    chunks = query_chunks(col, "backpropagation gradients", n_results=1)
    assert "text" in chunks[0]
    assert "video_id" in chunks[0]
    assert "title" in chunks[0]
    assert "start" in chunks[0]


def test_index_book_chapters_adds_to_collection(tmp_path):
    persist = str(tmp_path / "idx")
    col = build_index(_make_transcripts(), persist_dir=persist)
    chapters = [{"name": "Ch1", "slug": "ch1", "prose": "attention is about weighting values by relevance"}]
    index_book_chapters(col, chapters)
    assert col.count() == 4  # 3 transcript + 1 chapter chunk
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_rag_index.py -v`
Expected: `ModuleNotFoundError: No module named 'utils.rag_index'`

- [ ] **Step 3: Implement rag_index.py**

Create `utils/rag_index.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_rag_index.py -v`
Expected: all 5 tests PASS. (Note: first run downloads the MiniLM model ~90MB — subsequent runs use cache.)

- [ ] **Step 5: Commit**

```bash
git add utils/rag_index.py tests/test_rag_index.py
git commit -m "feat: add ChromaDB RAG index with sentence-transformer embeddings"
```

---

## Task 7: Topic writer — RAG-during-writing and per-subtopic checkpoints

**Files:**
- Modify: `pipeline/topic_writer.py`
- Modify: `tests/test_topic_writer.py`

- [ ] **Step 1: Read existing test_topic_writer.py**

Read `tests/test_topic_writer.py` to understand existing fixtures before adding new tests.

- [ ] **Step 2: Write failing tests for new writer functions**

Append to `tests/test_topic_writer.py`:

```python
from unittest.mock import MagicMock, patch
from pipeline.topic_writer import write_subtopic, write_topic_with_subtopics


def _make_group_with_subtopics():
    return {
        "name": "Attention Mechanism", "slug": "attention-mechanism",
        "video_ids": ["v1"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
        "subtopics": [
            {"name": "Scaled Dot-Product", "slug": "scaled-dot-product", "description": "The core formula"},
            {"name": "Multi-Head Attention", "slug": "multi-head-attention", "description": "Parallel heads"},
        ],
    }


def test_write_subtopic_uses_rag_chunks(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "This is a detailed explanation of scaled dot product attention " * 100
    rag_col = MagicMock()
    rag_col.query.return_value = {
        "documents": [["attention uses query key value", "softmax normalizes weights"]],
        "metadatas": [[
            {"video_id": "v1", "title": "T1", "start": 0.0},
            {"video_id": "v1", "title": "T1", "start": 5.0},
        ]],
    }
    group = _make_group_with_subtopics()
    subtopic = group["subtopics"][0]
    result = write_subtopic(group, subtopic, rag_col, llm, base_dir=tmp_path)
    assert isinstance(result, str)
    assert len(result) > 0
    rag_col.query.assert_called_once()


def test_write_subtopic_checkpoints_result(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "content " * 500
    rag_col = MagicMock()
    rag_col.query.return_value = {
        "documents": [["chunk text"]],
        "metadatas": [[{"video_id": "v1", "title": "T", "start": 0.0}]],
    }
    group = _make_group_with_subtopics()
    subtopic = group["subtopics"][0]
    write_subtopic(group, subtopic, rag_col, llm, base_dir=tmp_path)
    write_subtopic(group, subtopic, rag_col, llm, base_dir=tmp_path)
    assert llm.complete.call_count == 1  # second call hits checkpoint


def test_write_topic_with_subtopics_assembles_sections(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "detailed section content " * 200
    rag_col = MagicMock()
    rag_col.query.return_value = {
        "documents": [["relevant chunk"]],
        "metadatas": [[{"video_id": "v1", "title": "T", "start": 0.0}]],
    }
    group = _make_group_with_subtopics()
    result = write_topic_with_subtopics(group, rag_col, llm, base_dir=tmp_path)
    assert result["name"] == "Attention Mechanism"
    assert "Scaled Dot-Product" in result["prose"]
    assert "Multi-Head Attention" in result["prose"]
```

- [ ] **Step 3: Run tests to confirm they fail**

Run: `pytest tests/test_topic_writer.py -v -k "write_subtopic or write_topic_with_subtopics"`
Expected: FAIL — functions not defined.

- [ ] **Step 4: Add write_subtopic and write_topic_with_subtopics to topic_writer.py**

Add these functions to `pipeline/topic_writer.py` (keep all existing functions intact):

```python
from utils.rag_index import query_chunks  # add this import at top


def write_subtopic(
    group: TopicGroup,
    subtopic,
    rag_col,
    llm_client,
    lang_instruction: str = "",
    min_words: int = 2000,
    base_dir: Path = Path("checkpoints"),
) -> str:
    slug = f"{group['slug']}__{subtopic['slug']}"
    if checkpoint_exists("04_subtopics", slug, base_dir=base_dir):
        return load_checkpoint("04_subtopics", slug, base_dir=base_dir)

    chunks = query_chunks(rag_col, f"{group['name']} {subtopic['name']} {subtopic['description']}", n_results=10)
    transcript_text = "\n".join(
        f"[{_fmt_ts(c['start'])}] {c['text']}  [Video: \"{c['title']}\"]"
        for c in chunks
    )

    system = (lang_instruction + "\n\n" + WRITE_SYSTEM).strip() if lang_instruction else WRITE_SYSTEM
    user = (
        f"Chapter: {group['name']}\n"
        f"Section: {subtopic['name']}\n"
        f"Scope: {subtopic['description']}\n\n"
        f"RELEVANT TRANSCRIPT SEGMENTS:\n{transcript_text}"
    )

    prose = llm_client.complete(system=system, user=user)
    if _word_count(prose) < min_words:
        prose = llm_client.complete(
            system=system,
            user=user + f"\n\nExpand to at least {min_words} words with deeper technical detail and worked examples.",
        )

    save_checkpoint("04_subtopics", slug, prose, base_dir=base_dir)
    return prose


def write_topic_with_subtopics(
    group: TopicGroup,
    rag_col,
    llm_client,
    lang_instruction: str = "",
    min_words_per_subtopic: int = 2000,
    base_dir: Path = Path("checkpoints"),
) -> Dict[str, Any]:
    slug = group["slug"]
    if checkpoint_exists("04_topics", slug, base_dir=base_dir):
        return load_checkpoint("04_topics", slug, base_dir=base_dir)

    sections = []
    for subtopic in group.get("subtopics", []):
        prose = write_subtopic(group, subtopic, rag_col, llm_client, lang_instruction, min_words_per_subtopic, base_dir)
        sections.append(f"### {subtopic['name']}\n\n{prose}")

    full_prose = "\n\n".join(sections)
    result = {"name": group["name"], "slug": slug, "prose": full_prose}
    save_checkpoint("04_topics", slug, result, base_dir=base_dir)
    return result
```

- [ ] **Step 5: Run tests to confirm they pass**

Run: `pytest tests/test_topic_writer.py -v`
Expected: all tests PASS (including pre-existing ones).

- [ ] **Step 6: Commit**

```bash
git add pipeline/topic_writer.py tests/test_topic_writer.py
git commit -m "feat: add RAG-during-writing with per-subtopic checkpoints to topic writer"
```

---

## Task 8: Coverage pass

**Files:**
- Modify: `pipeline/topic_writer.py`
- Modify: `tests/test_topic_writer.py`

- [ ] **Step 1: Write failing tests for coverage_pass**

Append to `tests/test_topic_writer.py`:

```python
from pipeline.topic_writer import coverage_pass


def _make_corrected_transcript(segs):
    return {
        "video_id": "v1", "title": "T", "full_text": "",
        "segments": [{"start": float(i), "end": float(i+1), "text": s} for i, s in enumerate(segs)],
        "corrections": [],
    }


def test_coverage_pass_appends_when_uncovered_content_exists(tmp_path):
    llm = MagicMock()
    llm.complete.return_value = "Additional content about completely new material here."
    rag_col = MagicMock()
    rag_col.query.return_value = {
        "documents": [["uncovered topic segment"]],
        "metadatas": [[{"video_id": "v1", "title": "T", "start": 0.0}]],
    }
    group = {
        "name": "Topic", "slug": "topic", "video_ids": ["v1"],
        "subtopics": [], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [_make_corrected_transcript(["totally different content xyz abc def"] * 20)]
    prose = "This is the written chapter content about attention."
    result = coverage_pass(group, transcripts, prose, rag_col, llm, target_coverage=0.99, base_dir=tmp_path)
    assert len(result) > len(prose)


def test_coverage_pass_skips_when_coverage_sufficient(tmp_path):
    llm = MagicMock()
    rag_col = MagicMock()
    group = {
        "name": "Topic", "slug": "topic", "video_ids": ["v1"],
        "subtopics": [], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [_make_corrected_transcript(["attention is all you need"] * 5)]
    prose = "attention is all you need transformer architecture"
    result = coverage_pass(group, transcripts, prose, rag_col, llm, target_coverage=0.5, base_dir=tmp_path)
    llm.complete.assert_not_called()
    assert result == prose
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_topic_writer.py -v -k "coverage_pass"`
Expected: FAIL — `coverage_pass` not defined.

- [ ] **Step 3: Add coverage_pass to topic_writer.py**

Append to `pipeline/topic_writer.py`:

```python
def coverage_pass(
    group: TopicGroup,
    transcripts: List[CorrectedTranscript],
    written_prose: str,
    rag_col,
    llm_client,
    target_coverage: float = 0.85,
    base_dir: Path = Path("checkpoints"),
) -> str:
    trans_by_id = {t["video_id"]: t for t in transcripts}
    group_segs = [
        seg["text"]
        for vid in group["video_ids"]
        for seg in trans_by_id.get(vid, {}).get("segments", [])
    ]
    if not group_segs:
        return written_prose

    prose_lower = written_prose.lower()
    covered = sum(
        1 for seg in group_segs
        if any(word in prose_lower for word in seg.lower().split()[:4] if len(word) > 4)
    )
    coverage = covered / len(group_segs)

    if coverage >= target_coverage:
        return written_prose

    uncovered = [
        seg for seg in group_segs
        if not any(word in prose_lower for word in seg.lower().split()[:4] if len(word) > 4)
    ][:15]

    addendum = llm_client.complete(
        system=WRITE_SYSTEM,
        user=(
            f"The following content from the source material was not covered in the chapter "
            f"on '{group['name']}'. Write 1-2 focused subsections to cover this material:\n\n"
            + "\n".join(f"- {s}" for s in uncovered)
        ),
    )
    return written_prose + "\n\n" + addendum
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_topic_writer.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/topic_writer.py tests/test_topic_writer.py
git commit -m "feat: add coverage pass to detect and fill uncovered transcript content"
```

---

## Task 9: Wire everything in run.py and update config.yaml

**Files:**
- Modify: `run.py`
- Modify: `config.yaml`

- [ ] **Step 1: Update config.yaml**

Replace the `pipeline:` section in `config.yaml` with:

```yaml
pipeline:
  batch_size: 4
  rate_limit_rpm: 6
  min_words_per_topic: 2500        # per subtopic (was per whole topic)
  subtopics_per_topic: 5           # how many subtopics to split each chapter into
  coverage_target: 0.85            # stop coverage pass when this fraction of segments are covered
  token_budget_usd: 5.00           # hard stop when spend exceeds this; null = unlimited
  dedup_threshold: 0.85            # Jaccard threshold for transcript deduplication
```

- [ ] **Step 2: Update run.py imports**

Add these imports to the top of `run.py` (after the existing imports):

```python
from pipeline.transcript_dedup import dedup_all
from pipeline.subtopic_splitter import split_all_topics
from utils.language_detect import detect_language, language_instruction
from utils.rag_index import build_index
from pipeline.topic_writer import write_topic_with_subtopics, coverage_pass
from llm.client import TokenBudgetExceeded
```

- [ ] **Step 3: Add Stage 2b (deduplication) to run.py**

After Stage 2 (transcription) block and before the `if _stop(3):` check, add:

```python
        # ── Stage 2b: Deduplicate transcript segments (no LLM) ───────────
        dedup_threshold = config.get("pipeline", {}).get("dedup_threshold", 0.85)
        corrected_for_dedup = corrected if _run(3) else transcripts
        deduped = dedup_all(corrected_for_dedup, threshold=dedup_threshold)
        removed = sum(
            len(t["segments"]) - len(d["segments"])
            for t, d in zip(corrected_for_dedup, deduped)
        )
        if removed:
            print(f"[Dedup] Removed {removed} duplicate segments")
```

- [ ] **Step 4: Add language detection to run.py**

After the dedup block, add:

```python
        # ── Language detection (no LLM) ──────────────────────────────────
        combined_sample = " ".join(t["full_text"][:500] for t in deduped)
        detected_lang = detect_language(combined_sample)
        lang_instr = language_instruction(detected_lang)
        if lang_instr:
            print(f"[Lang] Detected '{detected_lang}' — adding translation instruction to all prompts")
```

- [ ] **Step 5: Add Stage 3b (subtopic splitting) to run.py**

After the existing Stage 4 (group + order) block and before `if _stop(5):`, replace the existing Stage 5 block with:

```python
        # ── Stage 3b: Split topics into subtopics (cheap LLM) ────────────
        if _run(5):
            subtopics_per_topic = config.get("pipeline", {}).get("subtopics_per_topic", 5)
            groups_with_subtopics = split_all_topics(
                groups, llm, n_subtopics=subtopics_per_topic,
                base_dir=base_dir, progress=progress,
            )
        else:
            groups_with_subtopics = groups

        # ── Stage 3c: Build RAG index over transcripts (no LLM) ──────────
        if _run(5):
            rag_persist = str(base_dir / "rag_index")
            rag_col = build_index(deduped, persist_dir=rag_persist)
            print(f"[RAG] Indexed {rag_col.count()} transcript segments")

        if _stop(5):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 5: Write + verify + polish (LLM) ───────────────────────
        if _run(5):
            min_words = config.get("pipeline", {}).get("min_words_per_topic", 2500)
            coverage_target = config.get("pipeline", {}).get("coverage_target", 0.85)

            written = []
            try:
                for group in groups_with_subtopics:
                    topic_result = write_topic_with_subtopics(
                        group, rag_col, llm,
                        lang_instruction=lang_instr,
                        min_words_per_subtopic=min_words,
                        base_dir=base_dir,
                    )
                    topic_result["prose"] = coverage_pass(
                        group, deduped, topic_result["prose"], rag_col, llm,
                        target_coverage=coverage_target, base_dir=base_dir,
                    )
                    written.append(topic_result)
                    if progress:
                        progress.advance("Stage 5: Write Topics")
            except TokenBudgetExceeded as e:
                print(f"\n[Budget] {e} — stopping with {len(written)} chapters written.")
                if not written:
                    return
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: all tests PASS (existing tests still pass, new tests pass).

- [ ] **Step 7: Smoke test with --limit 2**

Run: `DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py --playlist "YOUR_TEST_PLAYLIST" --limit 2`
Expected: pipeline runs through stages, shows dedup stats, language detection, subtopic split, RAG indexing, and produces a PDF.

- [ ] **Step 8: Commit**

```bash
git add run.py config.yaml
git commit -m "feat: wire dedup, language detection, subtopic splitting, RAG-during-writing into pipeline"
```

---

## Self-Review Checklist

- [x] **Spec: book length fix** — covered by Tasks 5, 7 (subtopic splitter + per-subtopic writer)
- [x] **Spec: token optimization** — covered by Task 6+7 (RAG retrieval replaces full transcript dump), Task 2 (dedup), Task 4 (budget)
- [x] **Spec: multilingual / Hinglish** — covered by Task 3 (langdetect + Hinglish heuristic) + Task 9 (injected into prompts)
- [x] **Spec: API key injection** — covered by Task 4 (LLMClient api_key param + client_from_config)
- [x] **Spec: coverage pass** — covered by Task 8
- [x] **Spec: per-subtopic checkpoints** — covered by Task 7 (`04_subtopics/{group}__{subtopic}`)
- [x] **Spec: RAG index for later chat use** — `utils/rag_index.py` exports `build_index`, `query_chunks`, `index_book_chapters` — all reusable in Plan 2 (FastAPI backend)
- [x] **No placeholders** — all steps have complete code
- [x] **Type consistency** — `SubTopic` defined in Task 1 used consistently in Tasks 5, 7. `query_chunks` defined in Task 6 imported in Task 7.
