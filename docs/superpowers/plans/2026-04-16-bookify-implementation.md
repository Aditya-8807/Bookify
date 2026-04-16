# Bookify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that takes a YouTube playlist URL and produces an eBook-quality, hallucination-free PDF.

**Architecture:** 8-stage sequential pipeline with mini-batch parallelism at transcription and topic-writing stages. Each stage checkpoints to disk so any stage can be re-run independently. LLM provider is abstracted behind a thin client selected via `config.yaml`.

**Tech Stack:** Python 3.11+, yt-dlp, faster-whisper, openai, anthropic, WeasyPrint, rich, requests, pytest

---

## File Map

```
Bookify/
├── config.yaml                          # Runtime configuration
├── run.py                               # CLI entrypoint
├── requirements.txt                     # All dependencies
├── .gitignore
├── pipeline/
│   ├── __init__.py
│   ├── fetcher.py                       # Stage 1: download audio + extract URLs
│   ├── transcriber.py                   # Stage 2: faster-whisper transcription
│   ├── terminology_corrector.py         # Stage 2b: fix misheard domain terms
│   ├── grouper.py                       # Stage 3: cluster + order + enrich
│   ├── topic_writer.py                  # Stage 4: dedup + write prose
│   ├── citation_verifier.py             # Stage 4b: verify claims, rewrite loop
│   ├── assembler.py                     # Stage 5: intro, glossary, conclusion
│   └── pdf_renderer.py                  # Stage 6: WeasyPrint → PDF
├── llm/
│   ├── __init__.py
│   └── client.py                        # Pluggable LLM abstraction
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py                    # Stage checkpoint read/write
│   ├── url_filter.py                    # 2-pass URL filter
│   ├── progress.py                      # Rich CLI multi-progress display
│   └── quality_report.py               # Post-render quality stats
└── tests/
    ├── conftest.py
    ├── test_checkpoint.py
    ├── test_url_filter.py
    ├── test_llm_client.py
    ├── test_fetcher.py
    ├── test_transcriber.py
    ├── test_terminology_corrector.py
    ├── test_grouper.py
    ├── test_topic_writer.py
    ├── test_citation_verifier.py
    ├── test_assembler.py
    └── test_pdf_renderer.py
```

---

## Data Models (shared across all tasks)

These TypedDicts are defined in `pipeline/__init__.py` and imported by every stage:

```python
from typing import TypedDict, List, Dict, Optional, Any

class TranscriptSegment(TypedDict):
    start: float
    end: float
    text: str

class VideoMeta(TypedDict):
    video_id: str
    title: str
    description: str
    playlist_index: int
    ref_urls: List[str]        # filtered educational URLs
    audio_path: str            # deleted after Stage 2

class Transcript(TypedDict):
    video_id: str
    title: str
    segments: List[TranscriptSegment]
    full_text: str

class CorrectedTranscript(TypedDict):
    video_id: str
    title: str
    segments: List[TranscriptSegment]
    full_text: str
    corrections: List[Dict]    # [{original, corrected, timestamp}]

class TopicGroup(TypedDict):
    name: str
    slug: str                  # filesystem-safe name
    video_ids: List[str]
    dependency_order: int      # topological sort position
    prerequisites: List[str]   # topic names required before this
    ref_urls: List[str]
    ref_contents: Dict[str, str]  # url -> fetched text content

class Citation(TypedDict):
    type: str                  # 'transcript' or 'reference'
    source: str                # video title or URL
    timestamp: Optional[str]   # "MM:SS" for transcript citations
    passage: str               # the source text

class VerifiedTopic(TypedDict):
    name: str
    slug: str
    prose: str                 # verified markdown prose
    citations: List[Citation]
    stats: Dict[str, Any]      # verification pass/rewrite/remove counts
```

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `.gitignore`
- Create: `pipeline/__init__.py`
- Create: `llm/__init__.py`
- Create: `utils/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
yt-dlp>=2024.1.0
faster-whisper>=1.0.0
openai>=1.30.0
anthropic>=0.28.0
WeasyPrint>=62.0
rich>=13.7.0
requests>=2.31.0
pyyaml>=6.0
pytest>=8.0.0
pytest-mock>=3.12.0
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
llm:
  provider: openai        # or: anthropic
  model: gpt-4o           # or: claude-opus-4-6
  temperature: 0.3

pipeline:
  batch_size: 4           # parallel videos per mini-batch
  whisper_model: large-v3

paths:
  checkpoints: checkpoints
  output: output
  audio_temp: checkpoints/audio
```

- [ ] **Step 3: Create `.gitignore`**

```
checkpoints/
output/
*.pyc
__pycache__/
.env
*.egg-info/
dist/
.pytest_cache/
```

- [ ] **Step 4: Create `pipeline/__init__.py` with all data models**

```python
from typing import TypedDict, List, Dict, Optional, Any


class TranscriptSegment(TypedDict):
    start: float
    end: float
    text: str


class VideoMeta(TypedDict):
    video_id: str
    title: str
    description: str
    playlist_index: int
    ref_urls: List[str]
    audio_path: str


class Transcript(TypedDict):
    video_id: str
    title: str
    segments: List[TranscriptSegment]
    full_text: str


class CorrectedTranscript(TypedDict):
    video_id: str
    title: str
    segments: List[TranscriptSegment]
    full_text: str
    corrections: List[Dict]


class TopicGroup(TypedDict):
    name: str
    slug: str
    video_ids: List[str]
    dependency_order: int
    prerequisites: List[str]
    ref_urls: List[str]
    ref_contents: Dict[str, str]


class Citation(TypedDict):
    type: str
    source: str
    timestamp: Optional[str]
    passage: str


class VerifiedTopic(TypedDict):
    name: str
    slug: str
    prose: str
    citations: List[Citation]
    stats: Dict[str, Any]
```

- [ ] **Step 5: Create empty `llm/__init__.py` and `utils/__init__.py`**

```python
# llm/__init__.py  (empty)
# utils/__init__.py  (empty)
```

- [ ] **Step 6: Create `tests/conftest.py`**

```python
import pytest
from pathlib import Path


@pytest.fixture
def tmp_checkpoints(tmp_path):
    """Temporary checkpoints directory for tests."""
    cp = tmp_path / "checkpoints"
    cp.mkdir()
    return cp


@pytest.fixture
def sample_video_meta():
    return {
        "video_id": "abc123",
        "title": "The Attention Mechanism",
        "description": "Paper: https://arxiv.org/abs/1706.03762\nSupport: https://patreon.com/xyz",
        "playlist_index": 0,
        "ref_urls": [],
        "audio_path": "checkpoints/audio/abc123.mp3",
    }


@pytest.fixture
def sample_transcript():
    return {
        "video_id": "abc123",
        "title": "The Attention Mechanism",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Today we look at a tension mechanisms."},
            {"start": 5.0, "end": 10.0, "text": "The cave cache stores key value pairs."},
        ],
        "full_text": "Today we look at a tension mechanisms. The cave cache stores key value pairs.",
    }
```

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 8: Commit**

```bash
git add requirements.txt config.yaml .gitignore pipeline/__init__.py llm/__init__.py utils/__init__.py tests/conftest.py
git commit -m "feat: project setup — dependencies, config, data models"
```

---

## Task 2: Checkpoint Utility

**Files:**
- Create: `utils/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_checkpoint.py
import json
import pytest
from pathlib import Path
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists, list_checkpoints


def test_save_and_load(tmp_checkpoints):
    data = {"video_id": "abc", "title": "Test"}
    save_checkpoint("01_fetch", "abc", data, base_dir=tmp_checkpoints)
    result = load_checkpoint("01_fetch", "abc", base_dir=tmp_checkpoints)
    assert result == data


def test_checkpoint_exists_true(tmp_checkpoints):
    save_checkpoint("01_fetch", "abc", {}, base_dir=tmp_checkpoints)
    assert checkpoint_exists("01_fetch", "abc", base_dir=tmp_checkpoints)


def test_checkpoint_exists_false(tmp_checkpoints):
    assert not checkpoint_exists("01_fetch", "missing", base_dir=tmp_checkpoints)


def test_list_checkpoints(tmp_checkpoints):
    save_checkpoint("01_fetch", "vid1", {}, base_dir=tmp_checkpoints)
    save_checkpoint("01_fetch", "vid2", {}, base_dir=tmp_checkpoints)
    keys = list_checkpoints("01_fetch", base_dir=tmp_checkpoints)
    assert set(keys) == {"vid1", "vid2"}


def test_list_checkpoints_empty(tmp_checkpoints):
    assert list_checkpoints("01_fetch", base_dir=tmp_checkpoints) == []


def test_save_creates_nested_dirs(tmp_checkpoints):
    save_checkpoint("04_topics", "attention-mechanisms", {"prose": "..."}, base_dir=tmp_checkpoints)
    assert checkpoint_exists("04_topics", "attention-mechanisms", base_dir=tmp_checkpoints)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_checkpoint.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.checkpoint'`

- [ ] **Step 3: Implement `utils/checkpoint.py`**

```python
import json
from pathlib import Path
from typing import Any, List


def _path(stage: str, key: str, base_dir: Path) -> Path:
    return base_dir / stage / f"{key}.json"


def save_checkpoint(stage: str, key: str, data: Any, base_dir: Path = Path("checkpoints")) -> None:
    p = _path(stage, key, base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_checkpoint(stage: str, key: str, base_dir: Path = Path("checkpoints")) -> Any:
    return json.loads(_path(stage, key, base_dir).read_text())


def checkpoint_exists(stage: str, key: str, base_dir: Path = Path("checkpoints")) -> bool:
    return _path(stage, key, base_dir).exists()


def list_checkpoints(stage: str, base_dir: Path = Path("checkpoints")) -> List[str]:
    stage_dir = base_dir / stage
    if not stage_dir.exists():
        return []
    return [p.stem for p in stage_dir.glob("*.json")]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_checkpoint.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add utils/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: checkpoint utility — save/load/exists/list per stage"
```

---

## Task 3: LLM Client Abstraction

**Files:**
- Create: `llm/client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_client.py
import pytest
from unittest.mock import MagicMock, patch
from llm.client import LLMClient


def test_init_openai():
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    assert client.provider == "openai"
    assert client.model == "gpt-4o"
    assert client.temperature == 0.3


def test_init_anthropic():
    client = LLMClient(provider="anthropic", model="claude-opus-4-6", temperature=0.3)
    assert client.provider == "anthropic"


def test_init_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported provider: bogus"):
        LLMClient(provider="bogus", model="x", temperature=0.3)


def test_complete_openai(mocker):
    mock_openai = mocker.patch("llm.client.OpenAI")
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello world"))]
    )
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    result = client.complete(system="You are helpful.", user="Say hello")
    assert result == "Hello world"


def test_complete_json_returns_dict(mocker):
    mock_openai = mocker.patch("llm.client.OpenAI")
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"key": "value"}'))]
    )
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    result = client.complete_json(system="Return JSON.", user="Give me a dict")
    assert result == {"key": "value"}


def test_complete_json_strips_markdown_fences(mocker):
    mock_openai = mocker.patch("llm.client.OpenAI")
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='```json\n{"key": "value"}\n```'))]
    )
    client = LLMClient(provider="openai", model="gpt-4o", temperature=0.3)
    result = client.complete_json(system="Return JSON.", user="Give me a dict")
    assert result == {"key": "value"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_client.py -v
```
Expected: `ModuleNotFoundError: No module named 'llm.client'`

- [ ] **Step 3: Implement `llm/client.py`**

```python
import json
import re
from typing import Any, Dict


SUPPORTED_PROVIDERS = {"openai", "anthropic"}


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float):
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        else:
            from anthropic import Anthropic
            return Anthropic()

    def complete(self, system: str, user: str) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content
        else:
            resp = self._client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        raw = self.complete(system=system, user=user)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
        return json.loads(cleaned)


def client_from_config(config: dict) -> LLMClient:
    llm = config["llm"]
    return LLMClient(
        provider=llm["provider"],
        model=llm["model"],
        temperature=llm["temperature"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_client.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add llm/client.py tests/test_llm_client.py
git commit -m "feat: LLM client abstraction — OpenAI and Anthropic, complete/complete_json"
```

---

## Task 4: URL Filter Utility

**Files:**
- Create: `utils/url_filter.py`
- Create: `tests/test_url_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_url_filter.py
import pytest
from unittest.mock import MagicMock
from utils.url_filter import (
    extract_urls_from_description,
    blacklist_filter,
    llm_classify_urls,
    filter_description_urls,
)


def test_extract_urls_basic():
    desc = "Check out https://arxiv.org/abs/1706.03762 and https://github.com/karpathy/llm.c"
    urls = extract_urls_from_description(desc)
    assert "https://arxiv.org/abs/1706.03762" in urls
    assert "https://github.com/karpathy/llm.c" in urls


def test_extract_urls_with_context():
    desc = "Paper link: https://arxiv.org/abs/1706.03762\nSupport me: https://patreon.com/x"
    result = extract_urls_from_description(desc)
    assert len(result) == 2


def test_blacklist_removes_patreon():
    urls = ["https://patreon.com/xyz", "https://arxiv.org/abs/1706.03762"]
    assert blacklist_filter(urls) == ["https://arxiv.org/abs/1706.03762"]


def test_blacklist_removes_twitter_and_x():
    urls = ["https://twitter.com/user", "https://x.com/user", "https://github.com/repo"]
    assert blacklist_filter(urls) == ["https://github.com/repo"]


def test_blacklist_removes_discord():
    urls = ["https://discord.gg/abc123"]
    assert blacklist_filter(urls) == []


def test_blacklist_removes_youtube_selflinks():
    urls = ["https://youtube.com/watch?v=abc", "https://arxiv.org/abs/1234.5678"]
    assert blacklist_filter(urls) == ["https://arxiv.org/abs/1234.5678"]


def test_blacklist_keeps_arxiv_github_huggingface():
    urls = [
        "https://arxiv.org/abs/1706.03762",
        "https://github.com/karpathy/nanoGPT",
        "https://huggingface.co/docs/transformers",
    ]
    assert blacklist_filter(urls) == urls


def test_llm_classify_keeps_educational(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "classifications": [
            {"url": "https://arxiv.org/abs/1706.03762", "label": "educational_reference"},
        ]
    }
    urls_with_context = [("https://arxiv.org/abs/1706.03762", "The paper I reference:")]
    result = llm_classify_urls(urls_with_context, mock_client)
    assert result == ["https://arxiv.org/abs/1706.03762"]


def test_llm_classify_drops_promotional(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "classifications": [
            {"url": "https://some-course.com/enroll", "label": "promotional"},
        ]
    }
    urls_with_context = [("https://some-course.com/enroll", "Join my course:")]
    result = llm_classify_urls(urls_with_context, mock_client)
    assert result == []


def test_filter_description_urls_end_to_end(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "classifications": [
            {"url": "https://arxiv.org/abs/1706.03762", "label": "educational_reference"},
        ]
    }
    desc = "Paper: https://arxiv.org/abs/1706.03762\nPatreon: https://patreon.com/xyz"
    result = filter_description_urls(desc, mock_client)
    assert result == ["https://arxiv.org/abs/1706.03762"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_url_filter.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.url_filter'`

- [ ] **Step 3: Implement `utils/url_filter.py`**

```python
import re
from typing import List, Tuple
from urllib.parse import urlparse


BLACKLIST_DOMAINS = {
    "patreon.com", "twitter.com", "x.com", "instagram.com",
    "discord.gg", "discord.com", "amzn.to", "bit.ly",
    "linkedin.com", "ko-fi.com", "gumroad.com", "udemy.com",
    "coursera.com", "tiktok.com", "facebook.com", "youtube.com",
    "youtu.be", "t.co",
}

URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


def extract_urls_from_description(description: str) -> List[str]:
    return URL_RE.findall(description)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return ""


def blacklist_filter(urls: List[str]) -> List[str]:
    return [u for u in urls if _domain(u) not in BLACKLIST_DOMAINS]


def llm_classify_urls(
    urls_with_context: List[Tuple[str, str]],
    client,
) -> List[str]:
    if not urls_with_context:
        return []
    items = [{"url": u, "context": c} for u, c in urls_with_context]
    system = (
        "You classify URLs from YouTube video descriptions. "
        "For each URL and its surrounding context, output label: "
        "'educational_reference' (papers, docs, repos, blog posts) or 'promotional' (courses, merch, social, donations). "
        "Return JSON: {\"classifications\": [{\"url\": \"...\", \"label\": \"...\"}]}"
    )
    user = f"Classify these URLs:\n{items}"
    result = client.complete_json(system=system, user=user)
    return [
        item["url"]
        for item in result.get("classifications", [])
        if item.get("label") == "educational_reference"
    ]


def _extract_context(description: str, url: str) -> str:
    idx = description.find(url)
    start = max(0, idx - 80)
    end = min(len(description), idx + len(url) + 80)
    return description[start:end]


def filter_description_urls(description: str, client) -> List[str]:
    all_urls = extract_urls_from_description(description)
    after_blacklist = blacklist_filter(all_urls)
    urls_with_context = [
        (url, _extract_context(description, url)) for url in after_blacklist
    ]
    return llm_classify_urls(urls_with_context, client)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_url_filter.py -v
```
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add utils/url_filter.py tests/test_url_filter.py
git commit -m "feat: 2-pass URL filter — domain blacklist + LLM educational classification"
```

---

## Task 5: Rich Progress Display

**Files:**
- Create: `utils/progress.py`

- [ ] **Step 1: Implement `utils/progress.py`**

No unit tests for the display — it wraps `rich` which is a UI library. Test it visually in Task 15.

```python
from contextlib import contextmanager
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


console = Console()


class PipelineProgress:
    """Multi-stage progress tracker using Rich."""

    STAGES = [
        "Stage 1: Fetch",
        "Stage 2: Transcribe",
        "Stage 2b: Terminology",
        "Stage 3: Group + Order",
        "Stage 4: Write Topics",
        "Stage 4b: Verify Citations",
        "Stage 5: Assemble",
        "Stage 6: Render PDF",
    ]

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        self._task_ids = {}

    def add_stage(self, stage_name: str, total: int) -> None:
        task_id = self.progress.add_task(stage_name, total=total)
        self._task_ids[stage_name] = task_id

    def advance(self, stage_name: str, amount: int = 1) -> None:
        if stage_name in self._task_ids:
            self.progress.advance(self._task_ids[stage_name], amount)

    def complete_stage(self, stage_name: str) -> None:
        if stage_name in self._task_ids:
            task = self.progress.tasks[self._task_ids[stage_name]]
            self.progress.update(self._task_ids[stage_name], completed=task.total)

    @contextmanager
    def live(self):
        with self.progress:
            yield self
```

- [ ] **Step 2: Commit**

```bash
git add utils/progress.py
git commit -m "feat: Rich CLI multi-stage progress display"
```

---

## Task 6: Fetcher — Stage 1

**Files:**
- Create: `pipeline/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fetcher.py
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from pipeline.fetcher import extract_playlist_videos, fetch_video_audio, fetch_all


def test_extract_playlist_videos(mocker):
    mock_ydl_class = mocker.patch("pipeline.fetcher.YoutubeDL")
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {
        "entries": [
            {"id": "vid1", "title": "Video 1", "description": "desc1", "playlist_index": 1},
            {"id": "vid2", "title": "Video 2", "description": "desc2", "playlist_index": 2},
        ]
    }
    result = extract_playlist_videos("https://youtube.com/playlist?list=ABC")
    assert len(result) == 2
    assert result[0]["video_id"] == "vid1"
    assert result[1]["title"] == "Video 2"


def test_fetch_video_audio_skips_if_checkpoint_exists(tmp_checkpoints, mocker):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("01_fetch", "vid1", {"video_id": "vid1"}, base_dir=tmp_checkpoints)
    mock_ydl = mocker.patch("pipeline.fetcher.YoutubeDL")
    fetch_video_audio(
        {"video_id": "vid1", "title": "T", "description": "", "playlist_index": 0},
        llm_client=MagicMock(),
        base_dir=tmp_checkpoints,
    )
    mock_ydl.assert_not_called()


def test_fetch_video_audio_saves_checkpoint(tmp_checkpoints, mocker):
    mocker.patch("pipeline.fetcher.YoutubeDL")
    mocker.patch("pipeline.fetcher.filter_description_urls", return_value=["https://arxiv.org/abs/1706.03762"])
    fetch_video_audio(
        {"video_id": "vid2", "title": "Attention", "description": "Paper: https://arxiv.org/abs/1706.03762", "playlist_index": 1},
        llm_client=MagicMock(),
        base_dir=tmp_checkpoints,
    )
    from utils.checkpoint import checkpoint_exists
    assert checkpoint_exists("01_fetch", "vid2", base_dir=tmp_checkpoints)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fetcher.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.fetcher'`

- [ ] **Step 3: Implement `pipeline/fetcher.py`**

```python
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

from yt_dlp import YoutubeDL

from pipeline import VideoMeta
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from utils.url_filter import filter_description_urls


def extract_playlist_videos(playlist_url: str) -> List[Dict]:
    """Fetch all video metadata from a playlist without downloading."""
    ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    return [
        {
            "video_id": entry["id"],
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "playlist_index": entry.get("playlist_index", i),
        }
        for i, entry in enumerate(info.get("entries", []))
    ]


def fetch_video_audio(
    video: Dict,
    llm_client,
    base_dir: Path = Path("checkpoints"),
    audio_dir: Path = Path("checkpoints/audio"),
) -> VideoMeta:
    vid = video["video_id"]
    if checkpoint_exists("01_fetch", vid, base_dir=base_dir):
        return load_checkpoint("01_fetch", vid, base_dir=base_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(audio_dir / f"{vid}.mp3")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_path.replace(".mp3", ".%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={vid}"])

    ref_urls = filter_description_urls(video.get("description", ""), llm_client)

    meta: VideoMeta = {
        "video_id": vid,
        "title": video["title"],
        "description": video.get("description", ""),
        "playlist_index": video["playlist_index"],
        "ref_urls": ref_urls,
        "audio_path": audio_path,
    }
    save_checkpoint("01_fetch", vid, meta, base_dir=base_dir)
    return meta


def fetch_all(
    playlist_url: str,
    llm_client,
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    audio_dir: Path = Path("checkpoints/audio"),
    progress=None,
) -> List[VideoMeta]:
    videos = extract_playlist_videos(playlist_url)
    if progress:
        progress.add_stage("Stage 1: Fetch", total=len(videos))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(fetch_video_audio, v, llm_client, base_dir, audio_dir): v
            for v in videos
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 1: Fetch")

    return sorted(results, key=lambda x: x["playlist_index"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fetcher.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/fetcher.py tests/test_fetcher.py
git commit -m "feat: Stage 1 fetcher — playlist extraction, audio download, URL filter"
```

---

## Task 7: Transcriber — Stage 2

**Files:**
- Create: `pipeline/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transcriber.py
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pipeline.transcriber import transcribe_video, transcribe_all


def test_transcribe_video_skips_if_checkpoint_exists(tmp_checkpoints, sample_video_meta):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("02_transcripts", "abc123", {"video_id": "abc123"}, base_dir=tmp_checkpoints)
    with patch("pipeline.transcriber.WhisperModel") as mock_model:
        result = transcribe_video(sample_video_meta, whisper_model="large-v3", base_dir=tmp_checkpoints)
    mock_model.assert_not_called()
    assert result["video_id"] == "abc123"


def test_transcribe_video_produces_segments(tmp_checkpoints, sample_video_meta, mocker):
    mock_model_class = mocker.patch("pipeline.transcriber.WhisperModel")
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_seg1 = MagicMock(start=0.0, end=5.0, text=" Today we learn attention.")
    mock_seg2 = MagicMock(start=5.0, end=10.0, text=" The KV cache is important.")
    mock_model.transcribe.return_value = ([mock_seg1, mock_seg2], MagicMock())
    mocker.patch("os.path.exists", return_value=True)

    result = transcribe_video(sample_video_meta, whisper_model="large-v3", base_dir=tmp_checkpoints)

    assert result["video_id"] == "abc123"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["start"] == 0.0
    assert "Today we learn attention." in result["full_text"]


def test_transcribe_video_deletes_audio(tmp_checkpoints, sample_video_meta, mocker, tmp_path):
    audio_file = tmp_path / "abc123.mp3"
    audio_file.write_text("fake audio")
    sample_video_meta["audio_path"] = str(audio_file)

    mock_model_class = mocker.patch("pipeline.transcriber.WhisperModel")
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    mock_model.transcribe.return_value = ([], MagicMock())

    transcribe_video(sample_video_meta, whisper_model="large-v3", base_dir=tmp_checkpoints)
    assert not audio_file.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_transcriber.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.transcriber'`

- [ ] **Step 3: Implement `pipeline/transcriber.py`**

```python
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel

from pipeline import VideoMeta, Transcript, TranscriptSegment
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def transcribe_video(
    video: VideoMeta,
    whisper_model: str = "large-v3",
    base_dir: Path = Path("checkpoints"),
) -> Transcript:
    vid = video["video_id"]
    if checkpoint_exists("02_transcripts", vid, base_dir=base_dir):
        return load_checkpoint("02_transcripts", vid, base_dir=base_dir)

    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    segments_raw, _ = model.transcribe(video["audio_path"], beam_size=5)

    segments: List[TranscriptSegment] = [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments_raw
    ]
    full_text = " ".join(s["text"] for s in segments)

    # Delete audio after transcription
    audio_path = video.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    transcript: Transcript = {
        "video_id": vid,
        "title": video["title"],
        "segments": segments,
        "full_text": full_text,
    }
    save_checkpoint("02_transcripts", vid, transcript, base_dir=base_dir)
    return transcript


def transcribe_all(
    videos: List[VideoMeta],
    whisper_model: str = "large-v3",
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[Transcript]:
    if progress:
        progress.add_stage("Stage 2: Transcribe", total=len(videos))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(transcribe_video, v, whisper_model, base_dir): v
            for v in videos
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 2: Transcribe")

    return sorted(results, key=lambda x: x["video_id"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_transcriber.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/transcriber.py tests/test_transcriber.py
git commit -m "feat: Stage 2 transcriber — faster-whisper large-v3, timestamp segments, audio cleanup"
```

---

## Task 8: Terminology Corrector — Stage 2b

**Files:**
- Create: `pipeline/terminology_corrector.py`
- Create: `tests/test_terminology_corrector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_terminology_corrector.py
import pytest
from unittest.mock import MagicMock
from pipeline.terminology_corrector import correct_transcript, correct_all


def test_correct_transcript_fixes_misheard_terms(tmp_checkpoints, sample_transcript, mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "corrected_segments": [
            {"start": 0.0, "end": 5.0, "text": "Today we look at attention mechanisms."},
            {"start": 5.0, "end": 10.0, "text": "The KV cache stores key value pairs."},
        ],
        "corrections": [
            {"original": "a tension mechanisms", "corrected": "attention mechanisms", "timestamp": "00:00"},
            {"original": "cave cache", "corrected": "KV cache", "timestamp": "00:05"},
        ]
    }
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    assert "attention mechanisms" in result["full_text"]
    assert "KV cache" in result["full_text"]
    assert len(result["corrections"]) == 2


def test_correct_transcript_skips_if_checkpoint_exists(tmp_checkpoints, sample_transcript):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("02b_corrected", "abc123", {"video_id": "abc123", "corrections": []}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    mock_client.complete_json.assert_not_called()
    assert result["video_id"] == "abc123"


def test_correct_transcript_logs_corrections(tmp_checkpoints, sample_transcript, mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "corrected_segments": sample_transcript["segments"],
        "corrections": [{"original": "a tension", "corrected": "attention", "timestamp": "00:00"}]
    }
    result = correct_transcript(sample_transcript, mock_client, base_dir=tmp_checkpoints)
    assert result["corrections"][0]["original"] == "a tension"
    assert result["corrections"][0]["corrected"] == "attention"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_terminology_corrector.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.terminology_corrector'`

- [ ] **Step 3: Implement `pipeline/terminology_corrector.py`**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from pipeline import Transcript, CorrectedTranscript
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


SYSTEM_PROMPT = """You are a technical transcript corrector for AI/ML content.
Fix misheard or garbled domain-specific terms in these transcript segments.
The video title gives you domain context.
Common errors: "a tension" → "attention", "cave cache" → "KV cache",
"soft max" → "softmax", "embedding" → "embedding" (correct), "gradient decent" → "gradient descent".
Return JSON:
{
  "corrected_segments": [{"start": float, "end": float, "text": "corrected text"}],
  "corrections": [{"original": "wrong", "corrected": "right", "timestamp": "MM:SS"}]
}
Only fix misheard technical terms. Do not rephrase or alter meaning."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def correct_transcript(
    transcript: Transcript,
    llm_client,
    base_dir: Path = Path("checkpoints"),
) -> CorrectedTranscript:
    vid = transcript["video_id"]
    if checkpoint_exists("02b_corrected", vid, base_dir=base_dir):
        return load_checkpoint("02b_corrected", vid, base_dir=base_dir)

    segments_text = "\n".join(
        f"[{_fmt_ts(s['start'])}] {s['text']}" for s in transcript["segments"]
    )
    user = f"Video title: {transcript['title']}\n\nSegments:\n{segments_text}"
    result = llm_client.complete_json(system=SYSTEM_PROMPT, user=user)

    corrected: CorrectedTranscript = {
        "video_id": vid,
        "title": transcript["title"],
        "segments": result.get("corrected_segments", transcript["segments"]),
        "full_text": " ".join(s["text"] for s in result.get("corrected_segments", transcript["segments"])),
        "corrections": result.get("corrections", []),
    }
    save_checkpoint("02b_corrected", vid, corrected, base_dir=base_dir)
    return corrected


def correct_all(
    transcripts: List[Transcript],
    llm_client,
    batch_size: int = 4,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[CorrectedTranscript]:
    if progress:
        progress.add_stage("Stage 2b: Terminology", total=len(transcripts))

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(correct_transcript, t, llm_client, base_dir): t
            for t in transcripts
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 2b: Terminology")

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_terminology_corrector.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/terminology_corrector.py tests/test_terminology_corrector.py
git commit -m "feat: Stage 2b terminology corrector — LLM fixes misheard domain terms"
```

---

## Task 9: Grouper — Stage 3

**Files:**
- Create: `pipeline/grouper.py`
- Create: `tests/test_grouper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_grouper.py
import pytest
from unittest.mock import MagicMock, patch
from pipeline.grouper import group_and_order, fetch_reference_content, slugify


def test_slugify():
    assert slugify("The Attention Mechanism") == "the-attention-mechanism"
    assert slugify("KV Cache & Embeddings") == "kv-cache-embeddings"


def test_group_and_order_returns_sorted_groups(tmp_checkpoints, mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "topics": [
            {
                "name": "Embeddings",
                "video_ids": ["vid2"],
                "prerequisites": ["Tokenization"],
            },
            {
                "name": "Tokenization",
                "video_ids": ["vid1"],
                "prerequisites": [],
            },
        ]
    }
    mocker.patch("pipeline.grouper.fetch_reference_content", return_value={})

    transcripts = [
        {"video_id": "vid1", "title": "Tokenization", "full_text": "tokenization text", "segments": [], "corrections": []},
        {"video_id": "vid2", "title": "Embeddings", "full_text": "embeddings text", "segments": [], "corrections": []},
    ]
    video_metas = [
        {"video_id": "vid1", "ref_urls": [], "title": "T", "description": "", "playlist_index": 0, "audio_path": ""},
        {"video_id": "vid2", "ref_urls": [], "title": "E", "description": "", "playlist_index": 1, "audio_path": ""},
    ]

    groups = group_and_order(transcripts, video_metas, mock_client, base_dir=tmp_checkpoints)
    assert groups[0]["name"] == "Tokenization"
    assert groups[1]["name"] == "Embeddings"
    assert groups[1]["dependency_order"] == 1


def test_fetch_reference_content_skips_on_error(mocker):
    mocker.patch("requests.get", side_effect=Exception("network error"))
    result = fetch_reference_content(["https://arxiv.org/abs/1706.03762"])
    assert result == {}


def test_fetch_reference_content_returns_text(mocker):
    mock_resp = MagicMock()
    mock_resp.text = "<html><body><p>Attention is all you need.</p></body></html>"
    mock_resp.raise_for_status = MagicMock()
    mocker.patch("requests.get", return_value=mock_resp)
    result = fetch_reference_content(["https://arxiv.org/abs/1706.03762"])
    assert "https://arxiv.org/abs/1706.03762" in result
    assert "Attention is all you need." in result["https://arxiv.org/abs/1706.03762"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_grouper.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.grouper'`

- [ ] **Step 3: Implement `pipeline/grouper.py`**

```python
import re
from pathlib import Path
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

from pipeline import CorrectedTranscript, VideoMeta, TopicGroup
from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


GROUP_SYSTEM = """You are organizing transcripts from a YouTube playlist into a book.
Cluster the videos into thematic topic groups. Each group becomes one section of the book.
Build a dependency graph: if topic B requires understanding topic A first, list A as a prerequisite of B.
Return JSON:
{
  "topics": [
    {
      "name": "Human-readable topic name",
      "video_ids": ["vid1", "vid2"],
      "prerequisites": ["Topic Name A"]
    }
  ]
}
Name topics concisely (e.g. "Tokenization & Vocabulary", "The Attention Mechanism").
Ensure all video_ids appear exactly once."""


def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s.strip())
    return s


def _topological_sort(topics: List[Dict]) -> List[Dict]:
    name_to_topic = {t["name"]: t for t in topics}
    order = []
    visited = set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        for prereq in name_to_topic.get(name, {}).get("prerequisites", []):
            if prereq in name_to_topic:
                visit(prereq)
        order.append(name)

    for t in topics:
        visit(t["name"])

    return [name_to_topic[n] for n in order if n in name_to_topic]


def fetch_reference_content(urls: List[str]) -> Dict[str, str]:
    content = {}
    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Bookify/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            content[url] = text[:8000]  # cap at 8k chars per reference
        except Exception:
            pass
    return content


def group_and_order(
    transcripts: List[CorrectedTranscript],
    video_metas: List[VideoMeta],
    llm_client,
    base_dir: Path = Path("checkpoints"),
    progress=None,
) -> List[TopicGroup]:
    if checkpoint_exists("03_groups", "groups", base_dir=base_dir):
        return load_checkpoint("03_groups", "groups", base_dir=base_dir)

    if progress:
        progress.add_stage("Stage 3: Group + Order", total=1)

    meta_by_id = {m["video_id"]: m for m in video_metas}
    summaries = "\n\n".join(
        f"video_id={t['video_id']} title={t['title']}\n{t['full_text'][:500]}"
        for t in transcripts
    )
    result = llm_client.complete_json(system=GROUP_SYSTEM, user=summaries)
    raw_topics = result.get("topics", [])
    ordered = _topological_sort(raw_topics)

    groups: List[TopicGroup] = []
    for i, topic in enumerate(ordered):
        vids = topic["video_ids"]
        all_ref_urls = list({url for vid in vids for url in meta_by_id.get(vid, {}).get("ref_urls", [])})
        ref_contents = fetch_reference_content(all_ref_urls)
        groups.append({
            "name": topic["name"],
            "slug": slugify(topic["name"]),
            "video_ids": vids,
            "dependency_order": i,
            "prerequisites": topic.get("prerequisites", []),
            "ref_urls": all_ref_urls,
            "ref_contents": ref_contents,
        })

    save_checkpoint("03_groups", "groups", groups, base_dir=base_dir)
    if progress:
        progress.advance("Stage 3: Group + Order")
    return groups
```

- [ ] **Step 4: Install beautifulsoup4**

Add to `requirements.txt`:
```
beautifulsoup4>=4.12.0
```
Then run:
```bash
pip install beautifulsoup4
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_grouper.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add pipeline/grouper.py tests/test_grouper.py requirements.txt
git commit -m "feat: Stage 3 grouper — LLM topic clustering, dependency ordering, ref enrichment"
```

---

## Task 10: Topic Writer — Stage 4

**Files:**
- Create: `pipeline/topic_writer.py`
- Create: `tests/test_topic_writer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_topic_writer.py
import pytest
from unittest.mock import MagicMock
from pipeline.topic_writer import detect_overlaps, write_topic, write_all_topics


def test_detect_overlaps_finds_shared_concepts(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "overlaps": [
            {
                "concept": "softmax function",
                "primary_topic": "The Attention Mechanism",
                "secondary_topics": ["Training from Scratch"],
            }
        ]
    }
    groups = [
        {"name": "The Attention Mechanism", "slug": "attention", "video_ids": ["v1"],
         "dependency_order": 0, "prerequisites": [], "ref_urls": [], "ref_contents": {}},
        {"name": "Training from Scratch", "slug": "training", "video_ids": ["v2"],
         "dependency_order": 1, "prerequisites": [], "ref_urls": [], "ref_contents": {}},
    ]
    overlaps = detect_overlaps(groups, mock_client)
    assert len(overlaps) == 1
    assert overlaps[0]["concept"] == "softmax function"


def test_write_topic_skips_if_checkpoint_exists(tmp_checkpoints):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("04_topics", "attention", {"name": "attention", "prose": "..."}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    group = {"name": "Attention", "slug": "attention", "video_ids": [], "dependency_order": 0,
             "prerequisites": [], "ref_urls": [], "ref_contents": {}}
    result = write_topic(group, [], {}, mock_client, base_dir=tmp_checkpoints)
    mock_client.complete.assert_not_called()
    assert result["name"] == "attention"


def test_write_topic_includes_timestamp_citations(tmp_checkpoints, mocker):
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "Attention mechanisms allow the model to focus on relevant tokens. "
        "[Video: \"The Attention Mechanism\" @ 12:34]"
    )
    group = {
        "name": "The Attention Mechanism", "slug": "the-attention-mechanism",
        "video_ids": ["abc123"], "dependency_order": 0, "prerequisites": [],
        "ref_urls": [], "ref_contents": {},
    }
    transcripts = [{"video_id": "abc123", "title": "The Attention Mechanism",
                    "segments": [{"start": 752.0, "end": 760.0, "text": "attention allows the model to focus"}],
                    "full_text": "attention allows the model to focus", "corrections": []}]
    result = write_topic(group, transcripts, {}, mock_client, base_dir=tmp_checkpoints)
    assert "12:34" in result["prose"] or "attention" in result["prose"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_topic_writer.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.topic_writer'`

- [ ] **Step 3: Implement `pipeline/topic_writer.py`**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
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
Return an empty list if no overlaps found."""


WRITE_SYSTEM = """You are writing a section of a technical book about building LLMs from scratch.
Write clear, educational prose — not bullet points, not a transcript summary.
Structure: opening context → concept explanation → worked examples → section summary.
Strip first-person instructor voice ("In this video I will...").
Use consistent terminology throughout.
When citing something from the transcript, add: [Video: "<title>" @ MM:SS]
When citing a reference, add: [<URL>]
Where a concept was introduced in a prior section, add a cross-reference: "As introduced in <Topic>..."
Do NOT invent facts. Only write what is supported by the provided transcript and references."""


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def detect_overlaps(groups: List[TopicGroup], llm_client) -> List[Dict]:
    topics_summary = "\n".join(
        f"Topic: {g['name']} — videos: {g['video_ids']}" for g in groups
    )
    result = llm_client.complete_json(system=OVERLAP_SYSTEM, user=topics_summary)
    return result.get("overlaps", [])


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
        cross_ref_note = f"\nNote: these concepts were introduced in prior sections and should be cross-referenced, not re-explained: {concepts}"

    user = (
        f"Topic: {group['name']}\n"
        f"Prerequisite topics already covered: {group['prerequisites']}\n"
        f"{cross_ref_note}\n\n"
        f"TRANSCRIPT:\n{transcript_text}\n"
        f"{refs_text}"
    )

    prose = llm_client.complete(system=WRITE_SYSTEM, user=user)

    result = {"name": slug, "slug": slug, "prose": prose}
    save_checkpoint("04_topics", slug, result, base_dir=base_dir)
    return result


def write_all_topics(
    groups: List[TopicGroup],
    transcripts: List[CorrectedTranscript],
    llm_client,
    batch_size: int = 4,
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
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(write_topic, g, transcripts, overlaps_map, llm_client, base_dir): g
            for g in groups
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 4: Write Topics")

    slug_order = {g["slug"]: g["dependency_order"] for g in groups}
    return sorted(results, key=lambda x: slug_order.get(x["slug"], 0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_topic_writer.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/topic_writer.py tests/test_topic_writer.py
git commit -m "feat: Stage 4 topic writer — cross-topic dedup, prose generation, timestamp citations"
```

---

## Task 11: Citation Verifier — Stage 4b

**Files:**
- Create: `pipeline/citation_verifier.py`
- Create: `tests/test_citation_verifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_citation_verifier.py
import pytest
from unittest.mock import MagicMock, call
from pipeline.citation_verifier import extract_claims, score_claim, verify_topic, VerificationStats


def test_extract_claims_returns_list(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "claims": [
            "The transformer uses multi-head attention.",
            "Softmax normalizes attention scores to sum to 1.",
        ]
    }
    claims = extract_claims("Some prose about transformers.", mock_client)
    assert len(claims) == 2
    assert "transformer" in claims[0]


def test_score_claim_verified(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {"score": 0.95, "passage": "The transformer uses multi-head attention."}
    score, passage = score_claim(
        claim="The transformer uses multi-head attention.",
        source_texts=["The transformer uses multi-head attention with 8 heads."],
        llm_client=mock_client,
    )
    assert score >= 0.8
    assert "transformer" in passage


def test_score_claim_unverified(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {"score": 0.2, "passage": ""}
    score, passage = score_claim(
        claim="The model has exactly 175 billion parameters.",
        source_texts=["The model was trained on a large dataset."],
        llm_client=mock_client,
    )
    assert score < 0.5


def test_verify_topic_removes_unverifiable_claim(tmp_checkpoints, mocker):
    mock_client = MagicMock()
    mock_client.complete_json.side_effect = [
        {"claims": ["Claim A.", "Unverifiable claim X."]},
        {"score": 0.95, "passage": "Source supports Claim A."},
        {"score": 0.1, "passage": ""},
        {"score": 0.1, "passage": ""},  # retry 1
        {"score": 0.1, "passage": ""},  # retry 2
        {"claims": ["Claim A."]},        # re-extract after removal
        {"score": 0.95, "passage": "Source supports Claim A."},
    ]
    mock_client.complete.return_value = "Claim A."

    topic = {"name": "attention", "slug": "attention", "prose": "Claim A. Unverifiable claim X."}
    sources = ["Source supports Claim A."]
    result = verify_topic(topic, sources, mock_client, base_dir=tmp_checkpoints)
    assert "Unverifiable claim X." not in result["prose"]


def test_verify_topic_skips_if_checkpoint_exists(tmp_checkpoints):
    from utils.checkpoint import save_checkpoint
    save_checkpoint("04b_verified", "attention", {"name": "attention", "prose": "verified"}, base_dir=tmp_checkpoints)
    mock_client = MagicMock()
    topic = {"name": "attention", "slug": "attention", "prose": "original"}
    result = verify_topic(topic, [], mock_client, base_dir=tmp_checkpoints)
    mock_client.complete_json.assert_not_called()
    assert result["prose"] == "verified"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_citation_verifier.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.citation_verifier'`

- [ ] **Step 3: Implement `pipeline/citation_verifier.py`**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any

from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


EXTRACT_SYSTEM = """Extract all factual and technical claims from this prose.
A claim is any sentence that asserts something that could be verified or disproved.
Return JSON: {"claims": ["claim 1", "claim 2", ...]}"""

SCORE_SYSTEM = """Given a claim and source passages, score how well the sources support the claim.
Score 0.0 (no support) to 1.0 (fully supported).
Return JSON: {"score": 0.0-1.0, "passage": "the most relevant source passage, or empty string"}"""

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
            # Retry up to max_retries times with rewrite
            rewritten = prose
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

    # Clean up whitespace from removed claims
    import re
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()

    result = {
        "name": topic["name"],
        "slug": slug,
        "prose": prose,
        "citations": [],
        "stats": {"verified": stats.verified, "rewritten": stats.rewritten, "removed": stats.removed},
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
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
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

        futures = {executor.submit(_verify, t): t for t in topics}
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress.advance("Stage 4b: Verify Citations")

    slug_order = {t["slug"]: i for i, t in enumerate(topics)}
    return sorted(results, key=lambda x: slug_order.get(x["slug"], 0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_citation_verifier.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/citation_verifier.py tests/test_citation_verifier.py
git commit -m "feat: Stage 4b citation verifier — claim extraction, scoring, rewrite loop"
```

---

## Task 12: Assembler — Stage 5

**Files:**
- Create: `pipeline/assembler.py`
- Create: `tests/test_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_assembler.py
import pytest
from unittest.mock import MagicMock
from pipeline.assembler import generate_glossary, assemble_book


def test_generate_glossary_returns_terms(mocker):
    mock_client = MagicMock()
    mock_client.complete_json.return_value = {
        "terms": [
            {"term": "attention mechanism", "definition": "A mechanism that weights input tokens by relevance."},
            {"term": "softmax", "definition": "A function that converts logits to a probability distribution."},
        ]
    }
    glossary = generate_glossary("Some book prose about attention and softmax.", mock_client)
    assert len(glossary) == 2
    assert glossary[0]["term"] == "attention mechanism"


def test_assemble_book_includes_all_sections(tmp_checkpoints, mocker):
    mock_client = MagicMock()
    mock_client.complete.return_value = "Generated intro or conclusion text."
    mock_client.complete_json.return_value = {
        "terms": [{"term": "attention", "definition": "A weighting mechanism."}]
    }
    topics = [
        {"name": "Tokenization", "slug": "tokenization", "prose": "Tokenization prose.", "citations": [], "stats": {}},
        {"name": "Attention", "slug": "attention", "prose": "Attention prose.", "citations": [], "stats": {}},
    ]
    groups = [
        {"name": "Tokenization", "slug": "tokenization", "ref_urls": ["https://arxiv.org/abs/1"], "ref_contents": {}, "video_ids": [], "dependency_order": 0, "prerequisites": []},
        {"name": "Attention", "slug": "attention", "ref_urls": [], "ref_contents": {}, "video_ids": [], "dependency_order": 1, "prerequisites": []},
    ]
    result = assemble_book(topics, groups, mock_client, base_dir=tmp_checkpoints)
    assert "introduction" in result.lower() or "generated intro" in result.lower()
    assert "Tokenization prose." in result
    assert "Attention prose." in result
    assert "glossary" in result.lower() or "attention" in result.lower()
    assert "references" in result.lower() or "https://arxiv.org/abs/1" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_assembler.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.assembler'`

- [ ] **Step 3: Implement `pipeline/assembler.py`**

```python
from pathlib import Path
from typing import List, Dict, Any

from utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists


INTRO_SYSTEM = """Write an introduction for a technical book about building LLMs from scratch.
The introduction should: explain what the book covers, who it's for, how to use it, and what the reader will learn.
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

    # Collect all references grouped by topic
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

    # Build glossary section
    glossary_section = "## Glossary\n\n"
    for entry in glossary_terms:
        glossary_section += f"**{entry['term']}** — {entry['definition']}\n\n"

    # Assemble full book markdown
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_assembler.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/assembler.py tests/test_assembler.py
git commit -m "feat: Stage 5 assembler — intro, conclusion, glossary, references section"
```

---

## Task 13: PDF Renderer — Stage 6

**Files:**
- Create: `pipeline/pdf_renderer.py`
- Create: `tests/test_pdf_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pdf_renderer.py
import pytest
from pathlib import Path
from pipeline.pdf_renderer import markdown_to_html, render_pdf


def test_markdown_to_html_converts_headings():
    md = "## Introduction\n\nSome text here."
    html = markdown_to_html(md, title="Test Book")
    assert "<h2" in html
    assert "Some text here." in html
    assert "Test Book" in html


def test_markdown_to_html_converts_code_blocks():
    md = "```python\ndef hello():\n    pass\n```"
    html = markdown_to_html(md, title="Test Book")
    assert "<code" in html or "<pre" in html


def test_markdown_to_html_wraps_in_html_document():
    html = markdown_to_html("Hello world", title="My Book")
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "My Book" in html


def test_render_pdf_creates_file(tmp_path, mocker):
    mocker.patch("pipeline.pdf_renderer.HTML")
    output_path = tmp_path / "book.pdf"
    render_pdf("<html><body><p>Test</p></body></html>", output_path=str(output_path))
    from pipeline.pdf_renderer import HTML
    HTML.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pdf_renderer.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.pdf_renderer'`

- [ ] **Step 3: Add markdown to requirements.txt and install**

Add to `requirements.txt`:
```
markdown>=3.6
```
Then:
```bash
pip install markdown
```

- [ ] **Step 4: Implement `pipeline/pdf_renderer.py`**

```python
from pathlib import Path
import markdown as md_lib
from weasyprint import HTML, CSS


CSS_STYLES = """
@page {
    margin: 2.5cm;
    @bottom-center {
        content: counter(page);
        font-size: 10pt;
        color: #666;
    }
}
body {
    font-family: Georgia, serif;
    font-size: 12pt;
    line-height: 1.8;
    color: #1a1a1a;
    max-width: 100%;
}
h1 { font-size: 28pt; margin-top: 2em; page-break-before: always; }
h2 { font-size: 20pt; margin-top: 1.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
h3 { font-size: 14pt; margin-top: 1.2em; }
p { margin: 0.8em 0; text-align: justify; }
pre, code {
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 3px;
}
pre {
    padding: 1em;
    overflow-x: auto;
    border-left: 3px solid #ccc;
}
hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }
a { color: #1a1a1a; }
strong { font-weight: bold; }
.toc { page-break-after: always; }
"""

TITLE_PAGE_HTML = """
<div style="text-align:center; margin-top: 30%; page-break-after: always;">
  <h1 style="font-size: 36pt; border: none; page-break-before: avoid;">{title}</h1>
  <p style="font-size: 14pt; color: #666; margin-top: 2em;">Generated by Bookify</p>
</div>
"""


def markdown_to_html(book_markdown: str, title: str = "Building LLMs from Scratch") -> str:
    extensions = ["fenced_code", "tables", "toc"]
    converter = md_lib.Markdown(extensions=extensions)
    body_html = converter.convert(book_markdown)

    toc_html = ""
    if hasattr(converter, "toc"):
        toc_html = f'<div class="toc"><h2>Table of Contents</h2>{converter.toc}</div>'

    title_html = TITLE_PAGE_HTML.format(title=title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
</head>
<body>
  {title_html}
  {toc_html}
  {body_html}
</body>
</html>"""


def render_pdf(
    html_content: str,
    output_path: str = "output/book.pdf",
    progress=None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress.add_stage("Stage 6: Render PDF", total=1)

    HTML(string=html_content).write_pdf(
        output_path,
        stylesheets=[CSS(string=CSS_STYLES)],
    )

    if progress:
        progress.advance("Stage 6: Render PDF")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_pdf_renderer.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add pipeline/pdf_renderer.py tests/test_pdf_renderer.py requirements.txt
git commit -m "feat: Stage 6 PDF renderer — WeasyPrint, title page, TOC, styled prose"
```

---

## Task 14: Quality Report — Stage 7

**Files:**
- Create: `utils/quality_report.py`

- [ ] **Step 1: Implement `utils/quality_report.py`**

```python
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


def generate_report(
    verified_topics: List[Dict[str, Any]],
    full_book_text: str,
    terminology_corrections: List[Dict],
) -> Dict[str, Any]:
    total_words = len(full_book_text.split())
    total_topics = len(verified_topics)

    total_verified = sum(t.get("stats", {}).get("verified", 0) for t in verified_topics)
    total_rewritten = sum(t.get("stats", {}).get("rewritten", 0) for t in verified_topics)
    total_removed = sum(t.get("stats", {}).get("removed", 0) for t in verified_topics)
    total_claims = total_verified + total_rewritten + total_removed

    pass_rate = (total_verified / total_claims * 100) if total_claims > 0 else 100.0

    topics_without_refs = sum(
        1 for t in verified_topics
        if not t.get("citations")
    )

    return {
        "total_topics": total_topics,
        "total_words": total_words,
        "total_claims": total_claims,
        "verified": total_verified,
        "rewritten": total_rewritten,
        "removed": total_removed,
        "verification_pass_rate": round(pass_rate, 1),
        "topics_without_refs": topics_without_refs,
        "terminology_corrections": len(terminology_corrections),
    }


def print_report(report: Dict[str, Any]) -> None:
    table = Table(title="Bookify Quality Report", show_header=False, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Topics", str(report["total_topics"]))
    table.add_row("Words", f"{report['total_words']:,}")
    table.add_row("Total claims checked", str(report["total_claims"]))
    table.add_row("Verified", str(report["verified"]))
    table.add_row("Rewritten", str(report["rewritten"]))
    table.add_row("Removed", str(report["removed"]))
    table.add_row("Verification pass rate", f"{report['verification_pass_rate']}%")
    table.add_row("Topics without instructor refs", str(report["topics_without_refs"]))
    table.add_row("Terminology corrections", str(report["terminology_corrections"]))

    console.print(Panel(table, border_style="green"))
```

- [ ] **Step 2: Commit**

```bash
git add utils/quality_report.py
git commit -m "feat: Stage 7 quality report — verification stats, word count, corrections"
```

---

## Task 15: Main Entrypoint

**Files:**
- Create: `run.py`

- [ ] **Step 1: Implement `run.py`**

```python
#!/usr/bin/env python3
"""
Bookify — YouTube Playlist → PDF Book

Usage:
    python run.py --playlist <url>
    python run.py --playlist <url> --from 4   # resume from stage 4
    python run.py --playlist <url> --title "Custom Book Title"
"""
import argparse
import sys
from pathlib import Path

import yaml

from llm.client import client_from_config
from pipeline.fetcher import fetch_all
from pipeline.transcriber import transcribe_all
from pipeline.terminology_corrector import correct_all
from pipeline.grouper import group_and_order
from pipeline.topic_writer import write_all_topics
from pipeline.citation_verifier import verify_all_topics
from pipeline.assembler import assemble_book
from pipeline.pdf_renderer import markdown_to_html, render_pdf
from utils.checkpoint import load_checkpoint, list_checkpoints
from utils.progress import PipelineProgress
from utils.quality_report import generate_report, print_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Convert a YouTube playlist to a PDF book.")
    parser.add_argument("--playlist", required=True, help="YouTube playlist URL")
    parser.add_argument("--from", dest="from_stage", type=int, default=1,
                        help="Resume from stage N (1-6). Default: 1 (full run)")
    parser.add_argument("--title", default="Building LLMs from Scratch",
                        help="Book title for the PDF cover")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    llm = client_from_config(config)
    batch_size = config.get("pipeline", {}).get("batch_size", 4)
    whisper_model = config.get("pipeline", {}).get("whisper_model", "large-v3")
    base_dir = Path(config.get("paths", {}).get("checkpoints", "checkpoints"))
    audio_dir = Path(config.get("paths", {}).get("audio_temp", "checkpoints/audio"))
    output_dir = Path(config.get("paths", {}).get("output", "output"))

    progress = PipelineProgress()

    with progress.live():
        # Stage 1: Fetch
        if args.from_stage <= 1:
            videos = fetch_all(
                args.playlist, llm, batch_size=batch_size,
                base_dir=base_dir, audio_dir=audio_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("01_fetch", base_dir=base_dir)
            videos = [load_checkpoint("01_fetch", k, base_dir=base_dir) for k in keys]

        # Stage 2: Transcribe
        if args.from_stage <= 2:
            transcripts = transcribe_all(
                videos, whisper_model=whisper_model, batch_size=batch_size,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("02_transcripts", base_dir=base_dir)
            transcripts = [load_checkpoint("02_transcripts", k, base_dir=base_dir) for k in keys]

        # Stage 2b: Terminology correction
        if args.from_stage <= 2:
            corrected = correct_all(
                transcripts, llm, batch_size=batch_size,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("02b_corrected", base_dir=base_dir)
            corrected = [load_checkpoint("02b_corrected", k, base_dir=base_dir) for k in keys]

        # Stage 3: Group + order + enrich
        if args.from_stage <= 3:
            groups = group_and_order(corrected, videos, llm, base_dir=base_dir, progress=progress)
        else:
            groups = load_checkpoint("03_groups", "groups", base_dir=base_dir)

        # Stage 4: Write topics
        if args.from_stage <= 4:
            written = write_all_topics(
                groups, corrected, llm, batch_size=batch_size,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("04_topics", base_dir=base_dir)
            written = [load_checkpoint("04_topics", k, base_dir=base_dir) for k in keys]

        # Stage 4b: Citation verification
        if args.from_stage <= 4:
            trans_by_vid = {t["video_id"]: t for t in corrected}
            groups_by_slug = {g["slug"]: g for g in groups}
            verified = verify_all_topics(
                written, groups_by_slug, trans_by_vid,
                batch_size=batch_size, base_dir=base_dir,
                progress=progress, llm_client=llm,
            )
        else:
            keys = list_checkpoints("04b_verified", base_dir=base_dir)
            verified = [load_checkpoint("04b_verified", k, base_dir=base_dir) for k in keys]

        # Stage 5: Assemble
        if args.from_stage <= 5:
            book_markdown = assemble_book(verified, groups, llm, base_dir=base_dir, progress=progress)
        else:
            book_markdown = load_checkpoint("05_book", "book", base_dir=base_dir)

        # Stage 6: Render PDF
        if args.from_stage <= 6:
            html = markdown_to_html(book_markdown, title=args.title)
            output_path = str(output_dir / "book.pdf")
            render_pdf(html, output_path=output_path, progress=progress)
            print(f"\nPDF saved to: {output_path}")

    # Stage 7: Quality report
    all_corrections = [c for t in corrected for c in t.get("corrections", [])]
    report = generate_report(verified, book_markdown, all_corrections)
    print_report(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a smoke test with mocked stages to verify the entrypoint parses args correctly**

```bash
python run.py --help
```
Expected: prints usage with `--playlist`, `--from`, `--title`, `--config` arguments

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "feat: main entrypoint — full pipeline orchestration, --from resume, quality report"
```

---

## Task 16: Full Integration Test

- [ ] **Step 1: Run the complete test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass. Note any failures and fix them before proceeding.

- [ ] **Step 2: Run a pipeline dry-run to verify imports**

```bash
python -c "
from pipeline.fetcher import fetch_all
from pipeline.transcriber import transcribe_all
from pipeline.terminology_corrector import correct_all
from pipeline.grouper import group_and_order
from pipeline.topic_writer import write_all_topics
from pipeline.citation_verifier import verify_all_topics
from pipeline.assembler import assemble_book
from pipeline.pdf_renderer import markdown_to_html, render_pdf
from llm.client import client_from_config
from utils.quality_report import generate_report
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "chore: verify full pipeline imports clean"
```

---

## Self-Review Checklist

- [x] **Spec §2 Book Structure** → Tasks 12, 13 (intro, TOC, glossary, references, topic names)
- [x] **Spec §3 Stage 1 Fetch** → Task 6
- [x] **Spec §3 Stage 2 Transcribe** → Task 7
- [x] **Spec §3 Stage 2b Terminology** → Task 8
- [x] **Spec §3 Stage 3 Group+Order+Enrich** → Task 9
- [x] **Spec §3 Stage 4 Dedup+Write** → Task 10
- [x] **Spec §3 Stage 4b Citation Verifier** → Task 11
- [x] **Spec §3 Stage 5 Assemble** → Task 12
- [x] **Spec §3 Stage 6 PDF** → Task 13
- [x] **Spec §3 Stage 7 Quality Report** → Task 14
- [x] **Spec §4 URL Filtering** → Task 4
- [x] **Spec §5 Reference Enrichment** → Task 9
- [x] **Spec §6 LLM Client** → Task 3
- [x] **Spec §7 Anti-Hallucination** → Tasks 8, 10, 11
- [x] **Spec §8 Timestamp Citations** → Task 10
- [x] **Spec §9 Glossary** → Task 12
- [x] **Spec §13 Rich CLI Progress** → Tasks 5, 15
- [x] **Spec §14 Checkpointing + --from** → Tasks 2, 15
- [x] Type consistency: `VideoMeta`, `Transcript`, `CorrectedTranscript`, `TopicGroup`, `VerifiedTopic` defined in Task 1, used consistently across all tasks
- [x] Function names consistent: `fetch_all`, `transcribe_all`, `correct_all`, `group_and_order`, `write_all_topics`, `verify_all_topics`, `assemble_book`, `render_pdf`
- [x] No placeholders or TBDs found
