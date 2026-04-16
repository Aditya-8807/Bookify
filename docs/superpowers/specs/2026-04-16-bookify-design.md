# Bookify — Design Spec
**Date:** 2026-04-16  
**Version:** V1  
**Goal:** Convert a YouTube playlist into an eBook-quality PDF using a reproducible, hallucination-free LLM pipeline.

---

## 1. Overview

Single command input:
```bash
python run.py --playlist "https://youtube.com/watch?v=...&list=..."
```

Single output: `output/book.pdf` — a clean, professional, fully-cited technical book.

The pipeline is checkpointed at every stage. Any stage can be re-run independently without re-executing prior stages.

---

## 2. Book Structure

```
Title Page
Table of Contents          ← auto-generated with page numbers
Introduction               ← LLM-written from full playlist context
<Topic Name>               ← one section per thematic group (no Chapter N numbering)
<Topic Name>
...
Conclusion                 ← LLM-written synthesis
References & Resources     ← all verified source URLs, grouped by topic
```

Chapters are named by topic only (e.g. "Tokenization & Vocabulary", "The Attention Mechanism"). No "Chapter 1, Chapter 2" numbering.

**Writing style rules (enforced via LLM system prompt):**
- Clear educational prose — no bullet dumps, no transcript paraphrasing
- Each topic section: opening context → concept explanation → worked examples → summary
- Code snippets preserved verbatim from transcript where instructor wrote/showed code
- Consistent terminology throughout
- No first-person instructor voice ("In this video I will..." stripped)
- Smooth transitions between sections within a topic

---

## 3. Pipeline Architecture

```
INPUT: --playlist <url>
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: FETCH                                       │
│  yt-dlp → per-video: title, description, audio file  │
│  Description parser → extract all URLs per video     │
│  URL filter (2-pass) → keep educational refs only    │
│  Checkpoint: checkpoints/01_fetch/<video_id>.json    │
└──────────────────────────┬──────────────────────────┘
                           │ mini-batch parallel (4 at a time)
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: TRANSCRIBE                                  │
│  faster-whisper large-v3 → raw transcript per video  │
│  Audio files deleted after transcription             │
│  Checkpoint: checkpoints/02_transcripts/<id>.json    │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: GROUP + ENRICH                              │
│  LLM reads all transcripts + titles                  │
│  → clusters videos into thematic topic groups        │
│  → pools + deduplicates ref URLs per topic group     │
│  → fetches ref content (best-effort, skips failures) │
│  Checkpoint: checkpoints/03_groups.json              │
└──────────────────────────┬──────────────────────────┘
                           │ parallel per topic group
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: WRITE TOPICS                                │
│  Per topic: transcript chunks + pooled refs          │
│  LLM writes coherent prose (synthesizes, not copies) │
│  Checkpoint: checkpoints/04_topics/<topic>.md        │
└──────────────────────────┬──────────────────────────┘
                           │ parallel per topic
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4b: CITATION VERIFIER                          │
│  Step 1: Extract all factual/technical claims        │
│  Step 2: Retrieve best matching source passage       │
│          (transcript chunk or reference doc)         │
│  Step 3: Score alignment per claim                   │
│          VERIFIED (>0.8)   → inject citation         │
│          PARTIAL  (0.5-0.8)→ rewrite conservatively  │
│          UNVERIFIED (<0.5) → rewrite paragraph       │
│                              → re-verify (max 2x)    │
│                              → if still fails: remove│
│                                + restructure for flow│
│  No [?] flags. No floated claims. Fix, never flag.   │
│  Checkpoint: checkpoints/04b_verified/<topic>.md     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: ASSEMBLE                                    │
│  LLM writes: intro, topic intros, conclusion         │
│  Orders topics logically                             │
│  Collects all citations → References section         │
│  Checkpoint: checkpoints/05_book.md                  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 6: RENDER PDF                                  │
│  WeasyPrint → clean professional PDF                 │
│  Title page, auto TOC with page numbers              │
│  Inline citations, References & Resources at end     │
│  Output: output/book.pdf                             │
└─────────────────────────────────────────────────────┘
```

---

## 4. URL Filtering (Stage 1)

Video descriptions contain a mix of educational references and promotional/social links. Two-pass filter:

**Pass 1 — Domain blacklist (fast, no LLM cost):**
Drop known non-educational domains: `patreon.com`, `twitter.com`, `x.com`, `instagram.com`, `discord.gg`, `youtube.com` (self-links), `amzn.to`, `bit.ly`, `linkedin.com`, `ko-fi.com`, `gumroad.com`, `udemy.com`, `coursera.com`, and similar.

**Pass 2 — LLM relevance classification (on remaining URLs):**
Input: URL + surrounding text context from description.  
LLM classifies: `educational_reference` | `promotional` | `uncertain`.  
Keep only `educational_reference`.

---

## 5. Reference Enrichment (Stage 3)

References are collected at the **topic group level**, not per-video:
- When videos are grouped into a topic, all reference URLs from every video in that group are pooled and deduplicated
- Fetched once, stored as reference content for the whole topic
- If a video has no description URLs → topic still proceeds on transcript alone
- Fetch failures (paywalled, dead links) → logged, skipped, pipeline continues

---

## 6. LLM Client Abstraction

Thin `LLMClient` class in `llm/client.py`. Provider selected via `config.yaml`:

```yaml
llm:
  provider: openai       # or: anthropic
  model: gpt-4o          # or: claude-opus-4-6
  temperature: 0.3
```

All pipeline stages import `LLMClient` — never call provider SDKs directly. Swapping provider requires only a config change.

---

## 7. Anti-Hallucination Strategy

Hallucinations are prevented structurally, not just via prompting:

1. **Grounded context only** — LLM is always given transcript chunks + reference content as the sole input. It synthesizes from provided material, never from parametric knowledge alone.
2. **Low temperature** (0.3) — reduces creative drift.
3. **Citation Verifier loop** — every factual claim is traced to a source passage before the text is finalized. Unverifiable claims are rewritten or removed, never kept.
4. **No floating claims** — final PDF contains only prose that passed the verifier. Reader can trace any claim via inline citations.

---

## 8. Checkpointing & Resumption

Every stage writes output to `checkpoints/`. On re-run, completed stages are skipped automatically.

```bash
python run.py --playlist <url>           # full run
python run.py --playlist <url> --from 4  # resume from topic writing
python run.py --playlist <url> --from 6  # re-render PDF only
```

`checkpoints/` is gitignored. `output/` is gitignored (large binary).

---

## 9. Project Structure

```
Bookify/
├── config.yaml
├── run.py
├── pipeline/
│   ├── fetcher.py          # Stage 1
│   ├── transcriber.py      # Stage 2
│   ├── grouper.py          # Stage 3
│   ├── topic_writer.py     # Stage 4
│   ├── citation_verifier.py# Stage 4b
│   ├── assembler.py        # Stage 5
│   └── pdf_renderer.py     # Stage 6
├── llm/
│   └── client.py
├── utils/
│   ├── url_filter.py
│   └── checkpoint.py
├── checkpoints/            # gitignored
├── output/                 # gitignored
├── requirements.txt
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-16-bookify-design.md
```

---

## 10. V2 Scope (Out of Scope for V1)

Auto-discovery of external references for topic groups with no instructor-provided refs:
- arXiv search (free Python library)
- Wikipedia API
- HuggingFace docs
- Papers with Code API

V2 adds this as an additional enrichment layer in Stage 3, triggered only when a topic group has zero instructor refs.

---

## 11. Non-Goals (V1)

- EPUB or HTML output (PDF only)
- Interactive web UI
- Cloud deployment
- Multi-playlist support
- Real-time progress UI (CLI stdout only)
