# Bookify — Open Source Scale & Multilingual Design
Date: 2026-06-03

## Goals
- Scale Bookify as a public open source project
- Add multilingual input support (auto-detect, always output English)
- Fix book length (43-video playlist should produce 400-600 pages, not 60)
- Optimize token cost via RAG-during-writing
- Add NoteGPT-style RAG chat over transcripts + generated book
- Build a Next.js frontend (hosted on Vercel) as a SaaS demo
- Make it a standout resume/portfolio project

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Next.js (Vercel)                    │
│  - Playlist URL input + API key + config form        │
│  - Job progress (SSE stream)                         │
│  - PDF viewer + download                            │
│  - Chat interface (NoteGPT-style Q&A)               │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP / SSE
┌──────────────────▼──────────────────────────────────┐
│              FastAPI Backend                         │
│  POST /jobs          → start pipeline job            │
│  GET  /jobs/{id}     → SSE progress stream           │
│  GET  /jobs/{id}/pdf → download PDF                  │
│  POST /jobs/{id}/chat → RAG Q&A                      │
│  asyncio background tasks (no Celery for v1)         │
└──────┬────────────────────┬────────────────────────--┘
       │                    │
┌──────▼──────┐    ┌────────▼────────┐
│  Pipeline   │    │   ChromaDB      │
│  (Python)   │    │  transcripts +  │
│             │    │  book chunks    │
└─────────────┘    └─────────────────┘
```

**Key constraints:**
- User API key travels in request headers only — never persisted server-side
- Jobs identified by UUID, stored in-memory (v1) or SQLite (v1.5)
- ChromaDB runs embedded (no separate service needed)
- Language detection via `langdetect` on raw transcript; prompts auto-switch to "translate + write in English"

---

## Book Length Fix

### Problem
- Grouper over-merges 43 videos into ~8-10 top-level topics
- `write_topic()` summarizes rather than exhausts transcript content
- Result: 43 videos × 45min → 60 pages (should be 400-600)

### Fix 1 — Sub-topic splitter
After grouping, split each top-level topic into 4-6 subtopics. Each subtopic written independently at `min_words: 2000`. 10 topics → 50-60 subtopics → proper book depth.

### Fix 2 — Coverage pass with budget
After writing, check which transcript segments weren't cited. Re-run writer on uncovered segments as additional subsections. Stop when >85% coverage OR token budget is exhausted. Prioritize least-covered topics first.

---

## Token Optimization

| Strategy | Saving |
|---|---|
| RAG-during-writing (retrieve top-K chunks per subtopic instead of full transcript) | ~95% input token reduction per write call |
| Transcript deduplication (cosine similarity, no API cost) | ~20-30% segment reduction |
| Tiered model usage (cheap model for grouping/dedup, user model for writing) | ~40% overall cost reduction |
| Per-subtopic checkpoints (never re-spend on completed work) | prevents waste on failure/resume |
| Hard token budget config | user controls max spend |

**Estimated cost for 43-video playlist:**
| Approach | Est. tokens | Est. cost (Gemini Flash) |
|---|---|---|
| Current | ~2M | ~$0.30 |
| Naive deep expansion | ~80M | ~$12 |
| With RAG-during-writing | ~8M | ~$1.20 |

---

## Multilingual Support

- `langdetect` runs on raw transcript before any LLM call
- Detected language stored in job metadata
- All LLM prompts append: "The source material may be in {language}. Always write output in English."
- No user action required — fully automatic
- Handles: Hindi, Hinglish, Spanish, French, etc.

---

## RAG / Chat Layer

### Indexing (two phases)
1. **During pipeline** — transcript segments embedded into ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` (free, local). Each chunk tagged `{video_id, timestamp, topic, subtopic}`.
2. **After book generation** — chapter text chunked + indexed. Tagged `{chapter, subtopic, page_range}`.

### Query flow
```
User question
  → embed (local model, free)
  → retrieve top-5 chunks (transcripts + book)
  → build prompt: chunks + question
  → LLM call (user's API key)
  → response with source citations
```

### Frontend chat
- Persistent per-job chat history (localStorage)
- Each answer shows source cards (video timestamp links + chapter refs)
- "Ask about this chapter" button per chapter

---

## Next.js Frontend

### Routes
```
/                    → Landing + "Create Book" form
/jobs/[id]           → Job progress + results
/jobs/[id]/chat      → Chat interface
```

### Config form fields
- YouTube Playlist URL
- API Key (model provider)
- Model selector (Gemini Flash / GPT-4o-mini / Claude Haiku)
- Temperature slider (0.1 – 1.0, default 0.3)
- Min words per subtopic (1000 – 5000, default 2000)
- Token budget ($0.50 – $20.00, default $2.00)
- Language: Auto-detect (default)

### Job progress page
- SSE-driven live progress bar per pipeline stage
- Stage labels: Fetching → Transcribing → Deduplicating → Grouping → Writing → Verifying → Rendering
- Live cost tracker ("$0.43 / $2.00 budget used")
- Download PDF + Chat buttons appear on completion

---

## Open Source Packaging

- `docker-compose.yml` — one-command local setup (Next.js + FastAPI + ChromaDB)
- `.env.example` with all keys documented
- GitHub Actions CI (lint + tests on PR)
- `CONTRIBUTING.md` + issue templates
- One-click deploy buttons (Railway, Render)
- Architecture diagram in README (Mermaid)

---

## Standout / Resume Features

- Live public demo at custom domain (e.g. `bookify.dev`)
- Free tier: 100k token limit per generation
- Public metrics on landing page (books generated, questions answered, languages detected)
- "Example books" gallery — 3-4 pre-generated PDFs showing output quality
- Technical blog post: "How I built a YouTube-to-book converter with RAG for $1.20"
- GitHub README with full architecture diagram

**Resume bullet:**
> Built Bookify — open source tool converting YouTube playlists into citation-backed PDF books via a 7-stage LLM pipeline with RAG Q&A. 400-600 pages per 40-hour playlist at ~$1.20 via RAG-during-writing optimization. Next.js frontend, FastAPI backend, ChromaDB vector store.
