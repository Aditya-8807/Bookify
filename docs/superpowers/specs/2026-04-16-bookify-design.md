# Bookify — Design Spec
**Date:** 2026-04-16  
**Version:** V1.1  
**Goal:** Convert a YouTube playlist into an eBook-quality PDF using a reproducible, hallucination-free LLM pipeline.

---

## 1. Overview

Single command input:
```bash
python run.py --playlist "https://youtube.com/watch?v=...&list=..."
```

Single output: `output/book.pdf` — a clean, professional, fully-cited technical book.

The pipeline is checkpointed at every stage. Any stage can be re-run independently without re-executing prior stages. A live Rich CLI progress display shows real-time status across all stages.

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
Glossary                   ← auto-generated technical terms + definitions
References & Resources     ← all verified source URLs, grouped by topic
```

Topics are named by subject only (e.g. "Tokenization & Vocabulary", "The Attention Mechanism"). No "Chapter 1, Chapter 2" numbering.

**Writing style rules (enforced via LLM system prompt):**
- Clear educational prose — no bullet dumps, no transcript paraphrasing
- Each topic section: opening context → concept explanation → worked examples → summary
- Code snippets preserved verbatim from transcript where instructor wrote/showed code
- Consistent terminology throughout (enforced by terminology correction pass)
- No first-person instructor voice ("In this video I will..." stripped)
- Smooth transitions between sections within a topic
- Cross-references used for duplicate concepts (e.g. "As introduced in Embeddings…")

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
│  faster-whisper large-v3 → raw transcript + timestamps│
│  Audio files deleted after transcription             │
│  Checkpoint: checkpoints/02_transcripts/<id>.json    │
└──────────────────────────┬──────────────────────────┘
                           │ parallel per video
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2b: TERMINOLOGY CORRECTION                     │
│  LLM reads transcript + video title as context       │
│  Fixes domain misheard terms:                        │
│    "a tension head" → "attention head"               │
│    "cave cache"     → "KV cache"                     │
│    "soft max"       → "softmax"                      │
│  Corrections logged for auditability                 │
│  Checkpoint: checkpoints/02b_corrected/<id>.json     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: GROUP + ORDER + ENRICH                      │
│  LLM reads all corrected transcripts + titles        │
│  → clusters videos into thematic topic groups        │
│  → builds concept dependency graph                   │
│     (e.g. Embeddings → Attention → Transformer)      │
│  → orders topics by dependency, not playlist order   │
│  → pools + deduplicates ref URLs per topic group     │
│  → fetches ref content (best-effort, skips failures) │
│  Checkpoint: checkpoints/03_groups.json              │
└──────────────────────────┬──────────────────────────┘
                           │ parallel per topic group
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: DEDUP + WRITE TOPICS                        │
│  Cross-topic deduplication pass:                     │
│    Detect overlapping concepts across groups         │
│    FIRST occurrence → full explanation               │
│    SUBSEQUENT occurrences → cross-reference + brief  │
│      recall ("As covered in <Topic>…")               │
│  Per topic: corrected transcript chunks + pooled refs│
│  LLM writes coherent prose (synthesizes, not copies) │
│  Timestamp citations injected during writing:        │
│    [Video: "Attention Mechanism" @ 12:34]            │
│  Checkpoint: checkpoints/04_topics/<topic>.md        │
└──────────────────────────┬──────────────────────────┘
                           │ parallel per topic
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4b: CITATION VERIFIER                          │
│  Step 1: Extract all factual/technical claims        │
│  Step 2: Retrieve best matching source passage       │
│          (transcript chunk with timestamp, or ref)   │
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
│  Applies concept dependency order to final sequence  │
│  Generates glossary from all defined technical terms │
│  Collects all citations → References section         │
│  Checkpoint: checkpoints/05_book.md                  │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 6: RENDER PDF                                  │
│  WeasyPrint → clean professional PDF                 │
│  Title page, auto TOC with page numbers              │
│  Inline timestamp citations                          │
│  Glossary before References                          │
│  References & Resources at end, grouped by topic     │
│  Output: output/book.pdf                             │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│ Stage 7: QUALITY REPORT                              │
│  Printed to stdout after PDF generation:             │
│  Topics: N  |  Words: N  |  Citations: N             │
│  Verification pass rate: N%                          │
│  Claims rewritten: N  |  Claims removed: N           │
│  Topics without instructor refs: N                   │
│  Terminology corrections made: N                     │
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

## 5. Terminology Correction Pass (Stage 2b)

faster-whisper is highly accurate but occasionally mishears domain-specific terms. A lightweight LLM pass immediately after transcription fixes these using the video title as domain context.

- Corrections are logged with before/after pairs for auditability
- Runs in parallel per video (same mini-batch as transcription)
- Does not alter sentence structure, only fixes misheard technical tokens

---

## 6. Concept Dependency Ordering (Stage 3)

After grouping videos into topic clusters, the LLM builds a directed dependency graph:

```
Tokenization → Embeddings → Attention → Transformer → Training Loop → Fine-tuning
```

Topics are ordered in the book by this graph (topological sort), not by playlist order. This ensures the reader always has prerequisite knowledge before encountering a new concept. Circular dependencies (rare) are broken by the LLM choosing the more foundational topic first.

---

## 7. Cross-Topic Deduplication (Stage 4)

The same concept often appears across multiple videos. Rather than repeating content:

- **First occurrence** → full explanation in the topic where it fits most naturally
- **Subsequent occurrences** → one-sentence recall + cross-reference: *"As introduced in Embeddings & Vector Representations, a token is…"*

This prevents repetition while preserving readability for readers who jump to a specific topic.

---

## 8. Timestamp-Based Citations

Transcript citations include video title and timestamp so any claim is directly verifiable:

```
[Video: "The Attention Mechanism" @ 14:22]
[Video: "Training from Scratch" @ 03:07]
```

These appear inline in the prose and are also collected in the References & Resources section alongside external URL references.

---

## 9. Glossary

Auto-generated at assembly time (Stage 5). The LLM extracts every technical term that is explicitly defined or explained anywhere in the book, and writes a one-sentence definition for each. Terms are sourced only from the verified prose — no external definitions introduced.

Placed between Conclusion and References & Resources.

---

## 10. Reference Enrichment (Stage 3)

References are collected at the **topic group level**, not per-video:
- When videos are grouped into a topic, all reference URLs from every video in that group are pooled and deduplicated
- Fetched once, stored as reference content for the whole topic
- If a video has no description URLs → topic still proceeds on transcript alone
- Fetch failures (paywalled, dead links) → logged, skipped, pipeline continues

---

## 11. LLM Client Abstraction

Thin `LLMClient` class in `llm/client.py`. Provider selected via `config.yaml`:

```yaml
llm:
  provider: openai       # or: anthropic
  model: gpt-4o          # or: claude-opus-4-6
  temperature: 0.3

pipeline:
  batch_size: 4          # parallel videos per mini-batch
```

All pipeline stages import `LLMClient` — never call provider SDKs directly. Swapping provider requires only a config change.

---

## 12. Anti-Hallucination Strategy

Hallucinations are prevented structurally, not just via prompting:

1. **Terminology correction** — domain terms fixed before they enter the LLM writing stage
2. **Grounded context only** — LLM is always given corrected transcript chunks + reference content as the sole input
3. **Low temperature** (0.3) — reduces creative drift
4. **Timestamp citations** — every transcript-sourced claim cites video title + timestamp
5. **Citation Verifier loop** — every factual claim is traced to a source passage before the text is finalized. Unverifiable claims are rewritten or removed, never kept
6. **No floating claims** — final PDF contains only prose that passed the verifier

---

## 13. Rich CLI Progress Display

Using the `rich` library, the terminal shows a live multi-panel display during pipeline execution:

```
Bookify Pipeline
─────────────────────────────────────────────────────
 Stage 1: Fetch          [████████████████████] 20/20 videos
 Stage 2: Transcribe     [████████████░░░░░░░░] 12/20 videos
 Stage 2b: Terminology   [████████░░░░░░░░░░░░]  8/20 videos
 Stage 3: Group + Order  [ waiting... ]
 ...
─────────────────────────────────────────────────────
 Elapsed: 4m 22s
```

Each stage shows its own progress bar. Parallel stages update concurrently.

---

## 14. Checkpointing & Resumption

Every stage writes output to `checkpoints/`. On re-run, completed stages are skipped automatically.

```bash
python run.py --playlist <url>           # full run
python run.py --playlist <url> --from 4  # resume from topic writing
python run.py --playlist <url> --from 6  # re-render PDF only
```

`checkpoints/` is gitignored. `output/` is gitignored (large binary).

---

## 15. Project Structure

```
Bookify/
├── config.yaml
├── run.py
├── pipeline/
│   ├── fetcher.py              # Stage 1: fetch + URL filter
│   ├── transcriber.py          # Stage 2: faster-whisper
│   ├── terminology_corrector.py# Stage 2b: LLM terminology fix
│   ├── grouper.py              # Stage 3: group + dependency order + enrich
│   ├── topic_writer.py         # Stage 4: dedup + write prose
│   ├── citation_verifier.py    # Stage 4b: verify + rewrite loop
│   ├── assembler.py            # Stage 5: assemble + glossary
│   └── pdf_renderer.py         # Stage 6: WeasyPrint PDF
├── llm/
│   └── client.py               # Pluggable LLM abstraction
├── utils/
│   ├── url_filter.py           # 2-pass URL filter
│   ├── checkpoint.py           # Stage checkpoint read/write
│   └── progress.py             # Rich CLI progress display
├── checkpoints/                # gitignored
├── output/                     # gitignored
├── requirements.txt
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-16-bookify-design.md
```

---

## 16. V2 Scope (Out of Scope for V1)

Auto-discovery of external references for topic groups with no instructor-provided refs:
- arXiv search (free Python library)
- Wikipedia API
- HuggingFace docs
- Papers with Code API

V2 adds this as an additional enrichment layer in Stage 3, triggered only when a topic group has zero instructor refs.

---

## 17. Non-Goals (V1)

- EPUB or HTML output (PDF only)
- Interactive web UI
- Multi-playlist support

---

## 18. V3 Scope (Post V2, Low Priority)

- Cloud deployment — host the pipeline as a web service so users can submit a playlist URL and receive a PDF without running locally
