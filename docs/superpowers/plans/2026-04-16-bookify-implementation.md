# Bookify Implementation Snapshot (Submission)

## Scope implemented

- Full playlist -> PDF pipeline is operational.
- Checkpoint-driven resumability works across stages.
- Gemini-based LLM stages wired through shared client abstraction.
- PDF renderer includes TOC, citations/footnotes, tables, mermaid diagrams, glossary, and references.

## Implemented architecture

```mermaid
flowchart TD
    A[run.py orchestration] --> B[pipeline/fetcher.py]
    B --> C[pipeline/transcriber.py]
    C --> D[pipeline/terminology_corrector.py]
    D --> E[pipeline/grouper.py]
    E --> F[pipeline/topic_writer.py]
    F --> G[pipeline/citation_verifier.py]
    G --> H[pipeline/prose_polisher.py]
    H --> I[pipeline/assembler.py]
    I --> J[pipeline/pdf_renderer.py]
    J --> K[output/book.pdf]
```

## Data artifacts

```mermaid
flowchart LR
    F1[01_fetch] --> F2[01b_ref_content]
    F2 --> T1[02_transcripts]
    T1 --> T2[02b_corrected]
    T2 --> G[03_groups]
    G --> W[04_topics]
    W --> V[04b_verified]
    V --> P[04c_polished]
    P --> B[05_book]
    B --> PDF[output/book.pdf]
```

## Submission-specific repository rules

- `.claude/` must remain untracked (`.gitignore`).
- `checkpoints/audio/` must be excluded from git and cleaned locally when needed.
- Keep checkpoints for reproducibility of submitted outputs.

## Current config baseline

```yaml
llm:
  provider: gemini
  model: gemini-flash-latest
  temperature: 0.3

pipeline:
  batch_size: 4
  rate_limit_rpm: 6
  min_words_per_topic: 8000
```

## Notes

- Stage 4 writing currently follows configured minimum target, but final generated chapter lengths can still vary by model behavior.
- For deterministic long-form outputs, rerun Stage 4+ with checkpoint reset (`04_*`, `05_book`) and keep model/rate settings fixed.
