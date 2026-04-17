# Bookify Design Spec (Current)

## Goal

Build a reproducible pipeline that turns playlist transcripts into a submission-quality PDF book with verified citations.

## End-to-end flow

```mermaid
flowchart TD
    P[Playlist URL] --> S1[Stage 1: Fetch metadata/audio + reference URLs]
    S1 --> S2[Stage 2: YouTube Transcript API transcripts]
    S2 --> S3[Stage 3: Terminology correction]
    S3 --> S4[Stage 4: Group and order topics]
    S4 --> S5a[Stage 5a: Write chapters]
    S5a --> S5b[Stage 5b: Verify claims/citations]
    S5b --> S5c[Stage 5c: Polish prose]
    S5c --> S6[Stage 6: Assemble book markdown]
    S6 --> S7[Stage 7: Render PDF with WeasyPrint]
    S7 --> OUT[output/book.pdf]
```

## Stage details

1. **Stage 1**: Playlist/video metadata, URL extraction/filtering, reference content snapshotting.
2. **Stage 2**: Transcript collection with checkpoints in `02_transcripts`.
3. **Stage 3**: Technical-term cleanup and transcript normalization.
4. **Stage 4**: Topic clustering and chapter ordering.
5. **Stage 5**: Chapter writing + verification + polish.
6. **Stage 6**: Introduction/conclusion/glossary/references assembly.
7. **Stage 7**: HTML+CSS rendering to final PDF.

## Verification design

```mermaid
flowchart LR
    W[Topic prose draft] --> V[Verifier]
    V -->|supported| K[Keep sentence]
    V -->|partially supported| R[Rewrite conservatively]
    V -->|unsupported| X[Remove + reflow]
    K --> O[Verified topic output]
    R --> O
    X --> O
```

## Checkpoint contract

- Pipeline is resumable by stage (`--from`, `--to`).
- Primary checkpoint folders:
  - `01_fetch`, `01b_ref_content`
  - `02_transcripts`, `02b_corrected`
  - `03_groups`
  - `04_topics`, `04b_verified`, `04c_polished`
  - `05_book`
- Local-only audio cache: `checkpoints/audio` (gitignored).

## Runtime config (current baseline)

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

## Output quality targets

- Coherent chapter flow by topic dependency.
- Citation markers transformed into footnotes during render.
- References grouped by topic.
- Deterministic reruns from checkpoints for submission reproducibility.
