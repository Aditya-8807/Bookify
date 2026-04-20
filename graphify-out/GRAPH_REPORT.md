# Graph Report - .  (2026-04-19)

## Corpus Check
- Corpus is ~23,190 words - fits in a single context window. You may not need a graph.

## Summary
- 247 nodes · 464 edges · 19 communities detected
- Extraction: 64% EXTRACTED · 36% INFERRED · 0% AMBIGUOUS · INFERRED: 168 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Checkpoint System|Checkpoint System]]
- [[_COMMUNITY_Book Assembly & Fetching|Book Assembly & Fetching]]
- [[_COMMUNITY_PDF Rendering|PDF Rendering]]
- [[_COMMUNITY_Pipeline Architecture & Design|Pipeline Architecture & Design]]
- [[_COMMUNITY_LLM Client|LLM Client]]
- [[_COMMUNITY_URL Filter Tests|URL Filter Tests]]
- [[_COMMUNITY_Transcriber Module|Transcriber Module]]
- [[_COMMUNITY_Terminology Corrector|Terminology Corrector]]
- [[_COMMUNITY_Topic Writer|Topic Writer]]
- [[_COMMUNITY_Checkpoint Data Store|Checkpoint Data Store]]
- [[_COMMUNITY_Citation Verifier|Citation Verifier]]
- [[_COMMUNITY_Data Models|Data Models]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_LLM Init|LLM Init]]
- [[_COMMUNITY_Utils Init|Utils Init]]
- [[_COMMUNITY_Requests Dependency|Requests Dependency]]
- [[_COMMUNITY_Markdown Dependency|Markdown Dependency]]
- [[_COMMUNITY_Pytest Dependency|Pytest Dependency]]

## God Nodes (most connected - your core abstractions)
1. `save_checkpoint()` - 23 edges
2. `main()` - 21 edges
3. `checkpoint_exists()` - 17 edges
4. `markdown_to_html()` - 14 edges
5. `LLMClient` - 13 edges
6. `load_checkpoint()` - 13 edges
7. `utils/checkpoint.py` - 13 edges
8. `correct_transcript()` - 11 edges
9. `group_and_order()` - 11 edges
10. `write_topic()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `get_cost_summary()`  [INFERRED]
  run.py → llm/client.py
- `test_extract_playlist_videos()` --calls--> `extract_playlist_videos()`  [INFERRED]
  tests/test_fetcher.py → pipeline/fetcher.py
- `main()` --calls--> `client_from_config()`  [INFERRED]
  run.py → llm/client.py
- `main()` --calls--> `live()`  [INFERRED]
  run.py → utils/progress.py
- `main()` --calls--> `fetch_and_checkpoint_ref_content()`  [INFERRED]
  run.py → pipeline/grouper.py

## Hyperedges (group relationships)
- **LLM-driven Pipeline Stages** — llm_client, pipeline_topic_writer, pipeline_citation_verifier, pipeline_prose_polisher [INFERRED 0.88]
- **Checkpoint-driven Resumability System** — utils_checkpoint, readme_runpy, checkpoint_04_topics [EXTRACTED 0.92]
- **PDF Rendering Stack** — pipeline_pdf_renderer, requirements_weasyprint, output_book_pdf [EXTRACTED 0.95]

## Communities

### Community 0 - "Checkpoint System"
Cohesion: 0.11
Nodes (32): checkpoint_exists(), list_checkpoints(), load_checkpoint(), _path(), save_checkpoint(), Checkpoint Contract Design, fetch_video_audio(), fetch_and_checkpoint_ref_content() (+24 more)

### Community 1 - "Book Assembly & Fetching"
Cohesion: 0.1
Nodes (22): assemble_book(), generate_glossary(), extract_playlist_videos(), fetch_all(), _playlist_url(), Extract pure playlist URL from any YouTube URL format., Fetch all video metadata from a playlist without downloading., render_pdf() (+14 more)

### Community 2 - "PDF Rendering"
Cohesion: 0.09
Nodes (29): _build_compact_toc_html(), _inject_captions_into_html(), _inject_table_captions(), _latex_to_text(), markdown_to_html(), _normalize_markdown_blocks(), _process_citations_all_html(), _process_citations_in_html() (+21 more)

### Community 3 - "Pipeline Architecture & Design"
Cohesion: 0.11
Nodes (27): Citation Verification Design, Runtime Config Baseline, Gemini-based LLM Client Abstraction, llm/client.py, pipeline/assembler.py, pipeline/citation_verifier.py, pipeline/fetcher.py, pipeline/grouper.py (+19 more)

### Community 4 - "LLM Client"
Cohesion: 0.19
Nodes (12): _clean_json(), client_from_config(), _enforce_rate_limit(), get_cost_summary(), _is_retryable(), LLMClient, test_complete_json_returns_dict(), test_complete_json_strips_markdown_fences() (+4 more)

### Community 5 - "URL Filter Tests"
Cohesion: 0.2
Nodes (16): test_blacklist_keeps_arxiv_github_huggingface(), test_blacklist_removes_discord(), test_blacklist_removes_patreon(), test_blacklist_removes_twitter_and_x(), test_blacklist_removes_youtube_selflinks(), test_extract_urls_basic(), test_extract_urls_with_context(), test_filter_description_urls_end_to_end() (+8 more)

### Community 6 - "Transcriber Module"
Cohesion: 0.36
Nodes (11): _patch_api(), _snippet(), test_merge_segments_groups_within_window(), test_transcribe_all_full_text_joins_segments(), test_transcribe_all_multiple_videos(), test_transcribe_all_produces_segments(), test_transcribe_all_saves_checkpoint(), test_transcribe_all_skips_checkpointed_videos() (+3 more)

### Community 7 - "Terminology Corrector"
Cohesion: 0.24
Nodes (11): _apply_corrections(), _correct_chunk(), correct_transcript(), _fmt_ts(), Apply correction substitutions to segment text using word-boundary matching., Return corrections list for one chunk of segments., test_apply_corrections_word_boundary(), test_correct_transcript_fixes_misheard_terms() (+3 more)

### Community 8 - "Topic Writer"
Cohesion: 0.31
Nodes (9): test_detect_overlaps_finds_shared_concepts(), test_write_topic_calls_llm_with_transcript(), test_write_topic_retries_when_draft_too_short(), test_write_topic_skips_if_checkpoint_exists(), detect_overlaps(), _fmt_ts(), _word_count(), write_all_topics() (+1 more)

### Community 9 - "Checkpoint Data Store"
Cohesion: 0.18
Nodes (11): Checkpoint: 01_fetch, Checkpoint: 01b_ref_content, Checkpoint: 02_transcripts, Checkpoint: 02b_corrected, Checkpoint: 03_groups, Checkpoint: 04_topics, Checkpoint: 04b_verified, Checkpoint: 04c_polished (+3 more)

### Community 10 - "Citation Verifier"
Cohesion: 0.36
Nodes (8): verify_all_topics(), verify_topic(), _prose_response(), Verify source texts are included in the LLM prompt., test_verify_all_topics_sequential(), test_verify_topic_single_pass(), test_verify_topic_skips_if_checkpoint_exists(), test_verify_topic_uses_sources()

### Community 11 - "Data Models"
Cohesion: 0.39
Nodes (8): Citation, CorrectedTranscript, TopicGroup, Transcript, TranscriptSegment, VerifiedTopic, VideoMeta, TypedDict

### Community 12 - "Test Fixtures"
Cohesion: 0.4
Nodes (2): Temporary checkpoints directory for tests., tmp_checkpoints()

### Community 13 - "Project Documentation"
Cohesion: 0.4
Nodes (5): Graphify Knowledge Graph Integration, Bookify Design Spec, Bookify Implementation Plan, Bookify Project, Bookify Pipeline

### Community 14 - "LLM Init"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Utils Init"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Requests Dependency"
Cohesion: 1.0
Nodes (1): requests dependency

### Community 17 - "Markdown Dependency"
Cohesion: 1.0
Nodes (1): markdown dependency

### Community 18 - "Pytest Dependency"
Cohesion: 1.0
Nodes (1): pytest dependency

## Knowledge Gaps
- **42 isolated node(s):** `Replace citation markers with superscript footnote numbers.     Returns (process`, `Apply one global citation index and append a single footnotes block.`, `Auto-quote node labels that contain parentheses or arrow-like chars,     which M`, `Convert a LaTeX math expression to a readable plain-text approximation.`, `Replace $$...$$ display math and $...$ inline math with readable text.     Weasy` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `LLM Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Utils Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Requests Dependency`** (1 nodes): `requests dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Markdown Dependency`** (1 nodes): `markdown dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pytest Dependency`** (1 nodes): `pytest dependency`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Book Assembly & Fetching` to `Checkpoint System`, `PDF Rendering`, `Pipeline Architecture & Design`, `LLM Client`, `Transcriber Module`, `Topic Writer`, `Citation Verifier`?**
  _High betweenness centrality (0.290) - this node is a cross-community bridge._
- **Why does `markdown_to_html()` connect `PDF Rendering` to `Book Assembly & Fetching`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Why does `utils/checkpoint.py` connect `Checkpoint System` to `Book Assembly & Fetching`, `Pipeline Architecture & Design`, `Transcriber Module`, `Terminology Corrector`, `Topic Writer`, `Citation Verifier`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `save_checkpoint()` (e.g. with `polish_topic()` and `_fetch_transcript()`) actually correct?**
  _`save_checkpoint()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `main()` (e.g. with `client_from_config()` and `PipelineProgress`) actually correct?**
  _`main()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `checkpoint_exists()` (e.g. with `polish_topic()` and `_fetch_transcript()`) actually correct?**
  _`checkpoint_exists()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `markdown_to_html()` (e.g. with `main()` and `test_markdown_to_html_converts_headings()`) actually correct?**
  _`markdown_to_html()` has 7 INFERRED edges - model-reasoned connections that need verification._