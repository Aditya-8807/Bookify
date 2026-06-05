# Graph Report - .  (2026-04-21)

## Corpus Check
- 39 files · ~44,706 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 300 nodes · 538 edges · 24 communities detected
- Extraction: 65% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 186 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Pipeline Checkpoint Data|Pipeline Checkpoint Data]]
- [[_COMMUNITY_Checkpoint Utilities|Checkpoint Utilities]]
- [[_COMMUNITY_Project Docs & Structure|Project Docs & Structure]]
- [[_COMMUNITY_PDF Renderer|PDF Renderer]]
- [[_COMMUNITY_Assembler & Fetcher|Assembler & Fetcher]]
- [[_COMMUNITY_LLM Client|LLM Client]]
- [[_COMMUNITY_URL Filter Tests|URL Filter Tests]]
- [[_COMMUNITY_Deep Topic Pipeline|Deep Topic Pipeline]]
- [[_COMMUNITY_Transcriber & Tests|Transcriber & Tests]]
- [[_COMMUNITY_Terminology Corrector|Terminology Corrector]]
- [[_COMMUNITY_Topic Writer & Tests|Topic Writer & Tests]]
- [[_COMMUNITY_Citation Verifier|Citation Verifier]]
- [[_COMMUNITY_Data Models|Data Models]]
- [[_COMMUNITY_Config & LLM Setup|Config & LLM Setup]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_LLM Init|LLM Init]]
- [[_COMMUNITY_Utils Init|Utils Init]]
- [[_COMMUNITY_Requests Dependency|Requests Dependency]]
- [[_COMMUNITY_Markdown Dependency|Markdown Dependency]]
- [[_COMMUNITY_Pytest Dependency|Pytest Dependency]]
- [[_COMMUNITY_Checkpoint Util Ref|Checkpoint Util Ref]]
- [[_COMMUNITY_Progress Util Ref|Progress Util Ref]]
- [[_COMMUNITY_Quality Report Ref|Quality Report Ref]]
- [[_COMMUNITY_URL Filter Util Ref|URL Filter Util Ref]]

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
- `assemble_book()` --calls--> `test_assemble_book_includes_all_sections()`  [INFERRED]
  pipeline/assembler.py → tests/test_assembler.py
- `main()` --calls--> `client_from_config()`  [INFERRED]
  run.py → llm/client.py
- `main()` --calls--> `live()`  [INFERRED]
  run.py → utils/progress.py
- `main()` --calls--> `fetch_and_checkpoint_ref_content()`  [INFERRED]
  run.py → pipeline/grouper.py

## Communities

### Community 0 - "Pipeline Checkpoint Data"
Cohesion: 0.07
Nodes (36): Checkpoint: 01_fetch, Checkpoint: 01b_ref_content, Checkpoint: 02_transcripts, Checkpoint: 02b_corrected, Checkpoint: 03_groups, Checkpoint: 04_topics, Checkpoint: 04b_verified, Checkpoint: 04c_polished (+28 more)

### Community 1 - "Checkpoint Utilities"
Cohesion: 0.12
Nodes (31): checkpoint_exists(), list_checkpoints(), load_checkpoint(), _path(), save_checkpoint(), Checkpoint Contract Design, fetch_video_audio(), fetch_and_checkpoint_ref_content() (+23 more)

### Community 2 - "Project Docs & Structure"
Cohesion: 0.07
Nodes (36): Graphify Knowledge Graph Integration, Bookify Design Spec, Bookify Implementation Plan, pipeline/assembler.py, Bookify, Checkpoints Directory, pipeline/citation_verifier.py, Deep Stage 1: Split into 6 Subtopics (+28 more)

### Community 3 - "PDF Renderer"
Cohesion: 0.09
Nodes (29): _build_compact_toc_html(), _inject_captions_into_html(), _inject_table_captions(), _latex_to_text(), markdown_to_html(), _normalize_markdown_blocks(), _process_citations_all_html(), _process_citations_in_html() (+21 more)

### Community 4 - "Assembler & Fetcher"
Cohesion: 0.11
Nodes (20): assemble_book(), generate_glossary(), extract_playlist_videos(), fetch_all(), _playlist_url(), Extract pure playlist URL from any YouTube URL format., Fetch all video metadata from a playlist without downloading., render_pdf() (+12 more)

### Community 5 - "LLM Client"
Cohesion: 0.19
Nodes (12): _clean_json(), client_from_config(), _enforce_rate_limit(), get_cost_summary(), _is_retryable(), LLMClient, test_complete_json_returns_dict(), test_complete_json_strips_markdown_fences() (+4 more)

### Community 6 - "URL Filter Tests"
Cohesion: 0.2
Nodes (16): test_blacklist_keeps_arxiv_github_huggingface(), test_blacklist_removes_discord(), test_blacklist_removes_patreon(), test_blacklist_removes_twitter_and_x(), test_blacklist_removes_youtube_selflinks(), test_extract_urls_basic(), test_extract_urls_with_context(), test_filter_description_urls_end_to_end() (+8 more)

### Community 7 - "Deep Topic Pipeline"
Cohesion: 0.28
Nodes (15): assemble_deep_book(), _build_refs_block(), _build_transcript_block(), _ck_base(), _fmt_ts(), _load_corrected_transcripts(), main(), _parse_verify_response() (+7 more)

### Community 8 - "Transcriber & Tests"
Cohesion: 0.36
Nodes (11): _patch_api(), _snippet(), test_merge_segments_groups_within_window(), test_transcribe_all_full_text_joins_segments(), test_transcribe_all_multiple_videos(), test_transcribe_all_produces_segments(), test_transcribe_all_saves_checkpoint(), test_transcribe_all_skips_checkpointed_videos() (+3 more)

### Community 9 - "Terminology Corrector"
Cohesion: 0.24
Nodes (11): _apply_corrections(), _correct_chunk(), correct_transcript(), _fmt_ts(), Apply correction substitutions to segment text using word-boundary matching., Return corrections list for one chunk of segments., test_apply_corrections_word_boundary(), test_correct_transcript_fixes_misheard_terms() (+3 more)

### Community 10 - "Topic Writer & Tests"
Cohesion: 0.31
Nodes (9): test_detect_overlaps_finds_shared_concepts(), test_write_topic_calls_llm_with_transcript(), test_write_topic_retries_when_draft_too_short(), test_write_topic_skips_if_checkpoint_exists(), detect_overlaps(), _fmt_ts(), _word_count(), write_all_topics() (+1 more)

### Community 11 - "Citation Verifier"
Cohesion: 0.36
Nodes (8): verify_all_topics(), verify_topic(), _prose_response(), Verify source texts are included in the LLM prompt., test_verify_all_topics_sequential(), test_verify_topic_single_pass(), test_verify_topic_skips_if_checkpoint_exists(), test_verify_topic_uses_sources()

### Community 12 - "Data Models"
Cohesion: 0.39
Nodes (8): Citation, CorrectedTranscript, TopicGroup, Transcript, TranscriptSegment, VerifiedTopic, VideoMeta, TypedDict

### Community 13 - "Config & LLM Setup"
Cohesion: 0.29
Nodes (7): Runtime Config Baseline, config.yaml, Gemini Flash LLM Provider, llm/client.py, run.py orchestrator, python-dotenv dependency, pyyaml dependency

### Community 14 - "Test Fixtures"
Cohesion: 0.4
Nodes (2): Temporary checkpoints directory for tests., tmp_checkpoints()

### Community 15 - "LLM Init"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Utils Init"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Requests Dependency"
Cohesion: 1.0
Nodes (1): requests dependency

### Community 18 - "Markdown Dependency"
Cohesion: 1.0
Nodes (1): markdown dependency

### Community 19 - "Pytest Dependency"
Cohesion: 1.0
Nodes (1): pytest dependency

### Community 20 - "Checkpoint Util Ref"
Cohesion: 1.0
Nodes (1): utils/checkpoint.py

### Community 21 - "Progress Util Ref"
Cohesion: 1.0
Nodes (1): utils/progress.py

### Community 22 - "Quality Report Ref"
Cohesion: 1.0
Nodes (1): utils/quality_report.py

### Community 23 - "URL Filter Util Ref"
Cohesion: 1.0
Nodes (1): utils/url_filter.py

## Knowledge Gaps
- **55 isolated node(s):** `Replace citation markers with superscript footnote numbers.     Returns (process`, `Apply one global citation index and append a single footnotes block.`, `Auto-quote node labels that contain parentheses or arrow-like chars,     which M`, `Convert a LaTeX math expression to a readable plain-text approximation.`, `Replace $$...$$ display math and $...$ inline math with readable text.     Weasy` (+50 more)
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
- **Thin community `Checkpoint Util Ref`** (1 nodes): `utils/checkpoint.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Progress Util Ref`** (1 nodes): `utils/progress.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Quality Report Ref`** (1 nodes): `utils/quality_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `URL Filter Util Ref`** (1 nodes): `utils/url_filter.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Assembler & Fetcher` to `Pipeline Checkpoint Data`, `Checkpoint Utilities`, `PDF Renderer`, `LLM Client`, `Transcriber & Tests`, `Topic Writer & Tests`, `Citation Verifier`?**
  _High betweenness centrality (0.198) - this node is a cross-community bridge._
- **Why does `markdown_to_html()` connect `PDF Renderer` to `Assembler & Fetcher`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `utils/checkpoint.py` connect `Checkpoint Utilities` to `Pipeline Checkpoint Data`, `Assembler & Fetcher`, `Transcriber & Tests`, `Terminology Corrector`, `Topic Writer & Tests`, `Citation Verifier`, `Config & LLM Setup`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `save_checkpoint()` (e.g. with `polish_topic()` and `_fetch_transcript()`) actually correct?**
  _`save_checkpoint()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `main()` (e.g. with `client_from_config()` and `PipelineProgress`) actually correct?**
  _`main()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `checkpoint_exists()` (e.g. with `polish_topic()` and `_fetch_transcript()`) actually correct?**
  _`checkpoint_exists()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `markdown_to_html()` (e.g. with `main()` and `test_markdown_to_html_converts_headings()`) actually correct?**
  _`markdown_to_html()` has 7 INFERRED edges - model-reasoned connections that need verification._