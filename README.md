# Bookify

Convert a YouTube playlist into a structured, eBook-quality PDF using a fully local transcription pipeline and your choice of LLM provider. No manual editing. No hallucinations left unchecked.

---

## What it does

Bookify takes a YouTube playlist URL and produces a print-ready PDF book with:

- Chapters grouped by topic (not one-per-video)
- Educational prose rewritten from transcripts — not a transcript dump
- Numbered superscript citations linked to a Footnotes section
- Mermaid architecture diagrams rendered as figures
- Markdown tables with captions
- Table of Contents, Glossary, Introduction, Conclusion
- References & Resources section with clickable links

All transcription runs **locally** on your machine via `faster-whisper`. Only the LLM stages call an external API.

---

## Pipeline overview

```
YouTube Playlist URL
  │
  ├─ Stage 1 : Fetch audio (yt-dlp + ffmpeg) + extract reference URLs from descriptions
  ├─ Stage 2 : Transcribe audio locally (faster-whisper, CPU, no API cost)
  ├─ Stage 3 : Terminology correction — fix domain jargon mangled by Whisper (LLM)
  ├─ Stage 4 : Group videos into topics + order chapters (LLM)
  ├─ Stage 5 : Write topic prose + verify citations + polish prose (LLM, heaviest stage)
  ├─ Stage 6 : Assemble full book — intro, conclusion, glossary, references (LLM)
  └─ Stage 7 : Render PDF (WeasyPrint, local)
```

Every stage saves a checkpoint to disk. A crash or API timeout mid-run loses nothing — re-run and it picks up exactly where it left off.

---

## System requirements

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.10+ | Runtime | [python.org](https://python.org) |
| ffmpeg | Audio extraction | `brew install ffmpeg` / `apt install ffmpeg` |
| Pango + Cairo | WeasyPrint PDF rendering | `brew install pango cairo` (macOS) |
| Mermaid CLI (`mmdc`) | Render architecture diagrams | `npm install -g @mermaid-js/mermaid-cli` |

---

## Installation

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/Bookify.git
cd Bookify
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your API key**

Create a `.env` file in the project root (it is gitignored):

```bash
# .env

# Pick ONE provider and fill in its key:
GEMINI_API_KEY=your-gemini-key-here
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

Bookify loads `.env` automatically at startup via `python-dotenv`.

---

## Configuration

Edit `config.yaml` to select your LLM provider and model:

```yaml
llm:
  provider: gemini            # options: gemini | openai | anthropic
  model: gemini-2.5-flash     # gemini-2.5-flash | gpt-4o | claude-opus-4-7
  temperature: 0.3

pipeline:
  batch_size: 4               # parallel videos per mini-batch (reduce if hitting rate limits)
  whisper_model: small        # tiny | base | small | medium | large-v3
  whisper_compute_type: int8  # int8 (fast, low RAM) | float16 | float32

paths:
  checkpoints: checkpoints    # all intermediate stage outputs saved here
  output: output              # final PDF written here
  audio_temp: checkpoints/audio
```

**Whisper model trade-off:**

| Model | RAM | Speed | Accuracy |
|---|---|---|---|
| `tiny` | ~400 MB | Fastest | Basic |
| `small` | ~1 GB | Fast | Good |
| `medium` | ~3 GB | Moderate | Better |
| `large-v3` | ~6 GB | Slow | Best |

For a machine with 8 GB RAM, `small` or `medium` is recommended.

---

## Running the pipeline

### Full run (all stages)

```bash
# macOS — WeasyPrint requires this library path
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"

# Linux
python run.py --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

Output is written to `output/book.pdf`.

### Test with a single video first

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" \
  --video-id VIDEO_ID_HERE
```

### Limit to first N videos (cheaper test run)

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" \
  --limit 5
```

### Transcription only (zero API cost)

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" \
  --from 1 --to 2
```

### Resume from a specific stage

If a run stopped partway through, resume without re-running earlier stages:

```bash
# Resume from Stage 3 onward (skips fetch + transcribe)
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID" \
  --from 3

# Re-render the PDF only (no LLM calls)
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist placeholder --from 7
```

### Run specific stages only

```bash
# Run only Stage 5 (write + verify + polish) through Stage 6 (assemble)
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "..." --from 5 --to 6
```

### Full CLI reference

```
python run.py [options]

Options:
  --playlist URL     YouTube playlist URL (required for Stage 1)
  --from N           Start from stage N (default: 1)
  --to N             Stop after stage N (default: 7, run all)
  --limit N          Process only the first N videos
  --video-id ID      Process only this specific video ID
  --title "..."      Book title (default: "Building LLMs from Scratch")
  --config PATH      Path to config file (default: config.yaml)
```

**Stage map:**

| Stage | Name | LLM? | Description |
|---|---|---|---|
| 1 | Fetch | Minimal | Download audio + extract reference URLs |
| 2 | Transcribe | No | Whisper runs locally |
| 3 | Terminology | Yes | Fix domain jargon in transcripts |
| 4 | Group & Order | Yes | Cluster videos into chapters |
| 5 | Write + Verify + Polish | Yes | Core prose generation (most tokens) |
| 6 | Assemble | Yes | Intro, conclusion, glossary, references |
| 7 | Render | No | WeasyPrint → PDF |

---

## Checkpoint system

All intermediate outputs are saved under `checkpoints/`:

```
checkpoints/
  01_fetch/          # video metadata + audio paths
  01b_ref_content/   # scraped reference URL content
  02_transcripts/    # raw Whisper transcripts
  02b_corrected/     # terminology-corrected transcripts
  03_groups/         # topic groups + chapter ordering
  04_topics/         # LLM-written prose per topic
  04b_verified/      # citation-verified prose
  04c_polished/      # final polished prose
  05_book/           # assembled full book markdown
```

To force a stage to re-run, delete its checkpoint directory:

```bash
# Re-run prose writing for all topics
rm -rf checkpoints/04_topics checkpoints/04b_verified checkpoints/04c_polished checkpoints/05_book

# Re-run for a single topic slug
rm checkpoints/04_topics/my-topic-slug.json
```

---

## Supported LLM providers

| Provider | `provider` value | Recommended model | Key env var |
|---|---|---|---|
| Google Gemini | `gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `claude-opus-4-7` | `ANTHROPIC_API_KEY` |

Switch providers by editing `config.yaml` — no code changes needed.

---

## Running tests

```bash
# macOS
DYLD_LIBRARY_PATH=/opt/homebrew/lib python -m pytest tests/ -v

# Linux
python -m pytest tests/ -v
```

All 49 tests run offline (no real API calls, no audio files needed).

---

## Project structure

```
Bookify/
├── run.py                        # Main entrypoint + CLI
├── config.yaml                   # Provider, model, paths config
├── requirements.txt
├── .env                          # API keys (gitignored)
│
├── pipeline/
│   ├── fetcher.py                # Stage 1 — yt-dlp audio + URL extraction
│   ├── transcriber.py            # Stage 2 — faster-whisper transcription
│   ├── terminology_corrector.py  # Stage 3 — LLM jargon correction
│   ├── grouper.py                # Stage 4 — LLM topic clustering
│   ├── topic_writer.py           # Stage 5a — LLM prose writing
│   ├── citation_verifier.py      # Stage 5b — LLM citation verification
│   ├── prose_polisher.py         # Stage 5c — LLM prose polish
│   ├── assembler.py              # Stage 6 — book assembly
│   └── pdf_renderer.py           # Stage 7 — WeasyPrint PDF rendering
│
├── llm/
│   └── client.py                 # Unified LLM client (Gemini / OpenAI / Anthropic)
│
├── utils/
│   ├── checkpoint.py             # Save / load / list checkpoint files
│   ├── progress.py               # Rich live progress display
│   ├── quality_report.py         # Post-run stats report
│   └── url_filter.py             # LLM-based reference URL filter
│
└── tests/                        # 49 unit tests, fully offline
```

---

## Cost estimate

For a 43-video playlist using Gemini 2.0 Flash ($0.10/1M input, $0.40/1M output, no thinking tokens):

| Stage | Typical tokens | Estimated cost |
|---|---|---|
| Stage 3 — Terminology | ~150k input | ~$0.02 |
| Stage 4 — Grouping | ~50k input | ~$0.01 |
| Stage 5 — Write + Verify + Polish | ~2–4M input | ~$0.40–0.80 |
| Stage 6 — Assemble | ~300k input | ~$0.05 |
| **Total** | | **~$0.50–$0.90** |

Transcription (Stage 2) costs **$0** — runs entirely on your local CPU.

---

## Troubleshooting

**WeasyPrint fails with `cannot load library 'libpango'` on macOS**

```bash
brew install pango cairo
# Then always run with:
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py ...
```

**Mermaid diagrams render as fallback text boxes**

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc --version   # verify it's on PATH
```

**Gemini 503 / rate limit errors**

The LLM client retries automatically with exponential backoff (up to 5 minutes). For persistent rate limits, reduce `batch_size` in `config.yaml` or switch to a different provider.

**Resuming after a crash**

Just re-run with `--from N` where N is the stage that failed. All earlier stages are checkpointed and will be skipped automatically.

---

## License

MIT
