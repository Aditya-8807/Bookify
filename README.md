# Bookify

Convert a YouTube playlist into an eBook-quality PDF using a reproducible, hallucination-free LLM pipeline.

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# macOS: WeasyPrint needs Homebrew system libraries
brew install pango cairo

# Set your LLM API key (OpenAI or Anthropic)
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

# Update config.yaml to match your provider, then run:
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py \
  --playlist "https://youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu"
```

Output: `output/book.pdf`

## Pipeline

```
playlist URL
  → Stage 1:  Fetch audio + extract reference URLs
  → Stage 2:  Transcribe with faster-whisper large-v3
  → Stage 2b: LLM terminology correction pass
  → Stage 3:  LLM topic clustering + dependency ordering + reference enrichment
  → Stage 4:  Write topic prose (parallel) + cross-topic deduplication
  → Stage 4b: Citation verifier — every claim traced to source, rewrite loop
  → Stage 5:  Assemble — intro, glossary, conclusion, references
  → Stage 6:  PDF render (WeasyPrint)
  → Stage 7:  Quality report
```

## Resuming

Every stage checkpoints to disk. Resume from any stage:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib python run.py --playlist "..." --from 5
```

Stages: 1=fetch, 2=transcribe, 3=group, 4=write+verify, 5=assemble, 6=render

## Configuration

Edit `config.yaml`:

```yaml
llm:
  provider: openai       # or: anthropic
  model: gpt-4o          # or: claude-opus-4-6
  temperature: 0.3

pipeline:
  batch_size: 4
  whisper_model: large-v3
```

## Running Tests

```bash
python -m pytest tests/
```
