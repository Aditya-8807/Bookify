#!/usr/bin/env python3
"""
Bookify — YouTube Playlist → PDF Book

Stage map:
  1  fetch audio + descriptions + ref content (no LLM except URL filter)
  2  transcribe with Whisper (NO LLM)
  3  terminology correction (LLM)
  4  group + order topics (LLM)
  5  write topic prose + citation verify + prose polish (LLM)
  6  assemble full book (LLM)
  7  render PDF

Usage:
  python run.py --playlist <url>
  python run.py --playlist <url> --from 2 --to 2   # transcription only
  python run.py --playlist <url> --from 3           # resume from terminology
"""
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from llm.client import client_from_config, get_cost_summary
from pipeline.fetcher import fetch_all
from pipeline.transcriber import transcribe_all
from pipeline.terminology_corrector import correct_all
from pipeline.grouper import group_and_order, fetch_and_checkpoint_ref_content
from pipeline.topic_writer import write_all_topics
from pipeline.citation_verifier import verify_all_topics
from pipeline.prose_polisher import polish_all_topics
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
    parser.add_argument("--playlist", default=None,
                        help="YouTube playlist URL (required only for Stage 1)")
    parser.add_argument("--from", dest="from_stage", type=int, default=1,
                        help="Start from stage N (default: 1)")
    parser.add_argument("--to", dest="to_stage", type=int, default=7,
                        help="Stop after stage N (default: 7 = run all)")
    parser.add_argument("--title", default="Building LLMs from Scratch")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N videos (useful for testing the full pipeline)")
    parser.add_argument("--video-id", dest="video_id", default=None,
                        help="Process only this specific video ID")
    args = parser.parse_args()

    if args.from_stage == 1 and not args.playlist:
        parser.error("--playlist is required when running from Stage 1")

    config = load_config(args.config)
    llm = client_from_config(config)
    batch_size = config.get("pipeline", {}).get("batch_size", 1)
    whisper_model = config.get("pipeline", {}).get("whisper_model", "small")
    whisper_compute_type = config.get("pipeline", {}).get("whisper_compute_type", "int8")
    base_dir = Path(config.get("paths", {}).get("checkpoints", "checkpoints"))
    audio_dir = Path(config.get("paths", {}).get("audio_temp", "checkpoints/audio"))
    output_dir = Path(config.get("paths", {}).get("output", "output"))

    progress = PipelineProgress()

    def _run(stage: int) -> bool:
        return args.from_stage <= stage <= args.to_stage

    def _stop(after: int) -> bool:
        return args.to_stage < after

    with progress.live():

        # ── Stage 1: Fetch ────────────────────────────────────────────────
        if _run(1):
            videos = fetch_all(
                args.playlist, llm, batch_size=batch_size,
                base_dir=base_dir, audio_dir=audio_dir, progress=progress,
            )
            fetch_and_checkpoint_ref_content(videos, base_dir=base_dir, progress=progress)
        else:
            keys = list_checkpoints("01_fetch", base_dir=base_dir)
            videos = [load_checkpoint("01_fetch", k, base_dir=base_dir) for k in keys]
            videos.sort(key=lambda v: v["playlist_index"])

        if args.video_id:
            videos = [v for v in videos if v["video_id"] == args.video_id]
            if not videos:
                print(f"ERROR: video_id '{args.video_id}' not found in checkpoints")
                return
        elif args.limit:
            videos = videos[:args.limit]

        vid_order = {v["video_id"]: v["playlist_index"] for v in videos}

        if _stop(2):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 2: Transcribe (Whisper — NO LLM) ───────────────────────
        if _run(2):
            transcripts = transcribe_all(
                videos, whisper_model=whisper_model,
                compute_type=whisper_compute_type,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("02_transcripts", base_dir=base_dir)
            transcripts = [load_checkpoint("02_transcripts", k, base_dir=base_dir) for k in keys]
            transcripts.sort(key=lambda t: vid_order.get(t["video_id"], 0))

        if _stop(3):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 3: Terminology correction (LLM) ────────────────────────
        if _run(3):
            corrected = correct_all(
                transcripts, llm, base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("02b_corrected", base_dir=base_dir)
            corrected = [load_checkpoint("02b_corrected", k, base_dir=base_dir) for k in keys]
            corrected.sort(key=lambda t: vid_order.get(t["video_id"], 0))

        if _stop(4):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 4: Group + order (LLM) ─────────────────────────────────
        if _run(4):
            groups = group_and_order(corrected, videos, llm, base_dir=base_dir, progress=progress)
        else:
            groups = load_checkpoint("03_groups", "groups", base_dir=base_dir)

        if _stop(5):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 5: Write + verify + polish (LLM) ───────────────────────
        if _run(5):
            written = write_all_topics(
                groups, corrected, llm, base_dir=base_dir, progress=progress,
            )
            trans_by_vid = {t["video_id"]: t for t in corrected}
            groups_by_slug = {g["slug"]: g for g in groups}
            verified = verify_all_topics(
                written, groups_by_slug, trans_by_vid,
                base_dir=base_dir, progress=progress, llm_client=llm,
            )
            verified = polish_all_topics(verified, llm, base_dir=base_dir, progress=progress)
        else:
            keys = list_checkpoints("04c_polished", base_dir=base_dir)
            if keys:
                verified = [load_checkpoint("04c_polished", k, base_dir=base_dir) for k in keys]
            else:
                keys = list_checkpoints("04b_verified", base_dir=base_dir)
                verified = [load_checkpoint("04b_verified", k, base_dir=base_dir) for k in keys]
            slug_order = {g["slug"]: i for i, g in enumerate(groups)}
            verified.sort(key=lambda t: slug_order.get(t["slug"], 0))

        if _stop(6):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 6: Assemble (LLM) ───────────────────────────────────────
        if _run(6):
            book_markdown = assemble_book(verified, groups, llm, base_dir=base_dir, progress=progress)
        else:
            book_markdown = load_checkpoint("05_book", "book", base_dir=base_dir)

        if _stop(7):
            print(f"\n[Cost] {get_cost_summary()}")
            return

        # ── Stage 7: Render PDF ───────────────────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)
        html = markdown_to_html(book_markdown, title=args.title)
        output_path = str(output_dir / "book.pdf")
        render_pdf(html, output_path=output_path, progress=progress)
        print(f"\nPDF saved to: {output_path}")

    # Quality report + cost
    try:
        all_corrections = [c for t in corrected for c in t.get("corrections", [])]
        report = generate_report(verified, book_markdown, all_corrections)
        print_report(report)
    except Exception:
        pass
    print(f"\n[Cost] {get_cost_summary()}")


if __name__ == "__main__":
    main()
