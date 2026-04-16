#!/usr/bin/env python3
"""
Bookify — YouTube Playlist → PDF Book

Usage:
    python run.py --playlist <url>
    python run.py --playlist <url> --from 4      # resume from stage 4
    python run.py --playlist <url> --title "My Book Title"
"""
import argparse
from pathlib import Path

import yaml

from llm.client import client_from_config
from pipeline.fetcher import fetch_all
from pipeline.transcriber import transcribe_all
from pipeline.terminology_corrector import correct_all
from pipeline.grouper import group_and_order
from pipeline.topic_writer import write_all_topics
from pipeline.citation_verifier import verify_all_topics
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
    parser.add_argument("--playlist", required=True, help="YouTube playlist URL")
    parser.add_argument(
        "--from", dest="from_stage", type=int, default=1,
        help="Resume from stage N (1=fetch, 2=transcribe, 3=group, 4=write, 5=assemble, 6=render). Default: 1"
    )
    parser.add_argument("--title", default="Building LLMs from Scratch", help="Book title for the PDF cover")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    llm = client_from_config(config)
    batch_size = config.get("pipeline", {}).get("batch_size", 4)
    whisper_model = config.get("pipeline", {}).get("whisper_model", "large-v3")
    base_dir = Path(config.get("paths", {}).get("checkpoints", "checkpoints"))
    audio_dir = Path(config.get("paths", {}).get("audio_temp", "checkpoints/audio"))
    output_dir = Path(config.get("paths", {}).get("output", "output"))

    progress = PipelineProgress()

    with progress.live():
        # Stage 1: Fetch
        if args.from_stage <= 1:
            videos = fetch_all(
                args.playlist, llm, batch_size=batch_size,
                base_dir=base_dir, audio_dir=audio_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("01_fetch", base_dir=base_dir)
            videos = [load_checkpoint("01_fetch", k, base_dir=base_dir) for k in keys]

        # Stage 2: Transcribe
        if args.from_stage <= 2:
            transcripts = transcribe_all(
                videos, whisper_model=whisper_model, batch_size=batch_size,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("02_transcripts", base_dir=base_dir)
            transcripts = [load_checkpoint("02_transcripts", k, base_dir=base_dir) for k in keys]

        # Stage 2b: Terminology correction
        if args.from_stage <= 2:
            corrected = correct_all(
                transcripts, llm, batch_size=batch_size,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("02b_corrected", base_dir=base_dir)
            corrected = [load_checkpoint("02b_corrected", k, base_dir=base_dir) for k in keys]

        # Stage 3: Group + order + enrich
        if args.from_stage <= 3:
            groups = group_and_order(corrected, videos, llm, base_dir=base_dir, progress=progress)
        else:
            groups = load_checkpoint("03_groups", "groups", base_dir=base_dir)

        # Stage 4: Write topics
        if args.from_stage <= 4:
            written = write_all_topics(
                groups, corrected, llm, batch_size=batch_size,
                base_dir=base_dir, progress=progress,
            )
        else:
            keys = list_checkpoints("04_topics", base_dir=base_dir)
            written = [load_checkpoint("04_topics", k, base_dir=base_dir) for k in keys]

        # Stage 4b: Citation verification
        if args.from_stage <= 4:
            trans_by_vid = {t["video_id"]: t for t in corrected}
            groups_by_slug = {g["slug"]: g for g in groups}
            verified = verify_all_topics(
                written, groups_by_slug, trans_by_vid,
                batch_size=batch_size, base_dir=base_dir,
                progress=progress, llm_client=llm,
            )
        else:
            keys = list_checkpoints("04b_verified", base_dir=base_dir)
            verified = [load_checkpoint("04b_verified", k, base_dir=base_dir) for k in keys]

        # Stage 5: Assemble
        if args.from_stage <= 5:
            book_markdown = assemble_book(verified, groups, llm, base_dir=base_dir, progress=progress)
        else:
            book_markdown = load_checkpoint("05_book", "book", base_dir=base_dir)

        # Stage 6: Render PDF
        if args.from_stage <= 6:
            html = markdown_to_html(book_markdown, title=args.title)
            output_path = str(output_dir / "book.pdf")
            render_pdf(html, output_path=output_path, progress=progress)
            print(f"\nPDF saved to: {output_path}")

    # Stage 7: Quality report
    all_corrections = [c for t in corrected for c in t.get("corrections", [])]
    report = generate_report(verified, book_markdown, all_corrections)
    print_report(report)


if __name__ == "__main__":
    main()
