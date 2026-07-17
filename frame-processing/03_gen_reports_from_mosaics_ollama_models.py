#!/usr/bin/env python3
"""
Stage 2 only: Discover INNER sequence dirs (same rule as stage 1),
read an EXISTING mosaic image from each sequence dir, and query an LLM (Ollama)
to generate an angiography-style report *based only on what is visible in the mosaic*.

CSV behavior (FINAL)
- Adds: Timestamp, Model Name
- Appends row-by-row (never rewrites the full file)
- If CSV exists, new rows are appended
- Output directory is ALWAYS:
    <base_path>_Output/
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.csv_helpers import append_csv_row, ensure_csv_header
from shared.image_utils import encode_image_base64
from shared.ollama_client import ollama_chat_with_images
from shared.prompts import (
    REPORT_CSV_COLUMNS,
    REPORT_GENERATION_PROMPT,
    build_report_error_row,
    build_report_missing_mosaic_row,
    build_report_row_from_parsed,
)
from shared.sequence_utils import find_sequence_dirs, load_mosaics
from shared.text_utils import safe_parse_json, utc_timestamp

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen3-vl:32b"
DEFAULT_TIMEOUT_S = 180


# -----------------------------
# Main processing loop
# -----------------------------
def run_llm(infos, out_path, columns, model, url, timeout, delay):
    total = len(infos)

    with tqdm(total=total, desc="Generating reports from mosaics", unit="seq") as pbar:
        for info in infos:
            row: Dict[str, Any] = {
                "Timestamp": utc_timestamp(),
                "Model Name": model,
                "sequence_dir": info.seq_rel,
            }

            if not info.ok:
                row.update(build_report_missing_mosaic_row(info.error or "Missing mosaic"))
            else:
                try:
                    images = [encode_image_base64(info.mosaic_path)]
                    raw = ollama_chat_with_images(REPORT_GENERATION_PROMPT, images, model, url, timeout)
                    parsed = safe_parse_json(raw)
                    if not parsed:
                        raise ValueError("Non-JSON response")
                    row.update(build_report_row_from_parsed(parsed))
                except Exception as e:
                    row.update(build_report_error_row(str(e)[:200]))

            append_csv_row(out_path, row, columns)
            pbar.update(1)
            if delay:
                time.sleep(delay)


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--frames_subdir", default="frames")
    parser.add_argument("--mosaic_name", default="mosaic.png")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[: args.limit]
    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)

    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "mosaics_generated_reports.csv"

    ensure_csv_header(out_csv, REPORT_CSV_COLUMNS)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")

    run_llm(
        infos=infos, out_path=out_csv, columns=REPORT_CSV_COLUMNS,
        model=args.model, url=args.url, timeout=args.timeout, delay=args.delay,
    )

    print("Done. Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
