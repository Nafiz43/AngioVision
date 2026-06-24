#!/usr/bin/env python3
"""
Stage 2 only: Discover INNER sequence dirs (same rule as stage 1),
read an EXISTING mosaic image from each sequence dir, and query RadFM
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

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.csv_helpers import append_csv_row, ensure_csv_header
from shared.image_utils import encode_image_base64
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
DEFAULT_RADFM_URL = "http://localhost:8000/generate"
DEFAULT_MODEL_NAME = "RadFM"
DEFAULT_TIMEOUT_S = 180


# -----------------------------
# RadFM-specific helpers
# -----------------------------
def radfm_generate_report(prompt, image_path, url, timeout, model_name="RadFM"):
    image_b64 = encode_image_base64(image_path)

    payload = {
        "image": image_b64,
        "prompt": prompt,
        "model": model_name,
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        for key in ("output", "generated_text", "response", "text", "result", "prediction", "report"):
            if key in result:
                return result[key]

        if isinstance(result, str):
            return result

        raise ValueError(f"Unexpected RadFM response format: {result}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"RadFM API request failed: {str(e)}")


# -----------------------------
# Main processing loop
# -----------------------------
def run_radfm(infos, out_path, columns, model, url, timeout, delay):
    total = len(infos)

    with tqdm(total=total, desc="Generating reports from mosaics (RadFM)", unit="seq") as pbar:
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
                    raw_response = radfm_generate_report(
                        prompt=REPORT_GENERATION_PROMPT,
                        image_path=info.mosaic_path,
                        url=url, timeout=timeout, model_name=model,
                    )
                    parsed = safe_parse_json(raw_response)
                    if not parsed:
                        raise ValueError(f"Non-JSON response from RadFM: {raw_response[:200]}")
                    row.update(build_report_row_from_parsed(parsed))
                except Exception as e:
                    error_msg = str(e)[:200]
                    row.update(build_report_error_row(error_msg, source="RadFM"))
                    print(f"Error processing {info.seq_rel}: {error_msg}", file=sys.stderr)

            append_csv_row(out_path, row, columns)
            pbar.update(1)
            if delay:
                time.sleep(delay)


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate angiography reports from mosaic images using RadFM"
    )
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_RADFM_URL)
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
    out_csv = output_root / "radfm_generated_reports.csv"

    ensure_csv_header(out_csv, REPORT_CSV_COLUMNS)

    print(f"RadFM Report Generation")
    print(f"{'=' * 50}")
    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")
    print(f"RadFM Endpoint: {args.url}")
    print(f"{'=' * 50}")

    run_radfm(
        infos=infos, out_path=out_csv, columns=REPORT_CSV_COLUMNS,
        model=args.model, url=args.url, timeout=args.timeout, delay=args.delay,
    )

    print(f"\nDone. Incremental results preserved.")
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
