#!/usr/bin/env python3
"""
03_gen_reports_from_mosaics_flammingo.py (Med-Flamingo version)

Stage 2: Discover INNER sequence dirs, read EXISTING mosaic images,
and query Med-Flamingo to generate angiography-style reports.

Requirements:
    pip install torch torchvision transformers Pillow pandas tqdm huggingface_hub --break-system-packages

CSV behavior:
- Adds: Timestamp, Model Name
- Appends row-by-row
- Output directory: <base_path>_Output/
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.csv_helpers import append_csv_row, ensure_csv_header
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
DEFAULT_MODEL_NAME = "med-flamingo/med-flamingo"
DEFAULT_TIMEOUT_S = 180


# -----------------------------
# Med-Flamingo Model Manager
# -----------------------------
class MedFlamingoModel:
    def __init__(self, model_name, device="auto"):
        print(f"Loading Med-Flamingo model: {model_name}...")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            self.device = "cpu"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, torch_dtype=torch.float32,
            ).to(self.device)
            self.model.eval()
            print("Model loaded on CPU")

    def generate_report(self, image_path, prompt, max_tokens=1024):
        try:
            image = Image.open(image_path).convert("RGB")
            full_prompt = f"<image>{prompt}"

            inputs = self.processor(
                text=full_prompt, images=image, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")


# -----------------------------
# Main processing loop
# -----------------------------
def run_inference(infos, out_path, columns, model, model_name, delay):
    total = len(infos)

    with tqdm(total=total, desc="Generating reports with Med-Flamingo", unit="seq") as pbar:
        for info in infos:
            row: Dict[str, Any] = {
                "Timestamp": utc_timestamp(),
                "Model Name": model_name,
                "sequence_dir": info.seq_rel,
            }

            if not info.ok:
                row.update(build_report_missing_mosaic_row(info.error or "Missing mosaic"))
            else:
                try:
                    raw = model.generate_report(info.mosaic_path, REPORT_GENERATION_PROMPT, max_tokens=1024)
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
    parser = argparse.ArgumentParser(
        description="Generate angiography reports using Med-Flamingo"
    )
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--frames_subdir", default="frames")
    parser.add_argument("--mosaic_name", default="mosaic.png")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Searching for sequences in: {args.base_path}")
    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    print(f"Found {len(seq_dirs)} sequence directories")

    if args.limit:
        seq_dirs = seq_dirs[: args.limit]
        print(f"Limited to first {args.limit} sequences")

    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)
    available = sum(1 for i in infos if i.ok)
    print(f"Available mosaics: {available}/{len(infos)}")

    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "medflamingo_generated_reports.csv"

    ensure_csv_header(out_csv, REPORT_CSV_COLUMNS)
    print(f"Output CSV: {out_csv}")

    try:
        model = MedFlamingoModel(args.model, device=args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nPlease ensure you have installed the required packages:")
        print("  pip install torch torchvision transformers Pillow pandas tqdm huggingface_hub --break-system-packages")
        sys.exit(1)

    print("\nStarting report generation...")
    run_inference(
        infos=infos, out_path=out_csv, columns=REPORT_CSV_COLUMNS,
        model=model, model_name=args.model, delay=args.delay,
    )

    print("\nDone! Incremental results preserved.")
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — partial results saved.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
