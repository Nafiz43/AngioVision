#!/usr/bin/env python3
"""
extract_labels_from_mosaics_biomedclip.py

Stage 2: Uses BioMedCLIP (medical vision-language model) to analyze mosaic images
and answer anatomical-level questions through zero-shot classification.

CSV behavior (FINAL)
- Adds: Timestamp, Model Name
- Removes: mosaic_file column
- Appends row-by-row (never rewrites the full file)
- If CSV exists, new rows are appended
- Output directory is ALWAYS:
    <base_path>_Output/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.biomedclip_utils import biomedclip_classify, load_biomedclip_model
from shared.csv_helpers import append_csv_row, ensure_csv_header
from shared.prompts import QUESTIONS_WITH_OPTIONS
from shared.sequence_utils import SequenceMosaicInfo, find_sequence_dirs, load_mosaics
from shared.text_utils import utc_timestamp

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# -----------------------------
# Main processing loop
# -----------------------------
def run_biomedclip_analysis(infos, out_path, columns, model, preprocess, tokenizer, device, model_name, delay=0.0):
    total = len(infos) * len(QUESTIONS_WITH_OPTIONS)

    with tqdm(total=total, desc="Analyzing mosaics with BioMedCLIP", unit="q") as pbar:
        for info in infos:
            for q_dict in QUESTIONS_WITH_OPTIONS:
                question = q_dict["question"]
                options = q_dict["options"]

                row = {
                    "Timestamp": utc_timestamp(),
                    "Model Name": model_name,
                    "sequence_dir": info.seq_rel,
                    "question": question,
                }

                if not info.ok:
                    row.update({
                        "answer": "Not stated",
                        "confidence": 0,
                        "evidence": "[]",
                        "notes": info.error,
                    })
                else:
                    result = biomedclip_classify(
                        image_path=info.mosaic_path,
                        question=question,
                        options=options,
                        model=model,
                        preprocess=preprocess,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    row.update({
                        "answer": result["answer"],
                        "confidence": result["confidence"],
                        "evidence": json.dumps(result["evidence"]),
                        "notes": result["notes"],
                    })

                append_csv_row(out_path, row, columns)
                pbar.update(1)

                if delay:
                    time.sleep(delay)


# -----------------------------
# Entrypoint
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract labels from angiography mosaics using BioMedCLIP"
    )
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default=None)
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
    out_csv = output_root / "mosaics_extracted_labels_biomedclip.csv"

    columns = [
        "Timestamp", "Model Name", "sequence_dir", "question",
        "answer", "confidence", "evidence", "notes",
    ]
    ensure_csv_header(out_csv, columns)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")

    model, preprocess, tokenizer, device = load_biomedclip_model(args.model, args.device)

    run_biomedclip_analysis(
        infos=infos, out_path=out_csv, columns=columns,
        model=model, preprocess=preprocess, tokenizer=tokenizer,
        device=device, model_name=args.model, delay=args.delay,
    )

    print("Done. Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
