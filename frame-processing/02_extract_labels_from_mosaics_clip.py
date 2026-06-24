#!/usr/bin/env python3
"""
extract_labels_from_mosaics_clip.py

Stage 2: Uses OpenAI CLIP (openai/clip-vit-base-patch32) to analyze mosaic images
and answer anatomical-level questions through zero-shot classification.

CSV behavior (FINAL)
- Adds: Timestamp, Model Name
- Removes: mosaic_file column
- Appends row-by-row (never rewrites the full file)
- If CSV exists, new rows are appended
- Output directory is ALWAYS:
    <base_path>_Output/

Install:
  pip install torch torchvision pandas numpy pillow tqdm git+https://github.com/openai/CLIP.git

Notes:
- CLIP is NOT a medical-specialized model; results may be weak on subtle angiography findings.
- Best used as a baseline / ablation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import clip  # from openai/CLIP
except ImportError:
    print("ERROR: openai/CLIP not found. Install with:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

from shared.csv_helpers import append_csv_row, ensure_csv_header
from shared.prompts import QUESTIONS_WITH_OPTIONS
from shared.sequence_utils import SequenceMosaicInfo, find_sequence_dirs, load_mosaics
from shared.text_utils import utc_timestamp

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_CLIP_MODEL = "ViT-B/32"


# -----------------------------
# CLIP Model Loading
# -----------------------------
def load_clip_model(model_name, device=None):
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Loading CLIP model: {model_name} on {dev}")
    model, preprocess = clip.load(model_name, device=dev)
    model.eval()
    print("CLIP model loaded successfully")
    return model, preprocess, dev


# -----------------------------
# CLIP Inference
# -----------------------------
def clip_classify(image_path, question, options, model, preprocess, device):
    try:
        img = Image.open(image_path).convert("RGB")
        image_input = preprocess(img).unsqueeze(0).to(device)

        prompts = [f"{question} {opt}" for opt in options]
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)

        with torch.no_grad():
            logits_per_image, _ = model(image_input, text_tokens)
            scores = logits_per_image.squeeze(0)
            probs = torch.softmax(scores, dim=0).detach().cpu().numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx] * 100.0)
        answer = options[best_idx]

        all_scores = {opt: float(prob * 100.0) for opt, prob in zip(options, probs)}
        top_indices = np.argsort(probs)[-3:][::-1]
        evidence = [f"{options[i]}: {probs[i]*100:.1f}%" for i in top_indices]

        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "all_scores": all_scores,
            "notes": "CLIP zero-shot classification",
        }
    except Exception as e:
        return {
            "answer": "Error",
            "confidence": 0,
            "evidence": [],
            "all_scores": {},
            "notes": f"CLIP error: {str(e)[:200]}",
        }


# -----------------------------
# Main processing loop
# -----------------------------
def run_clip_analysis(infos, out_path, columns, model, preprocess, device, model_name, delay=0.0):
    total = len(infos) * len(QUESTIONS_WITH_OPTIONS)

    with tqdm(total=total, desc="Analyzing mosaics with CLIP", unit="q") as pbar:
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
                    result = clip_classify(
                        image_path=info.mosaic_path,
                        question=question,
                        options=options,
                        model=model,
                        preprocess=preprocess,
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
        description="Extract labels from angiography mosaics using OpenAI CLIP"
    )
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL)
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
    out_csv = output_root / "mosaics_extracted_labels_clip.csv"

    columns = [
        "Timestamp", "Model Name", "sequence_dir", "question",
        "answer", "confidence", "evidence", "notes",
    ]
    ensure_csv_header(out_csv, columns)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"CLIP Model: {args.clip_model}")

    model, preprocess, device = load_clip_model(args.clip_model, args.device)

    run_clip_analysis(
        infos=infos, out_path=out_csv, columns=columns,
        model=model, preprocess=preprocess, device=device,
        model_name=args.clip_model, delay=args.delay,
    )

    print("Done. Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
