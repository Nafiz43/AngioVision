#!/usr/bin/env python3
"""
extract_labels_from_mosaics_medclip.py

Stage 2: Uses MedCLIP (medical vision-language model) to analyze mosaic images
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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
except ImportError:
    print("ERROR: MedCLIP not found. Install with:")
    print("  pip install medclip")
    print("Or clone from: https://github.com/RyanWangZf/MedCLIP")
    sys.exit(1)

from shared.csv_helpers import append_csv_row, ensure_csv_header
from shared.prompts import QUESTIONS_WITH_OPTIONS
from shared.sequence_utils import SequenceMosaicInfo, find_sequence_dirs, load_mosaics
from shared.text_utils import utc_timestamp

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_MODEL_NAME = "medclip-vit"


# -----------------------------
# MedCLIP Model Loading
# -----------------------------
def load_medclip_model(model_name, device=None):
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model_name_l = (model_name or "").strip().lower()
    if model_name_l not in {"medclip-vit", "medclip-resnet"}:
        raise ValueError("Model must be 'medclip-vit' or 'medclip-resnet'")

    print(f"Loading MedCLIP model: {model_name_l} on {dev}")

    try:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        model = model.to(dev)
        model.eval()

        processor = MedCLIPProcessor()

        print("MedCLIP model loaded successfully")
        return model, processor, dev
    except Exception as e:
        print(f"ERROR loading MedCLIP pretrained weights: {e}")
        print("\nCommon fixes:")
        print("  1) Install MedCLIP from GitHub (recommended if state_dict mismatch):")
        print("     pip uninstall -y medclip")
        print("     pip install git+https://github.com/RyanWangZf/MedCLIP.git")
        print("  2) Ensure torch/torchvision versions match your CUDA/driver setup.")
        raise


# -----------------------------
# MedCLIP Inference
# -----------------------------
def medclip_classify(image_path, question, options, model, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        prompts = [f"{question} {opt}" for opt in options]

        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

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
            "notes": "MedCLIP zero-shot classification",
        }
    except Exception as e:
        return {
            "answer": "Error",
            "confidence": 0,
            "evidence": [],
            "all_scores": {},
            "notes": f"MedCLIP error: {str(e)[:200]}",
        }


# -----------------------------
# Main processing loop
# -----------------------------
def run_medclip_analysis(infos, out_path, columns, model, processor, device, model_name, delay=0.0):
    total = len(infos) * len(QUESTIONS_WITH_OPTIONS)

    with tqdm(total=total, desc="Analyzing mosaics with MedCLIP", unit="q") as pbar:
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
                    result = medclip_classify(
                        image_path=info.mosaic_path,
                        question=question,
                        options=options,
                        model=model,
                        processor=processor,
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
        description="Extract labels from angiography mosaics using MedCLIP"
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
    out_csv = output_root / "mosaics_extracted_labels_medclip.csv"

    columns = [
        "Timestamp", "Model Name", "sequence_dir", "question",
        "answer", "confidence", "evidence", "notes",
    ]
    ensure_csv_header(out_csv, columns)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")

    model, processor, device = load_medclip_model(args.model, args.device)

    run_medclip_analysis(
        infos=infos, out_path=out_csv, columns=columns,
        model=model, processor=processor, device=device,
        model_name=args.model, delay=args.delay,
    )

    print("Done. Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
