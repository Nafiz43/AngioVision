#!/usr/bin/env python3

"""
Validation Mosaic QA + Evaluation (STRICT YES/NO VERSION)

Pipeline:
1. Discover validation sequence directories
2. Load validation CSV (ground truth)
3. Match rows using SOPInstanceUID
4. Send mosaic image to VLM
5. Ask validation question
6. Force binary prediction yes/no
7. Compute evaluation metrics
"""

import argparse
import base64
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


DEFAULT_BASE_PATH = Path(
"/data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed"
)

DEFAULT_GT_CSV = Path(
"/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"
)

DEFAULT_MODEL = "qwen3-vl:32b"
DEFAULT_URL = "http://localhost:11434/api/chat"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# STRICT PROMPT
# -----------------------------
BASE_PROMPT = """
You are a medical image analysis system.

You will be shown an angiography mosaic image containing multiple frames.

Answer the question using ONLY the image.

Question: {QUESTION}

STRICT RULES
- Output must be exactly ONE word.
- Allowed answers:
yes
no

Do NOT output:
- explanations
- punctuation
- JSON
- additional text
- confidence
- notes

VALID OUTPUT
yes
no

FINAL ANSWER:
"""


# -----------------------------
# Helpers
# -----------------------------

def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def b64_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode()


def normalize_binary(x):

    if x is None:
        return None

    s = str(x).strip().lower()

    if s.startswith("yes"):
        return "yes"

    if s.startswith("no"):
        return "no"

    return None


# -----------------------------
# Ollama Call
# -----------------------------

def query_model(prompt, image_b64, model, url):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 2
        },
    }

    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()

    return r.json()["message"]["content"]


# -----------------------------
# Directory Discovery
# -----------------------------

def find_sequence_dirs(base_path, frames_subdir):

    seq_dirs = []

    for d in base_path.rglob("*"):

        if not d.is_dir():
            continue

        frames_dir = d / frames_subdir

        if not frames_dir.exists():
            continue

        try:
            has_images = any(
                p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()
            )
        except Exception:
            has_images = False

        if has_images:
            seq_dirs.append(d)

    return sorted(seq_dirs)


# -----------------------------
# Metadata Reader
# -----------------------------

def read_metadata(metadata_path):

    if not metadata_path.exists():
        return {}

    df = pd.read_csv(metadata_path)

    info = {}

    for _, r in df.iterrows():
        k = str(r["Information"]).strip()
        v = str(r["Value"]).strip()
        info[k] = v

    return info


# -----------------------------
# Main
# -----------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--gt_csv", type=Path, default=DEFAULT_GT_CSV)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)

    parser.add_argument("--frames_subdir", default="frames")
    parser.add_argument("--mosaic_name", default="mosaic.png")

    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    gt = pd.read_csv(args.gt_csv)

    if args.limit:
        gt = gt.head(args.limit)

    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)

    sop_map = {}

    for d in seq_dirs:

        metadata = read_metadata(d / "metadata.csv")

        sop = metadata.get("SOPInstanceUID")

        if sop:
            sop_map[sop] = d

    output_root = Path(f"{args.base_path}_Output")
    output_root.mkdir(exist_ok=True)

    pred_csv = output_root / "validation_predictions.csv"
    metrics_csv = output_root / "validation_metrics.csv"

    rows = []

    print("Validation rows:", len(gt))
    print("Sequence dirs:", len(seq_dirs))

    with tqdm(total=len(gt)) as pbar:

        for _, r in gt.iterrows():

            sop = str(r["SOPInstanceUID"])
            question = str(r["Question"])
            gt_answer = str(r["Answer"])

            seq_dir = sop_map.get(sop)

            pred = "unclear"

            if seq_dir:

                mosaic = seq_dir / args.mosaic_name

                if mosaic.exists():

                    prompt = BASE_PROMPT.format(QUESTION=question)

                    try:

                        raw = query_model(
                            prompt,
                            b64_image(mosaic),
                            args.model,
                            args.url
                        )

                        raw = raw.strip().lower()

                        if raw.startswith("yes"):
                            pred = "yes"

                        elif raw.startswith("no"):
                            pred = "no"

                    except Exception:
                        pred = "unclear"

            rows.append(
                {
                    "timestamp": utc_timestamp(),
                    "SOPInstanceUID": sop,
                    "question": question,
                    "gt_answer": gt_answer,
                    "pred_answer": pred,
                }
            )

            pbar.update(1)

    df = pd.DataFrame(rows)

    df.to_csv(pred_csv, index=False)

    # -----------------------------
    # Metrics
    # -----------------------------

    df["gt_norm"] = df["gt_answer"].apply(normalize_binary)
    df["pred_norm"] = df["pred_answer"].apply(normalize_binary)

    eval_df = df.dropna(subset=["gt_norm", "pred_norm"])

    if len(eval_df) > 0:

        y_true = eval_df["gt_norm"].map({"yes":1,"no":0})
        y_pred = eval_df["pred_norm"].map({"yes":1,"no":0})

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    else:

        accuracy = precision = recall = f1 = None

    metrics = pd.DataFrame(
        [
            {"metric":"accuracy","value":accuracy},
            {"metric":"precision","value":precision},
            {"metric":"recall","value":recall},
            {"metric":"f1","value":f1},
        ]
    )

    metrics.to_csv(metrics_csv, index=False)

    print("\n===== FINAL METRICS =====")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1       :", f1)

    print("\nPredictions saved:", pred_csv)
    print("Metrics saved:", metrics_csv)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)