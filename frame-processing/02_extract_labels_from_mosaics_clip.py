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
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    import clip  # from openai/CLIP
except ImportError:
    print("ERROR: openai/CLIP not found. Install with:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# -----------------------------
# Questions and answer options
# -----------------------------
QUESTIONS_WITH_OPTIONS = [
    {
        "question": "Which artery is catheterized?",
        "options": [
            "No catheter visible",
            "Femoral artery",
            "Radial artery",
            "Brachial artery",
            "Carotid artery",
            "Vertebral artery",
            "Coronary artery",
            "Renal artery",
            "Mesenteric artery",
            "Iliac artery",
            "Unclear or other artery",
        ],
    },
    {
        "question": "Is variant anatomy present?",
        "options": [
            "No variant anatomy visible",
            "Yes, variant anatomy present",
            "Unclear if variant anatomy present",
        ],
    },
    {
        "question": "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
        "options": [
            "No hemorrhage or extravasation",
            "Yes, hemorrhage present",
            "Yes, contrast extravasation present",
            "Yes, both hemorrhage and extravasation present",
            "Unclear",
        ],
    },
    {
        "question": "Is there evidence of arterial or venous dissection?",
        "options": [
            "No dissection visible",
            "Yes, arterial dissection present",
            "Yes, venous dissection present",
            "Unclear if dissection present",
        ],
    },
    {
        "question": "Is stenosis present in any visualized vessel?",
        "options": [
            "No stenosis visible",
            "Yes, mild stenosis present",
            "Yes, moderate stenosis present",
            "Yes, severe stenosis present",
            "Unclear if stenosis present",
        ],
    },
    {
        "question": "Is an endovascular stent visible in this sequence?",
        "options": [
            "No stent visible",
            "Yes, stent visible",
            "Unclear if stent present",
        ],
    },
]

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_CLIP_MODEL = "ViT-B/32"  # common + lightweight
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# CSV helpers (append-only)
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)

def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)

# -----------------------------
# Directory discovery
# -----------------------------
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    seq_dirs: List[Path] = []
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists():
            continue
        try:
            if any(p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()):
                seq_dirs.append(d)
        except PermissionError:
            continue
    return sorted(seq_dirs, key=lambda p: p.as_posix())

# -----------------------------
# CLIP Model Loading
# -----------------------------
def load_clip_model(
    model_name: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any, torch.device]:
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Loading CLIP model: {model_name} on {dev}")
    model, preprocess = clip.load(model_name, device=dev)
    model.eval()
    print("✓ CLIP model loaded successfully")
    return model, preprocess, dev

# -----------------------------
# CLIP Inference
# -----------------------------
def clip_classify(
    image_path: Path,
    question: str,
    options: List[str],
    model: Any,
    preprocess: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Zero-shot multiple choice using CLIP:
      - build prompts from (question + option)
      - compute image-text similarity
      - softmax over options -> confidence
    """
    try:
        img = Image.open(image_path).convert("RGB")
        image_input = preprocess(img).unsqueeze(0).to(device)

        # Simple prompt format; you can tweak wording here if needed
        prompts = [f"{question} {opt}" for opt in options]
        text_tokens = clip.tokenize(prompts, truncate=True).to(device)

        with torch.no_grad():
            # logits_per_image: [1, N]
            logits_per_image, _ = model(image_input, text_tokens)
            scores = logits_per_image.squeeze(0)  # [N]
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
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@dataclass
class SequenceMosaicInfo:
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    ok: bool
    error: Optional[str] = None

def load_mosaics(seq_dirs: List[Path], base_path: Path, mosaic_name: str) -> List[SequenceMosaicInfo]:
    infos: List[SequenceMosaicInfo] = []
    for d in seq_dirs:
        rel = d.relative_to(base_path).as_posix()
        mp = d / mosaic_name
        exists = mp.exists()
        infos.append(
            SequenceMosaicInfo(
                seq_dir=d,
                seq_rel=rel,
                mosaic_path=mp,
                ok=exists,
                error=None if exists else "Missing mosaic",
            )
        )
    return infos

# -----------------------------
# Main processing loop
# -----------------------------
def run_clip_analysis(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    columns: List[str],
    model: Any,
    preprocess: Any,
    device: torch.device,
    model_name: str,
    delay: float = 0.0,
) -> None:
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
                    row.update(
                        {
                            "answer": "Not stated",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": info.error,
                        }
                    )
                else:
                    result = clip_classify(
                        image_path=info.mosaic_path,
                        question=question,
                        options=options,
                        model=model,
                        preprocess=preprocess,
                        device=device,
                    )
                    row.update(
                        {
                            "answer": result["answer"],
                            "confidence": result["confidence"],
                            "evidence": json.dumps(result["evidence"]),
                            "notes": result["notes"],
                        }
                    )

                append_csv_row(out_path, row, columns)
                pbar.update(1)

                if delay:
                    time.sleep(delay)

# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract labels from angiography mosaics using OpenAI CLIP"
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Base path to DICOM sequences",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=DEFAULT_CLIP_MODEL,
        help="CLIP model name (e.g., 'ViT-B/32', 'ViT-L/14')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or None for auto",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between analyses (seconds)",
    )
    parser.add_argument(
        "--frames_subdir",
        default="frames",
        help="Subdirectory name containing frames",
    )
    parser.add_argument(
        "--mosaic_name",
        default="mosaic.png",
        help="Mosaic filename to analyze",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sequences to process",
    )

    args = parser.parse_args()

    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[: args.limit]

    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)

    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "mosaics_extracted_labels_clip.csv"

    columns = [
        "Timestamp",
        "Model Name",
        "sequence_dir",
        "question",
        "answer",
        "confidence",
        "evidence",
        "notes",
    ]
    ensure_csv_header(out_csv, columns)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"CLIP Model: {args.clip_model}")

    model, preprocess, device = load_clip_model(args.clip_model, args.device)

    run_clip_analysis(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=model,
        preprocess=preprocess,
        device=device,
        model_name=args.clip_model,
        delay=args.delay,
    )

    print("Done ✔ Incremental results preserved.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
