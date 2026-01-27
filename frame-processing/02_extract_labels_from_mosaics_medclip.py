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

Notes:
- MedCLIP output typically provides `outputs.logits` (NOT `logits_per_image` like OpenAI CLIP).
- If `from_pretrained()` fails with a checkpoint/state_dict mismatch, install MedCLIP from GitHub:
    pip uninstall -y medclip
    pip install git+https://github.com/RyanWangZf/MedCLIP.git
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

from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModelViT

# # Try importing MedCLIP
# try:
#     from medclip import MedCLIPModel(
#         MedCLIPModel,
#         MedCLIPVisionModelViT,
#         MedCLIPVisionModelResNet,
#         MedCLIPProcessor,
#     )
# except ImportError:
#     print("ERROR: MedCLIP not found. Install with:")
#     print("  pip install medclip")
#     print("If weights/loading fail, try GitHub version:")
#     print("  pip uninstall -y medclip")
#     print("  pip install git+https://github.com/RyanWangZf/MedCLIP.git")
#     sys.exit(1)


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
DEFAULT_MODEL_NAME = "medclip-vit"  # "medclip-vit" or "medclip-resnet"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# CSV helpers (append-only)
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    """Create CSV with headers if it doesn't exist."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    """Append a single row to CSV."""
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


# -----------------------------
# Directory discovery
# -----------------------------
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    """Find all sequence directories containing frames."""
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
# MedCLIP Model Loading
# -----------------------------
def load_medclip_model(
    model_name: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any, torch.device]:
    """
    Load MedCLIP model and processor.

    Args:
        model_name: 'medclip-vit' or 'medclip-resnet'
        device: 'cuda', 'cpu', or None for auto-detection

    Returns:
        (model, processor, device)
    """
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    model_name_l = (model_name or "").strip().lower()
    if model_name_l not in {"medclip-vit", "medclip-resnet"}:
        raise ValueError("Model must be 'medclip-vit' or 'medclip-resnet'")

    print(f"Loading MedCLIP model: {model_name_l} on {dev}")

    try:
        vision_cls = MedCLIPVisionModelViT

        model = MedCLIPModel(vision_cls=vision_cls)
        # Many installs use an instance method that loads default pretrained weights
        model.from_pretrained()
        model = model.to(dev)
        model.eval()

        processor = MedCLIPProcessor()

        print("✓ MedCLIP model loaded successfully")
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
def _extract_option_scores_from_outputs(outputs: Any) -> torch.Tensor:
    """
    MedCLIP outputs usually contain `logits`. For 1 image and N text prompts, logits is often [N, 1].
    This returns a 1D tensor of length N with option scores.
    """
    if not hasattr(outputs, "logits"):
        raise AttributeError(f"MedCLIP outputs missing `logits`. Available attrs: {dir(outputs)}")

    logits = outputs.logits
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)

    # Common shapes:
    # - [N, 1] (N texts vs 1 image)
    # - [1, N] (1 image vs N texts)
    # - [N] (already flattened)
    if logits.ndim == 2:
        if logits.shape[1] == 1:
            scores = logits[:, 0]
        elif logits.shape[0] == 1:
            scores = logits[0, :]
        else:
            # fallback: flatten
            scores = logits.reshape(-1)
    else:
        scores = logits.reshape(-1)

    return scores


def medclip_classify(
    image_path: Path,
    question: str,
    options: List[str],
    model: Any,
    processor: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Use MedCLIP to classify which option best matches the image for a given question.

    Returns:
        Dict with 'answer', 'confidence', 'evidence', 'all_scores', and 'notes'
    """
    try:
        image = Image.open(image_path).convert("RGB")

        # Combine question + option (simple prompt format)
        prompts = [f"{question} {opt}" for opt in options]

        inputs = processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        # Move to device (some processor outputs include non-tensor fields; guard it)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            scores = _extract_option_scores_from_outputs(outputs)

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
def run_medclip_analysis(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    columns: List[str],
    model: Any,
    processor: Any,
    device: torch.device,
    model_name: str,
    delay: float = 0.0,
) -> None:
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
                    row.update(
                        {
                            "answer": "Not stated",
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": info.error,
                        }
                    )
                else:
                    result = medclip_classify(
                        image_path=info.mosaic_path,
                        question=question,
                        options=options,
                        model=model,
                        processor=processor,
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
        description="Extract labels from angiography mosaics using MedCLIP"
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Base path to DICOM sequences",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="MedCLIP model variant (medclip-vit or medclip-resnet)",
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
    out_csv = output_root / "mosaics_extracted_labels_medclip.csv"

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
    print(f"Model: {args.model}")

    model, processor, device = load_medclip_model(args.model, args.device)

    run_medclip_analysis(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=model,
        processor=processor,
        device=device,
        model_name=args.model,
        delay=args.delay,
    )

    print("Done ✔ Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
