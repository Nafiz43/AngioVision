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

# Try importing BioMedCLIP
try:
    from transformers import AutoModel, AutoTokenizer
    import open_clip
except ImportError:
    print("ERROR: Required packages not found. Install with:")
    print("  pip install transformers open_clip_torch")
    sys.exit(1)

# -----------------------------
# Questions and answer options
# -----------------------------
# BioMedCLIP works with classification, so we define possible answers for each question
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
            "Unclear or other artery"
        ]
    },
    {
        "question": "Is variant anatomy present?",
        "options": [
            "No variant anatomy visible",
            "Yes, variant anatomy present",
            "Unclear if variant anatomy present"
        ]
    },
    {
        "question": "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
        "options": [
            "No hemorrhage or extravasation",
            "Yes, hemorrhage present",
            "Yes, contrast extravasation present",
            "Yes, both hemorrhage and extravasation present",
            "Unclear"
        ]
    },
    {
        "question": "Is there evidence of arterial or venous dissection?",
        "options": [
            "No dissection visible",
            "Yes, arterial dissection present",
            "Yes, venous dissection present",
            "Unclear if dissection present"
        ]
    },
    {
        "question": "Is stenosis present in any visualized vessel?",
        "options": [
            "No stenosis visible",
            "Yes, mild stenosis present",
            "Yes, moderate stenosis present",
            "Yes, severe stenosis present",
            "Unclear if stenosis present"
        ]
    },
    {
        "question": "Is an endovascular stent visible in this sequence?",
        "options": [
            "No stent visible",
            "Yes, stent visible",
            "Unclear if stent present"
        ]
    },
]

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
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
        if any(p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()):
            seq_dirs.append(d)
    return sorted(seq_dirs, key=lambda p: p.as_posix())


# -----------------------------
# BioMedCLIP Model Loading
# -----------------------------
def load_biomedclip_model(model_name: str, device: str = None) -> Tuple[Any, Any, Any, torch.device]:
    """
    Load BioMedCLIP model using open_clip architecture.
    
    Args:
        model_name: Model identifier (default: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
        device: 'cuda', 'cpu', or None for auto-detection
    
    Returns:
        Tuple of (model, preprocess, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Loading BioMedCLIP model: {model_name} on {device}")
    
    try:
        # Load model using open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Load tokenizer
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        print("✓ BioMedCLIP model loaded successfully")
        return model, preprocess_val, tokenizer, device
        
    except Exception as e:
        print(f"Error loading BioMedCLIP: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages:")
        print("   pip install open_clip_torch transformers")
        print("2. Ensure PyTorch is installed:")
        print("   pip install torch torchvision")
        print("3. Check internet connection for model download")
        raise


# -----------------------------
# BioMedCLIP Inference
# -----------------------------
def biomedclip_classify(
    image_path: Path,
    question: str,
    options: List[str],
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    """
    Use BioMedCLIP to classify which option best matches the image for a given question.
    
    Returns:
        Dict with 'answer', 'confidence', 'all_scores', 'evidence', and 'notes'
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Prepare text prompts by combining question with each option
        prompts = [f"{question} {option}" for option in options]
        
        # Tokenize text
        text_tokens = tokenizer(prompts).to(device)
        
        # Get predictions
        with torch.no_grad():
            # Encode image and text
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores (cosine similarity)
            # Shape: [1, num_options]
            similarity = (100.0 * image_features @ text_features.T)
            probs = torch.softmax(similarity, dim=-1).cpu().numpy()[0]
        
        # Get the best matching option
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx] * 100)  # Convert to percentage
        answer = options[best_idx]
        
        # Create score dictionary for all options
        all_scores = {opt: float(prob * 100) for opt, prob in zip(options, probs)}
        
        # Get top 3 options as evidence
        top_indices = np.argsort(probs)[-3:][::-1]
        evidence = [
            f"{options[i]}: {probs[i]*100:.1f}%"
            for i in top_indices
        ]
        
        return {
            "answer": answer,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "all_scores": all_scores,
            "notes": "BioMedCLIP zero-shot classification"
        }
        
    except Exception as e:
        return {
            "answer": "Error",
            "confidence": 0,
            "evidence": [],
            "all_scores": {},
            "notes": f"BioMedCLIP error: {str(e)[:200]}"
        }


# -----------------------------
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SequenceMosaicInfo:
    """Information about a sequence's mosaic image."""
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    ok: bool
    error: Optional[str] = None


def load_mosaics(seq_dirs: List[Path], base_path: Path, mosaic_name: str) -> List[SequenceMosaicInfo]:
    """Check which sequences have valid mosaic images."""
    infos = []
    for d in seq_dirs:
        rel = d.relative_to(base_path).as_posix()
        mp = d / mosaic_name
        infos.append(
            SequenceMosaicInfo(
                seq_dir=d,
                seq_rel=rel,
                mosaic_path=mp,
                ok=mp.exists(),
                error=None if mp.exists() else "Missing mosaic"
            )
        )
    return infos


# -----------------------------
# Main processing loop
# -----------------------------
def run_biomedclip_analysis(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    columns: List[str],
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: torch.device,
    model_name: str,
    delay: float = 0.0
):
    """Run BioMedCLIP analysis on all sequences and questions."""
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
                    # Mosaic doesn't exist
                    row.update({
                        "answer": "Not stated",
                        "confidence": 0,
                        "evidence": "[]",
                        "notes": info.error
                    })
                else:
                    # Run BioMedCLIP classification
                    result = biomedclip_classify(
                        image_path=info.mosaic_path,
                        question=question,
                        options=options,
                        model=model,
                        preprocess=preprocess,
                        tokenizer=tokenizer,
                        device=device
                    )
                    
                    row.update({
                        "answer": result["answer"],
                        "confidence": result["confidence"],
                        "evidence": json.dumps(result["evidence"]),
                        "notes": result["notes"]
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
    parser.add_argument(
        "--base_path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Base path to DICOM sequences"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="BioMedCLIP model identifier"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'cpu', or None for auto"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between analyses (seconds)"
    )
    parser.add_argument(
        "--frames_subdir",
        default="frames",
        help="Subdirectory name containing frames"
    )
    parser.add_argument(
        "--mosaic_name",
        default="mosaic.png",
        help="Mosaic filename to analyze"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sequences to process"
    )

    args = parser.parse_args()

    # Find sequence directories
    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[:args.limit]
    
    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)

    # Setup output path
    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "mosaics_extracted_labels_biomedclip.csv"

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

    # Load BioMedCLIP model
    model, preprocess, tokenizer, device = load_biomedclip_model(args.model, args.device)

    # Run analysis
    run_biomedclip_analysis(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
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
