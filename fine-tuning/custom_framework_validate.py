#!/usr/bin/env python3
"""
run_clip_binary_qa_on_validation_sequences.py

Use a trained AngioVision CLIP-style model checkpoint to answer a fixed set of
binary QA questions on a validation dataset where each "study" is a single sequence dir.

Your validation dir structure (each inner dir is a sequence):
  /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed/
    <SOPInstanceUID or sequence_dir_name>/
        frames/
            <image files>
        metadata.csv   # contains: AccessionNumber, SOPInstanceUID
        mosaic.png     # optional, not used
        ...

What this script does:
1) Scans all inner sequence dirs under --data_dir
2) For each sequence dir:
   - reads metadata.csv to get AccessionNumber and SOPInstanceUID
   - loads all frame images from frames/
   - encodes the sequence with the trained vision tower (ViT + pooling)
   - performs binary QA via CLIP similarity between the image embedding and text hypotheses
3) Writes a CSV with:
   AccessionNumber, SOPInstanceUID, Question, Answer

Binary QA strategy:
- For each question, we create two hypotheses (YES vs NO) as short clinical statements.
- Compute similarity(image, yes_hyp) vs similarity(image, no_hyp)
- Answer is YES if sim_yes > sim_no else NO

NOTE ABOUT "LABELED DATASET":
- You asked to "see performance on a labeled dataset", but the labeling file/format
  was not provided here. This script produces predictions.
- If you provide a label CSV (AccessionNumber/SOPInstanceUID + ground-truth answers),
  I can extend this to compute accuracy/F1/confusion matrix per question.

Example:
  python run_clip_binary_qa_on_validation_sequences.py \
    --checkpoint /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/checkpoint_epoch_1.pt \
    --data_dir /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
    --out_csv ./clip_binary_qa_predictions.csv \
    --device cuda \
    --vit_name google/vit-base-patch16-224-in21k \
    --bert_name bert-base-uncased \
    --embed_dim 256 \
    --pooling max \
    --frame_chunk_size 64 \
    --max_frames 512

"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import ViTModel, BertModel, BertTokenizer

# -----------------------------
# HF image processor compatibility
# -----------------------------
try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:
    _ViTProcessor = None

try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:
    _ViTFeatureExtractor = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

QA_QUESTIONS = [
    "Is variant anatomy present?",
    "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
    "Is there evidence of arterial or venous dissection?",
    "Is stenosis present in any visualized vessel?",
    "Is an endovascular stent visible in this sequence?",
]

POOL_CHOICES = ("max", "mean", "logsumexp")


# -----------------------------
# Helpers
# -----------------------------
def get_vit_processor(vit_name: str):
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(vit_name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(vit_name)
    raise ImportError("Neither ViTImageProcessor nor ViTFeatureExtractor is available in transformers.")


def list_images(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists() or not frames_dir.is_dir():
        return []
    imgs = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)


def safe_read_metadata_csv(seq_dir: Path) -> Optional[Tuple[str, str]]:
    """
    Reads seq_dir/metadata.csv and extracts:
      AccessionNumber, SOPInstanceUID
    Returns (AccessionNumber, SOPInstanceUID) or None if missing/invalid.
    """
    meta_path = seq_dir / "metadata.csv"
    if not meta_path.exists():
        return None

    try:
        df = pd.read_csv(meta_path)
    except Exception:
        return None

    # tolerate different casing/spaces
    cols = {c.strip(): c for c in df.columns}
    if "AccessionNumber" not in cols or "SOPInstanceUID" not in cols:
        return None

    acc = str(df.loc[0, cols["AccessionNumber"]]).strip()
    sop = str(df.loc[0, cols["SOPInstanceUID"]]).strip()
    if not acc or not sop or acc.lower() == "nan" or sop.lower() == "nan":
        return None
    return acc, sop


def sample_or_limit(paths: List[Path], max_frames: Optional[int]) -> List[Path]:
    if max_frames is None or max_frames <= 0:
        return paths
    if len(paths) <= max_frames:
        return paths
    # uniform subsample (keeps temporal coverage)
    idxs = torch.linspace(0, len(paths) - 1, steps=max_frames).long().tolist()
    return [paths[i] for i in idxs]


def make_yes_no_hypotheses(question: str) -> Tuple[str, str]:
    """
    Convert each question into a pair of symmetric statements.
    This is important: CLIP-style zero-shot works better with explicit statements.
    """
    q = question.strip().rstrip("?").lower()

    if "variant anatomy" in q:
        yes = "Angiography demonstrates variant vascular anatomy."
        no = "Angiography demonstrates no variant vascular anatomy."
        return yes, no

    if "hemorrhage" in q or "extravasation" in q:
        yes = "Angiography shows hemorrhage or contrast extravasation."
        no = "Angiography shows no hemorrhage and no contrast extravasation."
        return yes, no

    if "dissection" in q:
        yes = "Angiography shows evidence of arterial or venous dissection."
        no = "Angiography shows no evidence of arterial or venous dissection."
        return yes, no

    if "stenosis" in q:
        yes = "Angiography shows stenosis in a visualized vessel."
        no = "Angiography shows no stenosis in any visualized vessel."
        return yes, no

    if "stent" in q:
        yes = "An endovascular stent is visible on angiography."
        no = "No endovascular stent is visible on angiography."
        return yes, no

    # fallback generic (still symmetric)
    yes = f"Yes. {question}"
    no = f"No. {question}"
    return yes, no


# -----------------------------
# Model (must match your training architecture)
# -----------------------------
class PooledCLIP(nn.Module):
    """
    Minimal inference version matching the training model:
    - ViTModel + projection
    - BertModel + projection
    - pooling across frames and sequences
    """

    def __init__(
        self,
        vit_name: str,
        bert_name: str,
        embed_dim: int = 256,
        frame_pooling: str = "max",
        sequence_pooling: str = "max",
    ):
        super().__init__()
        if frame_pooling not in POOL_CHOICES:
            raise ValueError(f"frame_pooling must be one of {POOL_CHOICES}, got {frame_pooling}")
        if sequence_pooling not in POOL_CHOICES:
            raise ValueError(f"sequence_pooling must be one of {POOL_CHOICES}, got {sequence_pooling}")

        self.vit = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.bert_hidden = self.bert.config.hidden_size

        self.frame_pooling = frame_pooling
        self.sequence_pooling = sequence_pooling

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.vit_hidden),
            nn.GELU(),
            nn.Linear(self.vit_hidden, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert_hidden, self.bert_hidden),
            nn.GELU(),
            nn.Linear(self.bert_hidden, embed_dim),
        )

        # needed because your checkpoint likely contains it
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @torch.no_grad()
    def encode_text(self, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        out = self.bert(**tok)
        cls = out.last_hidden_state[:, 0, :]
        emb = F.normalize(self.text_proj(cls), dim=-1)
        return emb  # (N, D)

    def _pool_stack(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "max":
            return x.max(dim=0).values
        if mode == "mean":
            return x.mean(dim=0)
        if mode == "logsumexp":
            return torch.logsumexp(x, dim=0)
        raise ValueError(mode)

    @torch.no_grad()
    def encode_sequence_from_frames(
        self,
        processor,
        frame_paths: List[Path],
        device: torch.device,
        frame_chunk_size: int = 16,
        max_frames: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode ONE sequence (since your validation has one sequence per study).
        Pools across frames according to self.frame_pooling.
        Returns normalized projected embedding: (1, D)
        """
        frame_paths = sample_or_limit(frame_paths, max_frames=max_frames)

        # We'll avoid storing all frame embeddings by doing running reduction.
        if self.frame_pooling == "max":
            running = torch.full((self.vit_hidden,), -1e9, device=device)
        elif self.frame_pooling == "mean":
            running = torch.zeros((self.vit_hidden,), device=device)
            count = 0
        elif self.frame_pooling == "logsumexp":
            running = torch.full((self.vit_hidden,), -float("inf"), device=device)
        else:
            raise ValueError(self.frame_pooling)

        # process frames in chunks
        for i in range(0, len(frame_paths), frame_chunk_size):
            chunk_paths = frame_paths[i : i + frame_chunk_size]
            imgs: List[Image.Image] = []
            for p in chunk_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue
            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            out = self.vit(pixel_values=pixel_values)
            feats = out.last_hidden_state[:, 0, :]  # (B, hidden)

            if self.frame_pooling == "max":
                running = torch.maximum(running, feats.max(dim=0).values)
            elif self.frame_pooling == "mean":
                running = running + feats.sum(dim=0)
                count += feats.size(0)
            else:  # logsumexp
                running = torch.logaddexp(running, torch.logsumexp(feats, dim=0))

        if self.frame_pooling == "mean":
            if count > 0:
                running = running / float(count)

        # In validation, each "study" has only one sequence -> no need for sequence pooling.
        seq_feat = running.unsqueeze(0)  # (1, hidden)
        img_emb = F.normalize(self.vision_proj(seq_feat), dim=-1)  # (1, D)
        return img_emb


# -----------------------------
# Main inference
# -----------------------------
@torch.no_grad()
def run_inference(
    model: PooledCLIP,
    processor,
    tokenizer,
    data_dir: Path,
    out_csv: Path,
    device: torch.device,
    frame_chunk_size: int,
    max_frames: Optional[int],
):
    seq_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    seq_dirs = sorted(seq_dirs)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    pbar = tqdm(seq_dirs, desc="Sequences", dynamic_ncols=True)
    for seq_dir in pbar:
        meta = safe_read_metadata_csv(seq_dir)
        if meta is None:
            continue
        acc, sop = meta

        frames_dir = seq_dir / "frames"
        frame_paths = list_images(frames_dir)
        if not frame_paths:
            continue

        # Encode (single) sequence as image embedding
        img_emb = model.encode_sequence_from_frames(
            processor=processor,
            frame_paths=frame_paths,
            device=device,
            frame_chunk_size=frame_chunk_size,
            max_frames=max_frames,
        )  # (1, D)

        # Answer each question
        for q in QA_QUESTIONS:
            yes_h, no_h = make_yes_no_hypotheses(q)
            txt_emb = model.encode_text(tokenizer, [yes_h, no_h], device=device)  # (2, D)

            sims = (img_emb @ txt_emb.t()).squeeze(0)  # (2,)
            pred = "YES" if sims[0].item() > sims[1].item() else "NO"

            rows.append(
                {
                    "AccessionNumber": acc,
                    "SOPInstanceUID": sop,
                    "Question": q,
                    "Answer": pred,
                }
            )

    # write output
    df = pd.DataFrame(rows, columns=["AccessionNumber", "SOPInstanceUID", "Question", "Answer"])
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote predictions to: {out_csv}")


def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path, device: torch.device):
    """
    Loads a training checkpoint that was saved like:
      torch.save({"model_state": model.state_dict(), ...}, path)
    or possibly directly as state_dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        # could be a raw state_dict or other dict; try best-effort
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}")
    print("[INFO] Checkpoint loaded.")


def build_argparser():
    ap = argparse.ArgumentParser(description="Run CLIP-style binary QA on validation sequence dirs.")

    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt (e.g., checkpoint_epoch_1.pt)",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Validation DICOM_Sequence_Processed directory containing per-sequence folders.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for predictions: AccessionNumber,SOPInstanceUID,Question,Answer",
    )

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--bert_name", type=str, default="bert-base-uncased")
    ap.add_argument("--embed_dim", type=int, default=256)

    ap.add_argument("--pooling", type=str, default="max", choices=POOL_CHOICES, help="Default pooling.")
    ap.add_argument(
        "--frame_pooling",
        type=str,
        default=None,
        choices=POOL_CHOICES,
        help="Pooling across frames. If omitted, uses --pooling.",
    )
    ap.add_argument(
        "--sequence_pooling",
        type=str,
        default=None,
        choices=POOL_CHOICES,
        help="Pooling across sequences. (Not used here since one sequence per study.)",
    )

    ap.add_argument("--frame_chunk_size", type=int, default=64, help="Frames per ViT forward pass.")
    ap.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="If >0, uniformly subsample frames per sequence to this many.",
    )

    return ap


def main():
    args = build_argparser().parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    pooling_default = args.pooling
    frame_pooling = args.frame_pooling if args.frame_pooling else pooling_default
    sequence_pooling = args.sequence_pooling if args.sequence_pooling else pooling_default

    processor = get_vit_processor(args.vit_name)
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    model = PooledCLIP(
        vit_name=args.vit_name,
        bert_name=args.bert_name,
        embed_dim=args.embed_dim,
        frame_pooling=frame_pooling,
        sequence_pooling=sequence_pooling,
    ).to(device)
    model.eval()

    ckpt_path = Path(args.checkpoint)
    load_checkpoint_into_model(model, ckpt_path, device=device)

    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None

    run_inference(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        data_dir=data_dir,
        out_csv=out_csv,
        device=device,
        frame_chunk_size=args.frame_chunk_size,
        max_frames=max_frames,
    )


if __name__ == "__main__":
    main()
