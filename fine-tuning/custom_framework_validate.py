#!/usr/bin/env python3
"""
custom_framework_validate.py

Run CLIP-style binary QA on validation sequence directories using a trained checkpoint.

DATA LAYOUT (your validation set):
  DATA_DIR/
    <sequence_dir_1>/
      frames/              # images
      metadata.csv         # key-value CSV: columns [Information, Value]
      mosaic.png           # optional
    <sequence_dir_2>/
      ...

metadata.csv format (IMPORTANT):
  Information,Value
  SOPInstanceUID,1.2.276...
  AccessionNumber,202510081160
  ...

OUTPUT:
- --out_csv: AccessionNumber,SOPInstanceUID,Question,Answer
- --error_csv (optional): seq_dir,status,details

Binary QA method:
- For each question, we build YES/NO hypothesis texts
- Compare cosine similarity between image embedding and text embeddings
- Choose YES if sim_yes > sim_no else NO

Example:
  python3 custom_framework_validate.py \
    --checkpoint /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/checkpoint_epoch_1.pt \
    --data_dir /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
    --out_csv /data/Deep_Angiography/AngioVision/fine-tuning/output/clip_binary_qa_predictions.csv \
    --error_csv /data/Deep_Angiography/AngioVision/fine-tuning/output/clip_binary_qa_errors.csv \
    --device cuda \
    --frame_chunk_size 64 \
    --max_frames 0 \
    --pooling max
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
# from configs.questions import QA_QUESTIONS
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

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.questions import QA_QUESTIONS

# QA_QUESTIONS = [
#     "Is the catheter tip located in the inferior mesenteric artery? Please state yes or no.",
#     "Is there an arterial abnormality in this angiogram? Please state yes or no.",
#     "Is contrast injected through a microcatheter (as opposed to the base catheter)? Please state yes or no.",
#     "Is the sheath tip located in the right external iliac artery? Please state yes or no.",
#     "Is there an acute arterial abnormality in this angiogram? Please state yes or no.",
#     "Is the catheter tip located in the celiac artery or one of it's branches? Please state yes or no.",
#     "Is the catheter tip located in the aorta? Please state yes or no.",
#     "Is there an acute arterial abnormality demonstrated in this angiogram? Please state yes or no.",
#     "Is there a chronic arterial abnormality demonstrated in this angiogram? Please state yes or no.",
#     "Is the catheter tip located in the right internal iliac artery? Please state yes or no.",
#     "Is the perfused organ demonstrated in this angiogram the spleen? Please state yes or no.",
#     "Is the perfused organ demonstrated in this angiogram the liver? Please state yes or no.",
#     "Is the catheter tip located in the superior mesenteric artery or one of it's branches? Please state yes or no.",
#     "Is the catheter tip located in the right hepatic artery or one of it's branches? Please state yes or no.",
#     "Is the sheath tip located in the left external iliac artery? Please state yes or no.",
#     "Is the perfused organ demonstrated in this angiogram the kidney? Please state yes or no.",
#     "Is there a vascular tumor demonstrated in this angiogram? Please state yes or no.",
#     "Is the catheter tip located in the renal artery? Please state yes or no.",
#     "Is the perfused organ demonstrated in this angiogram the bowel? Please state yes or no.",
#     "Is the catheter tip located in the celiac artery? Please state yes or no.",
#     "Is there a vascular aberrancy demonstrated in this angiogram? Please state yes or no.",
#     "Is the origin of the right inferior epigastric artery opacified in this angiogram? Please state yes or no.",
#     "Does the angiogram demonstrate competitive inflow from another mesenteric vessel? Please state yes or no.",
#     "Is stenosis of the celiac artery, either acute or chronic, demonstrated on this angiogram? Please state yes or no.",
#     "Is the origin of the deep circumflex iliac artery opacified in this angiogram? Please state yes or no.",
#     "Is the left colic artery opacified in this angiogram? Please state yes or no.",
#     "Is the catheter tip located in the left colic artery or one of it's branches? Please state yes or no.",
#     "Does the angiogram demonstrate acute gastrointestinal bleeding? Please state yes or no.",
#     "Are chronic atherosclerotic calcifications identified on the iliac arteries in this angiogram? Please state yes or no.",
#     "Is the catheter tip located in the superior mesenteric artery? Please state yes or no.",
#     "Is the gastroduodenal artery opacified in this angiogram? Please state yes or no.",
#     "Is the catheter tip located in the gastroduodenal artery or one of it's branches? Please state yes or no.",
#     "Is the gastroduadenal artery patent in this angiogram? Please state yes or no.",
#     "Is the dorsal pancreatic artery opacified in this angiogram? Please state yes or no."
# ]

POOL_CHOICES = ("max", "mean", "logsumexp")


# -----------------------------
# Utilities
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


def find_frames_dir(seq_dir: Path) -> Optional[Path]:
    for name in ("frames", "Frames"):
        d = seq_dir / name
        if d.exists() and d.is_dir():
            return d
    return None


def uniform_subsample(paths: List[Path], max_frames: Optional[int]) -> List[Path]:
    if max_frames is None or max_frames <= 0 or len(paths) <= max_frames:
        return paths
    idxs = torch.linspace(0, len(paths) - 1, steps=max_frames).long().tolist()
    return [paths[i] for i in idxs]


def read_metadata_key_value_csv(seq_dir: Path) -> Tuple[Optional[str], Optional[str], str]:
    """
    Reads seq_dir/metadata.csv where format is:
      Information,Value
      SOPInstanceUID,....
      AccessionNumber,....

    Returns: (AccessionNumber, SOPInstanceUID, status)
      status = "ok" if successful else reason
    """
    meta_path = seq_dir / "metadata.csv"
    if not meta_path.exists():
        return None, None, "missing_metadata_csv"

    try:
        df = pd.read_csv(meta_path)
    except Exception as e:
        return None, None, f"metadata_read_error:{type(e).__name__}"

    if df.empty or len(df.columns) < 2:
        return None, None, "metadata_empty_or_bad_format"

    # Find the two columns (case-insensitive)
    cols = [c.strip().lower() for c in df.columns]
    try:
        info_idx = cols.index("information")
        val_idx = cols.index("value")
    except ValueError:
        # fallback: assume first two cols are key/value
        info_idx, val_idx = 0, 1

    info_col = df.columns[info_idx]
    val_col = df.columns[val_idx]

    kv: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row.get(info_col, "")).strip()
        v = str(row.get(val_col, "")).strip()
        if k:
            kv[k] = v

    # Try common variants
    acc = kv.get("AccessionNumber") or kv.get("Accession Number") or kv.get("Accession")
    sop = kv.get("SOPInstanceUID") or kv.get("SOP Instance UID") or kv.get("SOPInstanceUid")

    if not acc or acc.lower() == "nan":
        return None, None, f"metadata_missing_accession(keys={list(kv.keys())[:20]})"
    if not sop or sop.lower() == "nan":
        return None, None, f"metadata_missing_sop(keys={list(kv.keys())[:20]})"

    return acc.strip(), sop.strip(), "ok"


def make_yes_no_hypotheses(question: str) -> Tuple[str, str]:
    q = question.strip().rstrip("?").lower()

    if "variant anatomy" in q:
        return (
            "Angiography demonstrates variant vascular anatomy.",
            "Angiography demonstrates no variant vascular anatomy.",
        )
    if "hemorrhage" in q or "extravasation" in q:
        return (
            "Angiography shows hemorrhage or contrast extravasation.",
            "Angiography shows no hemorrhage and no contrast extravasation.",
        )
    if "dissection" in q:
        return (
            "Angiography shows evidence of arterial or venous dissection.",
            "Angiography shows no evidence of arterial or venous dissection.",
        )
    if "stenosis" in q:
        return (
            "Angiography shows stenosis in a visualized vessel.",
            "Angiography shows no stenosis in any visualized vessel.",
        )
    if "stent" in q:
        return (
            "An endovascular stent is visible on angiography.",
            "No endovascular stent is visible on angiography.",
        )

    return (f"Yes. {question}", f"No. {question}")


# -----------------------------
# Model (must match your training architecture)
# -----------------------------
class PooledCLIP(nn.Module):
    """
    Inference version matching training:
    ViT + projection, BERT + projection, CLIP-style embedding space.
    Validation: each sequence dir is one "study" (single sequence), so we pool only over frames.
    """

    def __init__(self, vit_name: str, bert_name: str, embed_dim: int, frame_pooling: str):
        super().__init__()
        if frame_pooling not in POOL_CHOICES:
            raise ValueError(f"frame_pooling must be one of {POOL_CHOICES}, got {frame_pooling}")

        self.vit = ViTModel.from_pretrained(vit_name)
        self.bert = BertModel.from_pretrained(bert_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.bert_hidden = self.bert.config.hidden_size
        self.frame_pooling = frame_pooling

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

        # Exists in training checkpoints
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
        return F.normalize(self.text_proj(cls), dim=-1)  # (N, D)

    @torch.no_grad()
    def encode_sequence_from_frames(
        self,
        processor,
        frame_paths: List[Path],
        device: torch.device,
        frame_chunk_size: int,
        max_frames: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], str]:
        """
        Returns (img_emb, status)
          img_emb: (1, D) normalized projected embedding
        status: ok or reason
        """
        frame_paths = uniform_subsample(frame_paths, max_frames)
        if not frame_paths:
            return None, "no_frames"

        # Running pooling accumulator
        if self.frame_pooling == "max":
            running = torch.full((self.vit_hidden,), -1e9, device=device)
            updated = False
        elif self.frame_pooling == "mean":
            running = torch.zeros((self.vit_hidden,), device=device)
            count = 0
        else:  # logsumexp
            running = torch.full((self.vit_hidden,), -float("inf"), device=device)
            updated = False

        for i in range(0, len(frame_paths), frame_chunk_size):
            chunk = frame_paths[i : i + frame_chunk_size]
            imgs: List[Image.Image] = []
            for p in chunk:
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
                updated = True
            elif self.frame_pooling == "mean":
                running = running + feats.sum(dim=0)
                count += feats.size(0)
            else:
                running = torch.logaddexp(running, torch.logsumexp(feats, dim=0))
                updated = True

        if self.frame_pooling == "mean":
            if count <= 0:
                return None, "no_readable_frames"
            running = running / float(count)
        else:
            if not updated:
                return None, "no_readable_frames"

        emb = F.normalize(self.vision_proj(running.unsqueeze(0)), dim=-1)  # (1,D)
        return emb, "ok"


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (showing up to 20): {unexpected[:20]}")
    print("[INFO] Checkpoint loaded.")


def run(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    processor = get_vit_processor(args.vit_name)
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    frame_pooling = args.frame_pooling if args.frame_pooling else args.pooling

    model = PooledCLIP(
        vit_name=args.vit_name,
        bert_name=args.bert_name,
        embed_dim=args.embed_dim,
        frame_pooling=frame_pooling,
    ).to(device)
    model.eval()

    load_checkpoint(model, Path(args.checkpoint), device=device)

    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    error_rows = []
    n_total = 0
    n_ok = 0
    skip_counts: Dict[str, int] = {}

    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None

    # Stream-write predictions so file is never "empty" if any success happens
    with open(out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["AccessionNumber", "SOPInstanceUID", "Question", "Answer"])

        seq_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        for seq_dir in tqdm(seq_dirs, desc="Sequences", dynamic_ncols=True):
            n_total += 1

            acc, sop, meta_status = read_metadata_key_value_csv(seq_dir)
            if meta_status != "ok":
                skip_counts[meta_status] = skip_counts.get(meta_status, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": meta_status, "details": ""})
                continue

            frames_dir = find_frames_dir(seq_dir)
            if frames_dir is None:
                s = "missing_frames_dir"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": ""})
                continue

            frame_paths = list_images(frames_dir)
            if not frame_paths:
                s = "no_frame_files"
                skip_counts[s] = skip_counts.get(s, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": s, "details": str(frames_dir)})
                continue

            img_emb, emb_status = model.encode_sequence_from_frames(
                processor=processor,
                frame_paths=frame_paths,
                device=device,
                frame_chunk_size=args.frame_chunk_size,
                max_frames=max_frames,
            )
            if emb_status != "ok" or img_emb is None:
                skip_counts[emb_status] = skip_counts.get(emb_status, 0) + 1
                error_rows.append({"seq_dir": str(seq_dir), "status": emb_status, "details": ""})
                continue

            # Answer questions
            for q in QA_QUESTIONS:
                yes_h, no_h = make_yes_no_hypotheses(q)
                txt_emb = model.encode_text(tokenizer, [yes_h, no_h], device=device)  # (2,D)
                sims = (img_emb @ txt_emb.t()).squeeze(0)  # (2,)
                pred = "YES" if sims[0].item() > sims[1].item() else "NO"
                writer.writerow([acc, sop, q, pred])

            n_ok += 1

    print(f"[INFO] Wrote predictions to: {out_csv}")

    print("\n[SUMMARY]")
    print(f"  Total sequence dirs seen: {n_total}")
    print(f"  Successfully predicted:  {n_ok}")
    if skip_counts:
        print("  Skipped counts:")
        for k, v in sorted(skip_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {k}: {v}")

    if args.error_csv:
        err_path = Path(args.error_csv)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(error_rows, columns=["seq_dir", "status", "details"]).to_csv(err_path, index=False)
        print(f"[INFO] Wrote error log to: {err_path}")


def build_argparser():
    ap = argparse.ArgumentParser(description="Run CLIP-style binary QA on sequence-level validation data.")

    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--error_csv", default="", type=str, help="Optional CSV logging skip/errors per sequence dir.")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--vit_name", default="google/vit-base-patch16-224-in21k", type=str)
    ap.add_argument("--bert_name", default="bert-base-uncased", type=str)
    ap.add_argument("--embed_dim", default=256, type=int)

    ap.add_argument("--pooling", default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", default="", choices=("",) + POOL_CHOICES)

    ap.add_argument("--frame_chunk_size", default=64, type=int)
    ap.add_argument("--max_frames", default=0, type=int)

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
