#!/usr/bin/env python3
"""
custom_framework_train.py

AngioVision fine-tuning: (Study) frames -> ViT -> pool -> visual embedding
Align visual embedding with report text embedding (BERT) using CLIP-style contrastive loss.

Directory layout assumed:
  BASE/<Anon Acc #>/<SOPInstanceUID>/frames/<image files>

NEW (requested):
- Always creates a loss CSV in --out_dir whenever the script is run.
- Loss CSV filename encodes ONLY:
    epochs, batch_size, max_sequences_per_study
  in "_" separated format:
    <epochs>_<batch_size>_<max_sequences_per_study>_loss.csv
  where max_sequences_per_study is "None" if not provided.

Example:
  3_2_None_loss.csv
  5_4_10_loss.csv
"""

import re
import ast
import math
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from transformers import ViTModel, BertModel, BertTokenizer


# ------------------------------------------------------------
# Transformers image processor compatibility
# ------------------------------------------------------------
try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:
    _ViTProcessor = None

try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:
    _ViTFeatureExtractor = None


# ------------------------------------------------------------
# Robust SOPInstanceUIDs parser
# ------------------------------------------------------------
def parse_sop_instance_uids(val) -> List[str]:
    """
    Handles:
    1) CSV-quoted comma-separated string: "uid1,uid2,uid3"
    2) python literal list/tuple: "['uid1','uid2']" or "('uid1','uid2')"
    3) NaN/None/empty safely
    """
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []

    s = str(val).strip()

    # Strip wrapping quotes if present
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()

    if not s:
        return []

    # Try python-literal if it looks like one
    if s[0] in "[(":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass  # fall back to comma split

    # Comma-separated
    return [tok.strip() for tok in re.split(r"\s*,\s*", s) if tok.strip()]


# ------------------------------------------------------------
# Frame discovery utilities
# ------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images_in_dir(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)


def find_frame_files_for_sop(base_frames_dir: Path, acc: str, sop_uid: str) -> List[Path]:
    """
    Expected layout:
      BASE/<Anon Acc #>/<SOPInstanceUID>/frames/*.img

    Safe search order:
      1) BASE/acc/sop/frames/*.img
      2) BASE/acc/sop/*.img
      3) BASE/acc/sop/<one-level-subdir>/*.img  (fallback)
      4) BASE/*/acc/sop/frames/*.img            (constrained fallback)
    """
    acc = str(acc).strip()
    sop_uid = str(sop_uid).strip()

    # 1) expected path
    sop_dir = base_frames_dir / acc / sop_uid
    frames_dir = sop_dir / "frames"
    imgs = _list_images_in_dir(frames_dir)
    if imgs:
        return imgs

    # 2) sometimes images are directly under sop dir
    imgs = _list_images_in_dir(sop_dir)
    if imgs:
        return imgs

    # 3) one-level nested fallback
    if sop_dir.exists() and sop_dir.is_dir():
        nested_imgs: List[Path] = []
        for child in sop_dir.iterdir():
            if child.is_dir():
                nested_imgs.extend(_list_images_in_dir(child))
        if nested_imgs:
            return sorted(nested_imgs)

    # 4) constrained glob fallback
    try:
        pattern = f"*/{acc}/{sop_uid}/frames"
        for candidate in base_frames_dir.glob(pattern):
            imgs = _list_images_in_dir(candidate)
            if imgs:
                return imgs
    except Exception:
        pass

    return []


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class StudyDataset(Dataset):
    """
    Each item returns:
      - sequences: List[List[Path]]  (list of sequences; each sequence is list of frame image paths)
      - text: report string

    Meta CSV expected columns:
      - "Anon Acc #"
      - "SOPInstanceUIDs"
    Reports CSV expected columns:
      - "Anon Acc #"
      - "radrpt" (or override)
    """

    def __init__(
        self,
        meta_csv: Path,
        reports_csv: Path,
        base_frames_dir: Path,
        report_text_col: str = "radrpt",
        anon_col: str = "Anon Acc #",
        sop_col: str = "SOPInstanceUIDs",
        min_frames_per_sequence: int = 1,
        max_sequences_per_study: Optional[int] = None,
        drop_missing_reports: bool = True,
    ):
        self.meta = pd.read_csv(meta_csv)
        self.reports = pd.read_csv(reports_csv)

        self.base_frames_dir = base_frames_dir
        self.report_text_col = report_text_col
        self.anon_col = anon_col
        self.sop_col = sop_col
        self.min_frames_per_sequence = min_frames_per_sequence
        self.max_sequences_per_study = max_sequences_per_study
        self.drop_missing_reports = drop_missing_reports

        # map acc -> report text
        self.report_map: Dict[str, str] = {}
        for _, r in self.reports.iterrows():
            acc = str(r.get(self.anon_col, "")).strip()
            txt = r.get(self.report_text_col, "")
            if isinstance(txt, float) and pd.isna(txt):
                txt = ""
            self.report_map[acc] = str(txt)

        if self.drop_missing_reports:
            keep = []
            for i, row in self.meta.iterrows():
                acc = str(row.get(self.anon_col, "")).strip()
                if acc in self.report_map and self.report_map[acc].strip():
                    keep.append(i)
            self.meta = self.meta.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        acc = str(row.get(self.anon_col, "")).strip()

        sop_uids = parse_sop_instance_uids(row.get(self.sop_col, ""))

        if self.max_sequences_per_study is not None:
            sop_uids = sop_uids[: self.max_sequences_per_study]

        sequences: List[List[Path]] = []
        for sop in sop_uids:
            frame_files = find_frame_files_for_sop(self.base_frames_dir, acc, sop)
            if len(frame_files) >= self.min_frames_per_sequence:
                sequences.append(frame_files)

        text = self.report_map.get(acc, "")

        return {"acc": acc, "sequences": sequences, "text": text}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [b for b in batch if b["text"] and isinstance(b["sequences"], list) and len(b["sequences"]) > 0]
    return {
        "acc": [b["acc"] for b in kept],
        "sequences": [b["sequences"] for b in kept],
        "text": [b["text"] for b in kept],
    }


# ------------------------------------------------------------
# Pooling helpers
# ------------------------------------------------------------
POOL_CHOICES = ("max", "mean", "logsumexp")


def pool_stack(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x: (N, D) tensor -> (D,)
    """
    if x.ndim != 2:
        raise ValueError(f"pool_stack expects (N,D), got {tuple(x.shape)}")
    if x.size(0) == 0:
        raise ValueError("pool_stack got empty N dimension")

    if mode == "max":
        return x.max(dim=0).values
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "logsumexp":
        return torch.logsumexp(x, dim=0)

    raise ValueError(f"Unknown pooling mode: {mode}. Choose from {POOL_CHOICES}.")


def get_vit_processor(vit_name: str):
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(vit_name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(vit_name)
    raise ImportError("Neither ViTImageProcessor nor ViTFeatureExtractor is available in transformers.")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class PooledCLIP(nn.Module):
    def __init__(
        self,
        vit_name: str,
        bert_name: str,
        embed_dim: int = 256,
        freeze_vision: bool = False,
        freeze_text: bool = False,
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

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        if freeze_vision:
            for p in self.vit.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.bert.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _init_running(self, device, d: int, mode: str):
        if mode == "max":
            return torch.full((d,), -1e9, device=device), None
        if mode == "mean":
            return torch.zeros((d,), device=device), torch.tensor(0, device=device)
        if mode == "logsumexp":
            return torch.full((d,), -float("inf"), device=device), None
        raise ValueError(f"Unknown pooling mode: {mode}")

    def _update_running(self, running, aux, chunk_feats: torch.Tensor, mode: str):
        if mode == "max":
            running = torch.maximum(running, chunk_feats.max(dim=0).values)
            return running, aux

        if mode == "mean":
            running = running + chunk_feats.sum(dim=0)
            aux = aux + chunk_feats.size(0)
            return running, aux

        if mode == "logsumexp":
            running = torch.logaddexp(running, torch.logsumexp(chunk_feats, dim=0))
            return running, aux

        raise ValueError(f"Unknown pooling mode: {mode}")

    def _finalize_running(self, running, aux, mode: str) -> torch.Tensor:
        if mode == "max":
            return running
        if mode == "mean":
            count = aux.item() if hasattr(aux, "item") else int(aux)
            return running / float(count) if count > 0 else running
        if mode == "logsumexp":
            return running
        raise ValueError(f"Unknown pooling mode: {mode}")

    def encode_frames_pooled(
        self,
        frame_images: List[Image.Image],
        processor,
        device: torch.device,
        chunk_size: int = 16,
        pooling: str = "max",
    ) -> torch.Tensor:
        running, aux = self._init_running(device, self.vit_hidden, pooling)

        for i in range(0, len(frame_images), chunk_size):
            chunk = frame_images[i : i + chunk_size]
            inputs = processor(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            out = self.vit(pixel_values=pixel_values)
            frame_emb = out.last_hidden_state[:, 0, :]  # CLS

            running, aux = self._update_running(running, aux, frame_emb, pooling)

        return self._finalize_running(running, aux, pooling)

    def forward(
        self,
        batch_sequences: List[List[List[Path]]],  # batch -> sequences -> frames(paths)
        texts: List[str],
        processor,
        tokenizer,
        device: torch.device,
        frame_chunk_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(batch_sequences)
        assert B == len(texts)

        study_visuals: List[torch.Tensor] = []
        for sequences in batch_sequences:
            seq_feats: List[torch.Tensor] = []
            for frame_paths in sequences:
                frames: List[Image.Image] = []
                for p in frame_paths:
                    try:
                        frames.append(Image.open(p).convert("RGB"))
                    except Exception:
                        continue

                if not frames:
                    continue

                seq_feat = self.encode_frames_pooled(
                    frames,
                    processor=processor,
                    device=device,
                    chunk_size=frame_chunk_size,
                    pooling=self.frame_pooling,
                )
                seq_feats.append(seq_feat)

            if not seq_feats:
                study_feat = torch.zeros(self.vit_hidden, device=device)
            else:
                seq_stack = torch.stack(seq_feats, dim=0)
                study_feat = pool_stack(seq_stack, self.sequence_pooling)

            study_visuals.append(study_feat)

        study_visuals = torch.stack(study_visuals, dim=0)
        image_embeds = F.normalize(self.vision_proj(study_visuals), dim=-1)

        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        tout = self.bert(**tok)
        tcls = tout.last_hidden_state[:, 0, :]
        text_embeds = F.normalize(self.text_proj(tcls), dim=-1)

        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits = logit_scale * (image_embeds @ text_embeds.t())

        return image_embeds, text_embeds, logits


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    B = logits.size(0)
    targets = torch.arange(B, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


# ------------------------------------------------------------
# Loss CSV naming + logging (NEW)
# ------------------------------------------------------------
def _loss_csv_name(epochs: int, batch_size: int, max_sequences_per_study) -> str:
    ms = "None" if max_sequences_per_study is None else str(max_sequences_per_study)
    return f"{epochs}_{batch_size}_{ms}_loss.csv"


def _init_loss_logger(out_dir: Path, epochs: int, batch_size: int, max_sequences_per_study) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / _loss_csv_name(epochs, batch_size, max_sequences_per_study)

    # Always ensure it exists + has header (even if training later skips everything)
    if (not log_path.exists()) or (log_path.stat().st_size == 0):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "step", "batch_size", "loss", "loss_ema"])

    return log_path


def _append_loss_row(log_path: Path, epoch: int, step: int, batch_size: int, loss_val: float, loss_ema: float):
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, step, batch_size, f"{loss_val:.8f}", f"{loss_ema:.8f}"])


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    pooling_default = args.pooling
    frame_pooling = args.frame_pooling if args.frame_pooling else pooling_default
    sequence_pooling = args.sequence_pooling if args.sequence_pooling else pooling_default

    print(f"[INFO] pooling: default={pooling_default}, frame={frame_pooling}, sequence={sequence_pooling}")

    dataset = StudyDataset(
        meta_csv=Path(args.meta_csv),
        reports_csv=Path(args.reports_csv),
        base_frames_dir=Path(args.base_frames_dir),
        report_text_col=args.report_text_col,
        anon_col=args.anon_col,
        sop_col=args.sop_col,
        min_frames_per_sequence=args.min_frames_per_sequence,
        max_sequences_per_study=args.max_sequences_per_study,
        drop_missing_reports=not args.keep_missing_reports,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    processor = get_vit_processor(args.vit_name)
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    model = PooledCLIP(
        vit_name=args.vit_name,
        bert_name=args.bert_name,
        embed_dim=args.embed_dim,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
        frame_pooling=frame_pooling,
        sequence_pooling=sequence_pooling,
    ).to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ALWAYS create the loss CSV at script start
    loss_log_path = _init_loss_logger(
        out_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_sequences_per_study=args.max_sequences_per_study,
    )
    print(f"[INFO] Loss CSV: {loss_log_path}")

    model.train()
    step = 0

    ema = None
    ema_beta = 0.98

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for batch in pbar:
            if len(batch["text"]) == 0:
                # nothing kept by collate
                continue

            _, _, logits = model(
                batch_sequences=batch["sequences"],
                texts=batch["text"],
                processor=processor,
                tokenizer=tokenizer,
                device=device,
                frame_chunk_size=args.frame_chunk_size,
            )

            loss = clip_loss(logits)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            step += 1
            loss_val = float(loss.detach().cpu().item())

            if ema is None:
                ema = loss_val
            else:
                ema = ema_beta * ema + (1 - ema_beta) * loss_val

            _append_loss_row(
                log_path=loss_log_path,
                epoch=epoch + 1,
                step=step,
                batch_size=len(batch["text"]),
                loss_val=loss_val,
                loss_ema=float(ema),
            )

            pbar.set_postfix(loss=loss_val, loss_ema=float(ema), bs=len(batch["text"]))

            if args.save_every > 0 and step % args.save_every == 0:
                torch.save(
                    {"step": step, "epoch": epoch + 1, "model_state": model.state_dict(), "opt_state": opt.state_dict()},
                    out_dir / f"checkpoint_step_{step}.pt",
                )

        torch.save(
            {"step": step, "epoch": epoch + 1, "model_state": model.state_dict(), "opt_state": opt.state_dict()},
            out_dir / f"checkpoint_epoch_{epoch+1}.pt",
        )

    print("[INFO] Training complete.")
    print(f"[INFO] Loss CSV saved at: {loss_log_path}")


def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--reports_csv", type=str, required=True)
    ap.add_argument("--base_frames_dir", type=str, required=True)

    ap.add_argument("--report_text_col", type=str, default="radrpt")
    ap.add_argument("--anon_col", type=str, default="Anon Acc #")
    ap.add_argument("--sop_col", type=str, default="SOPInstanceUIDs")

    ap.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--bert_name", type=str, default="bert-base-uncased")

    ap.add_argument("--embed_dim", type=int, default=256)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--frame_chunk_size", type=int, default=16)

    ap.add_argument("--min_frames_per_sequence", type=int, default=1)
    ap.add_argument("--max_sequences_per_study", type=int, default=None)

    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")

    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--out_dir", type=str, default="./checkpoints")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--keep_missing_reports", action="store_true")

    ap.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=POOL_CHOICES,
        help="Default pooling mode used for both frame and sequence pooling unless overridden.",
    )
    ap.add_argument(
        "--frame_pooling",
        type=str,
        default=None,
        choices=POOL_CHOICES,
        help="Pooling across frames within a SOP/sequence. If omitted, uses --pooling.",
    )
    ap.add_argument(
        "--sequence_pooling",
        type=str,
        default=None,
        choices=POOL_CHOICES,
        help="Pooling across sequences/SOPs within a study. If omitted, uses --pooling.",
    )

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
