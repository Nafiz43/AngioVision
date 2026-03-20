#!/usr/bin/env python3
"""
custom_framework_train.py

AngioVision fine-tuning: (Study) frames -> ViT -> pool -> visual embedding
Align visual embedding with report text embedding (BERT) using CLIP-style contrastive loss.
"""

from __future__ import annotations

import re
import ast
import math
import argparse
import csv
import subprocess
import sys
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
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []

    s = str(val).strip()

    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()

    if not s:
        return []

    if s[0] in "[(":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

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
    acc = str(acc).strip()
    sop_uid = str(sop_uid).strip()

    sop_dir = base_frames_dir / acc / sop_uid
    frames_dir = sop_dir / "frames"
    imgs = _list_images_in_dir(frames_dir)
    if imgs:
        return imgs

    imgs = _list_images_in_dir(sop_dir)
    if imgs:
        return imgs

    if sop_dir.exists() and sop_dir.is_dir():
        nested_imgs: List[Path] = []
        for child in sop_dir.iterdir():
            if child.is_dir():
                nested_imgs.extend(_list_images_in_dir(child))
        if nested_imgs:
            return sorted(nested_imgs)

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
        max_frames_per_sequence: Optional[int] = None,
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
        self.max_frames_per_sequence = max_frames_per_sequence
        self.drop_missing_reports = drop_missing_reports

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

            if self.max_frames_per_sequence is not None and len(frame_files) > self.max_frames_per_sequence:
                idxs = torch.linspace(0, len(frame_files) - 1, steps=self.max_frames_per_sequence).long().tolist()
                frame_files = [frame_files[i] for i in idxs]

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
        vit_trainable_blocks: int = 3,
        vit_unfreeze_patch_embed: bool = False,
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
        else:
            self._configure_partial_vit_finetuning(
                vit_trainable_blocks=vit_trainable_blocks,
                vit_unfreeze_patch_embed=vit_unfreeze_patch_embed,
            )

        if freeze_text:
            for p in self.bert.parameters():
                p.requires_grad = False

    def _configure_partial_vit_finetuning(
        self,
        vit_trainable_blocks: int,
        vit_unfreeze_patch_embed: bool,
    ) -> None:
        # Freeze everything first
        for p in self.vit.parameters():
            p.requires_grad = False

        layers = None
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            layers = self.vit.encoder.layer

        if layers is None:
            raise RuntimeError("Could not locate ViT encoder layers at self.vit.encoder.layer")

        num_layers = len(layers)
        n = max(0, min(int(vit_trainable_blocks), num_layers))

        # Unfreeze last N transformer blocks
        if n > 0:
            for block in layers[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

        # Unfreeze final layernorm
        if hasattr(self.vit, "layernorm") and self.vit.layernorm is not None:
            for p in self.vit.layernorm.parameters():
                p.requires_grad = True

        # Optional pooler
        if hasattr(self.vit, "pooler") and self.vit.pooler is not None:
            for p in self.vit.pooler.parameters():
                p.requires_grad = True

        # Optional patch embeddings
        if vit_unfreeze_patch_embed:
            if hasattr(self.vit, "embeddings") and self.vit.embeddings is not None:
                for p in self.vit.embeddings.parameters():
                    p.requires_grad = True

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

    def _vit_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.vit.parameters())

    def _bert_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.bert.parameters())

    def encode_framepaths_pooled(
        self,
        frame_paths: List[Path],
        processor,
        device: torch.device,
        chunk_size: int = 16,
        pooling: str = "max",
        vit_image_size: Optional[int] = None,
    ) -> torch.Tensor:
        running, aux = self._init_running(device, self.vit_hidden, pooling)

        vit_ctx = torch.no_grad() if self._vit_is_frozen() else torch.enable_grad()

        for i in range(0, len(frame_paths), chunk_size):
            chunk_paths = frame_paths[i : i + chunk_size]

            chunk_imgs: List[Image.Image] = []
            for p in chunk_paths:
                try:
                    chunk_imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue

            if not chunk_imgs:
                continue

            if vit_image_size is not None:
                inputs = processor(
                    images=chunk_imgs,
                    return_tensors="pt",
                    size={"height": vit_image_size, "width": vit_image_size},
                )
            else:
                inputs = processor(images=chunk_imgs, return_tensors="pt")

            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            with vit_ctx:
                out = self.vit(pixel_values=pixel_values)
                frame_emb = out.last_hidden_state[:, 0, :]
                running, aux = self._update_running(running, aux, frame_emb, pooling)

            del pixel_values, out, frame_emb, inputs, chunk_imgs

        return self._finalize_running(running, aux, pooling)

    def forward(
        self,
        batch_sequences: List[List[List[Path]]],
        texts: List[str],
        processor,
        tokenizer,
        device: torch.device,
        frame_chunk_size: int = 16,
        vit_image_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(batch_sequences)
        assert B == len(texts)

        study_visuals: List[torch.Tensor] = []
        for sequences in batch_sequences:
            seq_feats: List[torch.Tensor] = []
            for frame_paths in sequences:
                if not frame_paths:
                    continue

                seq_feat = self.encode_framepaths_pooled(
                    frame_paths=frame_paths,
                    processor=processor,
                    device=device,
                    chunk_size=frame_chunk_size,
                    pooling=self.frame_pooling,
                    vit_image_size=vit_image_size,
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

        tok = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        bert_ctx = torch.no_grad() if self._bert_is_frozen() else torch.enable_grad()
        with bert_ctx:
            tout = self.bert(**tok)
            tcls = tout.last_hidden_state[:, 0, :]

        text_embeds = F.normalize(self.text_proj(tcls), dim=-1)

        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return image_embeds, text_embeds, logit_scale


def clip_loss_chunked(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    chunk: int = 4,
) -> torch.Tensor:
    B = image_embeds.size(0)
    device = image_embeds.device
    targets = torch.arange(B, device=device)

    loss_i_sum = 0.0
    for i0 in range(0, B, chunk):
        i1 = min(B, i0 + chunk)
        logits_part = logit_scale * (image_embeds[i0:i1] @ text_embeds.t())
        loss_i_sum = loss_i_sum + F.cross_entropy(logits_part, targets[i0:i1], reduction="sum")
        del logits_part
    loss_i = loss_i_sum / B

    loss_t_sum = 0.0
    for t0 in range(0, B, chunk):
        t1 = min(B, t0 + chunk)
        logits_part = logit_scale * (text_embeds[t0:t1] @ image_embeds.t())
        loss_t_sum = loss_t_sum + F.cross_entropy(logits_part, targets[t0:t1], reduction="sum")
        del logits_part
    loss_t = loss_t_sum / B

    return 0.5 * (loss_i + loss_t)


# ------------------------------------------------------------
# Run-dir + loss CSV naming/logging
# ------------------------------------------------------------
def _run_dir_name(
    epochs: int,
    batch_size: int,
    max_sequences_per_study: Optional[int],
    max_frames_per_sequence: Optional[int],
) -> str:
    ms = "None" if max_sequences_per_study is None else str(max_sequences_per_study)
    mf = "None" if max_frames_per_sequence is None else str(max_frames_per_sequence)
    return f"{epochs}_{batch_size}_{ms}_{mf}"


def _loss_csv_name(
    epochs: int,
    batch_size: int,
    max_sequences_per_study: Optional[int],
    max_frames_per_sequence: Optional[int],
) -> str:
    ms = "None" if max_sequences_per_study is None else str(max_sequences_per_study)
    mf = "None" if max_frames_per_sequence is None else str(max_frames_per_sequence)
    return f"{epochs}_{batch_size}_{ms}_{mf}_loss.csv"


def _init_loss_logger(
    run_dir: Path,
    epochs: int,
    batch_size: int,
    max_sequences_per_study: Optional[int],
    max_frames_per_sequence: Optional[int],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / _loss_csv_name(epochs, batch_size, max_sequences_per_study, max_frames_per_sequence)

    if (not log_path.exists()) or (log_path.stat().st_size == 0):
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "step", "batch_size", "loss", "loss_ema"])

    return log_path


def _append_loss_row(log_path: Path, epoch: int, step: int, batch_size: int, loss_val: float, loss_ema: float):
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, step, batch_size, f"{loss_val:.8f}", f"{loss_ema:.8f}"])


def _checkpoint_payload(model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int) -> Dict[str, Any]:
    return {
        "step": step,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
    }


def _save_last_checkpoint(run_dir: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int) -> Path:
    ckpt_path = run_dir / "last.pt"
    torch.save(_checkpoint_payload(model, opt, step, epoch), ckpt_path)
    return ckpt_path


def _save_epoch_checkpoint(run_dir: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int) -> Path:
    ckpt_path = run_dir / f"epoch_{epoch}.pt"
    torch.save(_checkpoint_payload(model, opt, step, epoch), ckpt_path)
    return ckpt_path


# ------------------------------------------------------------
# Subprocess runner helpers
# ------------------------------------------------------------
def _run_subprocess(cmd: List[str]) -> None:
    print("\n[INFO] Running subprocess:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_post_training_pipeline(args, run_name: str, run_dir: Path, loss_csv: Path) -> None:
    val_data_dir = Path(args.val_data_dir)
    if not val_data_dir.exists():
        raise SystemExit(f"[ERROR] Validation data dir does not exist: {val_data_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    if not loss_csv.exists():
        raise SystemExit(f"[ERROR] Loss CSV not found: {loss_csv}")

    pred_csv = output_dir / f"clip_binary_qa_predictions_{run_name}.csv"
    err_csv = output_dir / f"clip_binary_qa_errors_{run_name}.csv"

    validate_device = args.validate_device
    if validate_device is None:
        validate_device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    validate_cmd = [
        sys.executable,
        args.validate_script,
        "--checkpoint",
        str(ckpt_path),
        "--data_dir",
        str(val_data_dir),
        "--out_csv",
        str(pred_csv),
        "--error_csv",
        str(err_csv),
        "--device",
        str(validate_device),
        "--frame_chunk_size",
        str(args.frame_chunk_size),
        "--pooling",
        str(args.pooling),
    ]
    _run_subprocess(validate_cmd)

    score_cmd = [
        sys.executable,
        args.calculate_score_script,
        "--pred_path",
        str(pred_csv),
    ]
    _run_subprocess(score_cmd)

    plot_cmd = [
        sys.executable,
        args.plot_loss_script,
        "--source_path",
        str(loss_csv),
    ]
    _run_subprocess(plot_cmd)

    print("\n[INFO] Post-training pipeline complete.")
    print(f"[INFO] Predictions: {pred_csv}")
    print(f"[INFO] Errors:      {err_csv}")
    print(f"[INFO] Loss CSV:    {loss_csv}")


# ------------------------------------------------------------
# Trainable parameter helpers
# ------------------------------------------------------------
def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_all_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def print_trainable_summary(model: "PooledCLIP") -> None:
    total = count_all_params(model)
    trainable = count_trainable_params(model)

    print(f"[INFO] Total params:      {total:,}")
    print(f"[INFO] Trainable params:  {trainable:,}")
    print(f"[INFO] Frozen params:     {total - trainable:,}")

    if hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layer"):
        n_layers = len(model.vit.encoder.layer)
        trainable_blocks = []
        for i, block in enumerate(model.vit.encoder.layer):
            block_trainable = any(p.requires_grad for p in block.parameters())
            if block_trainable:
                trainable_blocks.append(i)
        print(f"[INFO] ViT trainable encoder blocks: {trainable_blocks} / total={n_layers}")

    vit_embeddings_trainable = (
        hasattr(model.vit, "embeddings")
        and model.vit.embeddings is not None
        and any(p.requires_grad for p in model.vit.embeddings.parameters())
    )
    print(f"[INFO] ViT embeddings trainable: {vit_embeddings_trainable}")

    vit_ln_trainable = (
        hasattr(model.vit, "layernorm")
        and model.vit.layernorm is not None
        and any(p.requires_grad for p in model.vit.layernorm.parameters())
    )
    print(f"[INFO] ViT final layernorm trainable: {vit_ln_trainable}")
    print(f"[INFO] BERT trainable: {any(p.requires_grad for p in model.bert.parameters())}")
    print(f"[INFO] vision_proj trainable: {any(p.requires_grad for p in model.vision_proj.parameters())}")
    print(f"[INFO] text_proj trainable: {any(p.requires_grad for p in model.text_proj.parameters())}")


def build_optimizer(model: "PooledCLIP", args) -> torch.optim.Optimizer:
    vision_backbone_params = [p for p in model.vit.parameters() if p.requires_grad]
    text_backbone_params = [p for p in model.bert.parameters() if p.requires_grad]

    head_params = []
    head_params.extend([p for p in model.vision_proj.parameters() if p.requires_grad])
    head_params.extend([p for p in model.text_proj.parameters() if p.requires_grad])
    if model.logit_scale.requires_grad:
        head_params.append(model.logit_scale)

    param_groups = []

    if vision_backbone_params:
        param_groups.append(
            {
                "params": vision_backbone_params,
                "lr": args.vision_backbone_lr,
                "weight_decay": args.weight_decay,
            }
        )

    if text_backbone_params:
        param_groups.append(
            {
                "params": text_backbone_params,
                "lr": args.text_lr,
                "weight_decay": args.weight_decay,
            }
        )

    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": args.head_lr,
                "weight_decay": args.weight_decay,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    print("[INFO] Optimizer parameter groups:")
    if vision_backbone_params:
        print(f"       vision backbone: {sum(p.numel() for p in vision_backbone_params):,} params, lr={args.vision_backbone_lr}")
    if text_backbone_params:
        print(f"       text backbone:   {sum(p.numel() for p in text_backbone_params):,} params, lr={args.text_lr}")
    if head_params:
        print(f"       projection/head: {sum(p.numel() for p in head_params):,} params, lr={args.head_lr}")

    return torch.optim.AdamW(param_groups)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pooling_default = args.pooling
    frame_pooling = args.frame_pooling if args.frame_pooling else pooling_default
    sequence_pooling = args.sequence_pooling if args.sequence_pooling else pooling_default
    print(f"[INFO] pooling: default={pooling_default}, frame={frame_pooling}, sequence={sequence_pooling}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = _run_dir_name(
        args.epochs,
        args.batch_size,
        args.max_sequences_per_study,
        args.max_frames_per_sequence,
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / "last.pt"
    loss_log_path = run_dir / _loss_csv_name(
        args.epochs,
        args.batch_size,
        args.max_sequences_per_study,
        args.max_frames_per_sequence,
    )

    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Loss CSV: {loss_log_path}")
    print(f"[INFO] Rolling checkpoint path: {ckpt_path}")
    print(f"[INFO] Per-epoch checkpoints will be stored as: {run_dir / 'epoch_<N>.pt'}")

    if ckpt_path.exists():
        print(f"[INFO] Existing checkpoint found at: {ckpt_path}")
        print("[INFO] Skipping training and using existing checkpoint.")
        run_post_training_pipeline(
            args=args,
            run_name=run_name,
            run_dir=run_dir,
            loss_csv=loss_log_path,
        )
        return

    dataset = StudyDataset(
        meta_csv=Path(args.meta_csv),
        reports_csv=Path(args.reports_csv),
        base_frames_dir=Path(args.base_frames_dir),
        report_text_col=args.report_text_col,
        anon_col=args.anon_col,
        sop_col=args.sop_col,
        min_frames_per_sequence=args.min_frames_per_sequence,
        max_sequences_per_study=args.max_sequences_per_study,
        max_frames_per_sequence=args.max_frames_per_sequence,
        drop_missing_reports=not args.keep_missing_reports,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
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
        vit_trainable_blocks=args.vit_trainable_blocks,
        vit_unfreeze_patch_embed=args.vit_unfreeze_patch_embed,
    ).to(device)

    print_trainable_summary(model)

    if args.vit_grad_ckpt and (not args.freeze_vision):
        if hasattr(model.vit, "gradient_checkpointing_enable"):
            model.vit.gradient_checkpointing_enable()
        if hasattr(model.vit.config, "use_cache"):
            model.vit.config.use_cache = False
        print("[INFO] ViT gradient checkpointing enabled.")

    opt = build_optimizer(model, args)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[INFO] AMP enabled.")

    loss_log_path = _init_loss_logger(
        run_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_sequences_per_study=args.max_sequences_per_study,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    print(f"[INFO] Loss CSV initialized at: {loss_log_path}")

    model.train()
    step = 0
    ema = None
    ema_beta = 0.98

    grad_accum = max(1, int(args.grad_accum))
    if grad_accum > 1:
        print(f"[INFO] Gradient accumulation: {grad_accum} steps (effective batch ~= batch_size * grad_accum)")

    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for batch in pbar:
            if len(batch["text"]) == 0:
                continue

            with torch.cuda.amp.autocast(enabled=use_amp):
                image_embeds, text_embeds, logit_scale = model(
                    batch_sequences=batch["sequences"],
                    texts=batch["text"],
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    frame_chunk_size=args.frame_chunk_size,
                    vit_image_size=args.vit_image_size,
                )

                loss = clip_loss_chunked(
                    image_embeds=image_embeds,
                    text_embeds=text_embeds,
                    logit_scale=logit_scale,
                    chunk=args.logits_chunk,
                )
                loss_to_backprop = loss / float(grad_accum)

            scaler.scale(loss_to_backprop).backward()

            do_step = ((step + 1) % grad_accum == 0)

            if do_step:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

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

            del loss, loss_to_backprop, image_embeds, text_embeds, logit_scale
            if args.empty_cache_each_step and device.type == "cuda":
                torch.cuda.empty_cache()

        last_ckpt = _save_last_checkpoint(
            run_dir=run_dir,
            model=model,
            opt=opt,
            step=step,
            epoch=epoch + 1,
        )
        epoch_ckpt = _save_epoch_checkpoint(
            run_dir=run_dir,
            model=model,
            opt=opt,
            step=step,
            epoch=epoch + 1,
        )

        print(f"[INFO] Saved rolling checkpoint: {last_ckpt}")
        print(f"[INFO] Saved epoch checkpoint:   {epoch_ckpt}")

    print("[INFO] Training complete.")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Loss CSV saved at: {loss_log_path}")
    print(f"[INFO] Final rolling checkpoint saved at: {ckpt_path}")
    print(f"[INFO] Per-epoch checkpoints saved under: {run_dir}")

    run_post_training_pipeline(args=args, run_name=run_name, run_dir=run_dir, loss_csv=loss_log_path)


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

    # Separate LRs for stability/efficiency
    ap.add_argument("--vision_backbone_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--text_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--frame_chunk_size", type=int, default=16)

    ap.add_argument("--min_frames_per_sequence", type=int, default=1)
    ap.add_argument("--max_sequences_per_study", type=int, default=None)

    ap.add_argument(
        "--max_frames_per_sequence",
        type=int,
        default=None,
        help="If set, uniformly sample at most this many frames per SOP/sequence to reduce memory/time.",
    )

    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")

    # New: partial ViT fine-tuning controls
    ap.add_argument(
        "--vit_trainable_blocks",
        type=int,
        default=3,
        help="Number of final ViT encoder blocks to unfreeze/train. Ignored if --freeze_vision is set.",
    )
    ap.add_argument(
        "--vit_unfreeze_patch_embed",
        action="store_true",
        help="Also unfreeze ViT patch/image embeddings. Usually keep this OFF initially.",
    )

    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--out_dir", type=str, default="/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints")
    ap.add_argument("--output_dir", type=str, default="/data/Deep_Angiography/AngioVision/fine-tuning/output")
    ap.add_argument(
        "--val_data_dir",
        type=str,
        default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed_Augmented",
        help="Validation data_dir passed to custom_framework_validate.py",
    )

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

    ap.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision.")
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    ap.add_argument("--vit_grad_ckpt", action="store_true", help="Enable ViT gradient checkpointing when vision trainable.")
    ap.add_argument("--logits_chunk", type=int, default=4, help="Chunk size for CLIP loss to avoid full BxB logits.")
    ap.add_argument("--vit_image_size", type=int, default=None, help="Force processor resize (e.g., 224) to control memory.")
    ap.add_argument("--empty_cache_each_step", action="store_true", help="Call torch.cuda.empty_cache() each step (helps fragmentation).")

    ap.add_argument("--validate_script", type=str, default="custom_framework_validate.py")
    ap.add_argument("--calculate_score_script", type=str, default="calculate_score.py")
    ap.add_argument("--plot_loss_script", type=str, default="plot_loss.py")

    ap.add_argument("--validate_device", type=str, default=None, help="Device string for validation (e.g., cuda or cpu). Default: auto.")

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)