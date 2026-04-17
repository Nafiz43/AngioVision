#!/usr/bin/env python3
"""
custom_framework_train.py

AngioVision fine-tuning: (Study) frames -> ViT -> temporal-aware pooling -> visual embedding.
Primary training path remains CLIP-style contrastive alignment between visual embeddings
and report text embeddings.

OPTIONAL GENERATION HEAD:
- Optional generative report head on top of the visual encoder.
- When --enable_generation is used, the model is trained on (visual data -> report text)
  using a decoder LM with cross-attention over visual sequence tokens.

PERFORMANCE OPTIMIZATIONS IN THIS VERSION:
1) Image decoding + ViT preprocessing (PIL.open, convert, resize, normalize) now happens
   inside the DataLoader workers via __getitem__. Previously num_workers were essentially
   idle - they only shuffled path strings - and all decode/normalize work serialized on
   the GPU thread.
2) The visual forward path now batches frames across the entire mini-batch into a single
   set of ViT calls (chunked by --frame_chunk_size for memory). Previously the ViT was
   called once per (study, sequence, frame_chunk), producing many tiny GPU launches.

Functionality is preserved end-to-end:
- Identical CLIP contrastive loss and identical generation loss.
- Identical temporal encoding semantics (per-frame sinusoidal positions inside each
  sequence; per-sequence sinusoidal positions across each study).
- Identical pooling semantics (frame-level then sequence-level).
- Identical cross-attention memory construction for the report decoder, including the
  optional pooled study token.
- Identical CLI surface (no flags removed; defaults unchanged).

Training modes:
1) CLIP only (default, backward compatible)
2) Joint CLIP + generation
3) Generation-dominant training by setting --clip_loss_weight 0
"""

from __future__ import annotations

import re
import ast
import math
import argparse
import csv
import os
import shlex
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

from transformers import (
    ViTModel,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

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
# Image processor utility (moved up so datasets can use it)
# ------------------------------------------------------------
def get_vit_processor(vit_name: str):
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(vit_name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(vit_name)
    raise ImportError("Neither ViTImageProcessor nor ViTFeatureExtractor is available in transformers.")


def _preprocess_frames_to_pixel_values(
    frame_files: List[Path],
    processor,
    vit_image_size: Optional[int],
) -> Tuple[Optional[torch.Tensor], List[int]]:
    """
    Open + convert + run image processor on a list of frame files.
    Silently skips frames that fail to open (mirrors original behavior).
    Returns (pixel_values_tensor_or_None, list_of_valid_local_positions).
    """
    imgs: List[Image.Image] = []
    valid_positions: List[int] = []

    for local_idx, p in enumerate(frame_files):
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
            valid_positions.append(local_idx)
        except Exception:
            continue

    if not imgs:
        return None, []

    if vit_image_size is not None:
        inputs = processor(
            images=imgs,
            return_tensors="pt",
            size={"height": vit_image_size, "width": vit_image_size},
        )
    else:
        inputs = processor(images=imgs, return_tensors="pt")

    pixel_values = inputs["pixel_values"]  # (T, 3, H, W) float32

    # Drop PIL handles eagerly to keep worker memory in check.
    for img in imgs:
        try:
            img.close()
        except Exception:
            pass
    del imgs, inputs

    return pixel_values, valid_positions


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class StudyDataset(Dataset):
    def __init__(
        self,
        meta_csv: Path,
        reports_csv: Path,
        base_frames_dir: Path,
        processor,
        vit_image_size: Optional[int] = None,
        report_text_col: str = "radrpt",
        anon_col: str = "Anon Acc #",
        sop_col: str = "SOPInstanceUIDs",
        report_type_col: str = "Type",
        min_frames_per_sequence: int = 1,
        max_sequences_per_study: Optional[int] = None,
        max_frames_per_sequence: Optional[int] = None,
        drop_missing_reports: bool = True,
        report_sampling: str = "uniform",
        report_sampling_seed: int = 42,
    ):
        self.meta = pd.read_csv(meta_csv)
        self.reports = pd.read_csv(reports_csv)

        self.base_frames_dir = base_frames_dir
        self.processor = processor
        self.vit_image_size = vit_image_size
        self.report_text_col = report_text_col
        self.anon_col = anon_col
        self.sop_col = sop_col
        self.report_type_col = report_type_col
        self.min_frames_per_sequence = min_frames_per_sequence
        self.max_sequences_per_study = max_sequences_per_study
        self.max_frames_per_sequence = max_frames_per_sequence
        self.drop_missing_reports = drop_missing_reports
        self.report_sampling = report_sampling
        self.report_sampling_seed = int(report_sampling_seed)

        if self.report_sampling != "uniform":
            raise ValueError(f"Unsupported report_sampling={self.report_sampling}. Only 'uniform' is supported.")

        self.report_map: Dict[str, List[Dict[str, str]]] = {}

        for _, r in self.reports.iterrows():
            acc = str(r.get(self.anon_col, "")).strip()
            if not acc:
                continue

            txt = r.get(self.report_text_col, "")
            if isinstance(txt, float) and pd.isna(txt):
                txt = ""
            txt = str(txt).strip()

            rep_type = r.get(self.report_type_col, "Unknown")
            if isinstance(rep_type, float) and pd.isna(rep_type):
                rep_type = "Unknown"
            rep_type = str(rep_type).strip()

            if not txt:
                continue

            if acc not in self.report_map:
                self.report_map[acc] = []

            self.report_map[acc].append({"text": txt, "type": rep_type})

        if self.drop_missing_reports:
            keep = []
            for i, row in self.meta.iterrows():
                acc = str(row.get(self.anon_col, "")).strip()
                if acc in self.report_map and len(self.report_map[acc]) > 0:
                    keep.append(i)
            self.meta = self.meta.loc[keep].reset_index(drop=True)

        self.report_count_map: Dict[str, int] = {
            acc: len(reports) for acc, reports in self.report_map.items()
        }

        n_accessions_with_reports = len(self.report_count_map)
        if n_accessions_with_reports > 0:
            counts = list(self.report_count_map.values())
            min_count = min(counts)
            max_count = max(counts)
            mean_count = sum(counts) / len(counts)
            print("[INFO] Report variant summary by accession:")
            print(f"       Accessions with >=1 report : {n_accessions_with_reports}")
            print(f"       Min reports per accession  : {min_count}")
            print(f"       Max reports per accession  : {max_count}")
            print(f"       Mean reports per accession : {mean_count:.2f}")
        else:
            print("[WARN] No usable reports found in reports_csv.")

    def __len__(self):
        return len(self.meta)

    def _sample_report_uniformly(self, reports_for_acc: List[Dict[str, str]], idx: int) -> Dict[str, str]:
        n = len(reports_for_acc)
        if n == 0:
            return {"text": "", "type": "Missing"}
        if n == 1:
            return reports_for_acc[0]
        sampled_idx = int(torch.randint(low=0, high=n, size=(1,)).item())
        return reports_for_acc[sampled_idx]

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        acc = str(row.get(self.anon_col, "")).strip()

        sop_uids = parse_sop_instance_uids(row.get(self.sop_col, ""))
        if self.max_sequences_per_study is not None:
            sop_uids = sop_uids[: self.max_sequences_per_study]

        # Worker-side preprocessing: load + resize + normalize per sequence.
        sequences_pv: List[torch.Tensor] = []
        sequences_positions: List[torch.Tensor] = []

        for sop in sop_uids:
            frame_files = find_frame_files_for_sop(self.base_frames_dir, acc, sop)

            if self.max_frames_per_sequence is not None and len(frame_files) > self.max_frames_per_sequence:
                idxs = torch.linspace(0, len(frame_files) - 1, steps=self.max_frames_per_sequence).long().tolist()
                frame_files = [frame_files[i] for i in idxs]

            if len(frame_files) < self.min_frames_per_sequence:
                continue

            pixel_values, valid_positions = _preprocess_frames_to_pixel_values(
                frame_files=frame_files,
                processor=self.processor,
                vit_image_size=self.vit_image_size,
            )
            if pixel_values is None or pixel_values.size(0) == 0:
                # All frames in this sequence failed to open; skip the sequence rather
                # than carry a degenerate "all -1e9" embedding into the pool.
                continue

            sequences_pv.append(pixel_values)
            sequences_positions.append(torch.tensor(valid_positions, dtype=torch.long))

        reports_for_acc = self.report_map.get(acc, [])
        chosen_report = self._sample_report_uniformly(reports_for_acc, idx)

        text = chosen_report.get("text", "")
        report_type = chosen_report.get("type", "Unknown")
        num_reports_for_acc = self.report_count_map.get(acc, 0)

        return {
            "acc": acc,
            "pixel_values": sequences_pv,         # List[Tensor(T_i, 3, H, W)]
            "positions": sequences_positions,     # List[Tensor(T_i,)]
            "text": text,
            "report_type": report_type,
            "num_reports_for_acc": num_reports_for_acc,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [
        b for b in batch
        if b["text"]
        and isinstance(b["pixel_values"], list)
        and len(b["pixel_values"]) > 0
    ]
    return {
        "acc": [b["acc"] for b in kept],
        "pixel_values": [b["pixel_values"] for b in kept],
        "positions": [b["positions"] for b in kept],
        "text": [b["text"] for b in kept],
        "report_type": [b["report_type"] for b in kept],
        "num_reports_for_acc": [b["num_reports_for_acc"] for b in kept],
    }


class HoldoutStudyDataset(Dataset):
    def __init__(
        self,
        meta_csv: Path,
        base_frames_dir: Path,
        processor,
        vit_image_size: Optional[int] = None,
        anon_col: str = "Anon Acc #",
        sop_col: str = "SOPInstanceUIDs",
        min_frames_per_sequence: int = 1,
        max_sequences_per_study: Optional[int] = None,
        max_frames_per_sequence: Optional[int] = None,
    ):
        self.meta = pd.read_csv(meta_csv)
        self.base_frames_dir = base_frames_dir
        self.processor = processor
        self.vit_image_size = vit_image_size
        self.anon_col = anon_col
        self.sop_col = sop_col
        self.min_frames_per_sequence = min_frames_per_sequence
        self.max_sequences_per_study = max_sequences_per_study
        self.max_frames_per_sequence = max_frames_per_sequence

        keep = []
        for i, row in self.meta.iterrows():
            acc = str(row.get(self.anon_col, "")).strip()
            if acc:
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

        sequences_pv: List[torch.Tensor] = []
        sequences_positions: List[torch.Tensor] = []
        kept_sops: List[str] = []
        total_frames = 0

        for sop in sop_uids:
            frame_files = find_frame_files_for_sop(self.base_frames_dir, acc, sop)

            if self.max_frames_per_sequence is not None and len(frame_files) > self.max_frames_per_sequence:
                idxs = torch.linspace(0, len(frame_files) - 1, steps=self.max_frames_per_sequence).long().tolist()
                frame_files = [frame_files[i] for i in idxs]

            if len(frame_files) < self.min_frames_per_sequence:
                continue

            pixel_values, valid_positions = _preprocess_frames_to_pixel_values(
                frame_files=frame_files,
                processor=self.processor,
                vit_image_size=self.vit_image_size,
            )
            if pixel_values is None or pixel_values.size(0) == 0:
                continue

            sequences_pv.append(pixel_values)
            sequences_positions.append(torch.tensor(valid_positions, dtype=torch.long))
            kept_sops.append(str(sop).strip())
            total_frames += int(pixel_values.size(0))

        return {
            "acc": acc,
            "pixel_values": sequences_pv,
            "positions": sequences_positions,
            "sop_uids": kept_sops,
            "num_sequences": len(sequences_pv),
            "num_frames": total_frames,
        }


def holdout_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [
        b for b in batch
        if isinstance(b["pixel_values"], list) and len(b["pixel_values"]) > 0
    ]
    return {
        "acc": [b["acc"] for b in kept],
        "pixel_values": [b["pixel_values"] for b in kept],
        "positions": [b["positions"] for b in kept],
        "sop_uids": [b["sop_uids"] for b in kept],
        "num_sequences": [b["num_sequences"] for b in kept],
        "num_frames": [b["num_frames"] for b in kept],
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


# ------------------------------------------------------------
# Temporal encoding helpers
# ------------------------------------------------------------
TEMPORAL_MODE_CHOICES = ("none", "sinusoidal")


def build_sinusoidal_position_encoding(
    positions: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape={tuple(positions.shape)}")

    positions = positions.to(device=device, dtype=torch.float32)
    pe = torch.zeros((positions.size(0), dim), device=device, dtype=torch.float32)

    if dim == 1:
        pe[:, 0] = positions
        return pe.to(dtype=dtype)

    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )

    pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    if dim > 1:
        cos_width = pe[:, 1::2].shape[1]
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term[:cos_width])

    return pe.to(dtype=dtype)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class PooledCLIP(nn.Module):
    def __init__(
        self,
        vit_name: str,
        text_model_name: str,
        embed_dim: int = 256,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        frame_pooling: str = "max",
        sequence_pooling: str = "max",
        vit_trainable_blocks: int = 3,
        vit_unfreeze_patch_embed: bool = False,
        temporal_mode: str = "sinusoidal",
        temporal_on_frames: bool = True,
        temporal_on_sequences: bool = False,
        frame_temporal_scale: float = 1.0,
        sequence_temporal_scale: float = 1.0,
        enable_generation: bool = False,
        decoder_model_name: str = "gpt2",
        freeze_decoder: bool = False,
        generation_use_study_token: bool = True,
    ):
        super().__init__()
        if frame_pooling not in POOL_CHOICES:
            raise ValueError(f"frame_pooling must be one of {POOL_CHOICES}, got {frame_pooling}")
        if sequence_pooling not in POOL_CHOICES:
            raise ValueError(f"sequence_pooling must be one of {POOL_CHOICES}, got {sequence_pooling}")
        if temporal_mode not in TEMPORAL_MODE_CHOICES:
            raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODE_CHOICES}, got {temporal_mode}")

        self.vit = ViTModel.from_pretrained(vit_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.text_hidden = self.text_model.config.hidden_size

        self.frame_pooling = frame_pooling
        self.sequence_pooling = sequence_pooling

        self.temporal_mode = temporal_mode
        self.temporal_on_frames = bool(temporal_on_frames)
        self.temporal_on_sequences = bool(temporal_on_sequences)
        self.frame_temporal_scale = float(frame_temporal_scale)
        self.sequence_temporal_scale = float(sequence_temporal_scale)

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.vit_hidden),
            nn.GELU(),
            nn.Linear(self.vit_hidden, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden, self.text_hidden),
            nn.GELU(),
            nn.Linear(self.text_hidden, embed_dim),
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self.enable_generation = bool(enable_generation)
        self.generation_use_study_token = bool(generation_use_study_token)
        self.decoder_model_name = decoder_model_name
        self.report_decoder = None
        self.decoder_hidden = None
        self.generation_visual_proj = None

        if self.enable_generation:
            decoder_config = AutoConfig.from_pretrained(decoder_model_name)
            setattr(decoder_config, "add_cross_attention", True)
            setattr(decoder_config, "is_decoder", True)
            self.report_decoder = AutoModelForCausalLM.from_pretrained(
                decoder_model_name,
                config=decoder_config,
                ignore_mismatched_sizes=True,
            )
            if hasattr(self.report_decoder.config, "n_embd"):
                self.decoder_hidden = int(self.report_decoder.config.n_embd)
            elif hasattr(self.report_decoder.config, "hidden_size"):
                self.decoder_hidden = int(self.report_decoder.config.hidden_size)
            elif hasattr(self.report_decoder.config, "d_model"):
                self.decoder_hidden = int(self.report_decoder.config.d_model)
            else:
                raise RuntimeError("Could not infer decoder hidden size from decoder config.")

            self.generation_visual_proj = nn.Sequential(
                nn.Linear(self.vit_hidden, self.vit_hidden),
                nn.GELU(),
                nn.Linear(self.vit_hidden, self.decoder_hidden),
            )

            if freeze_decoder:
                for p in self.report_decoder.parameters():
                    p.requires_grad = False

        if freeze_vision:
            for p in self.vit.parameters():
                p.requires_grad = False
        else:
            self._configure_partial_vit_finetuning(
                vit_trainable_blocks=vit_trainable_blocks,
                vit_unfreeze_patch_embed=vit_unfreeze_patch_embed,
            )

        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

    def _configure_partial_vit_finetuning(
        self,
        vit_trainable_blocks: int,
        vit_unfreeze_patch_embed: bool,
    ) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False

        layers = None
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            layers = self.vit.encoder.layer

        if layers is None:
            raise RuntimeError("Could not locate ViT encoder layers at self.vit.encoder.layer")

        num_layers = len(layers)
        n = max(0, min(int(vit_trainable_blocks), num_layers))

        if n > 0:
            for block in layers[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

        if hasattr(self.vit, "layernorm") and self.vit.layernorm is not None:
            for p in self.vit.layernorm.parameters():
                p.requires_grad = True

        if hasattr(self.vit, "pooler") and self.vit.pooler is not None:
            for p in self.vit.pooler.parameters():
                p.requires_grad = True

        if vit_unfreeze_patch_embed:
            if hasattr(self.vit, "embeddings") and self.vit.embeddings is not None:
                for p in self.vit.embeddings.parameters():
                    p.requires_grad = True

    def _vit_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.vit.parameters())

    def _text_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.text_model.parameters())

    def _decoder_is_frozen(self) -> bool:
        if self.report_decoder is None:
            return True
        return all(not p.requires_grad for p in self.report_decoder.parameters())

    def _add_temporal_encoding(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if self.temporal_mode == "none" or scale == 0.0:
            return x
        pe = build_sinusoidal_position_encoding(
            positions=positions,
            dim=x.size(-1),
            device=x.device,
            dtype=x.dtype,
        )
        return x + (scale * pe)

    def _vit_forward_chunked(
        self,
        big_pixel_values: torch.Tensor,
        frame_chunk_size: int,
    ) -> torch.Tensor:
        """
        Run the ViT over a stack of (N_total, 3, H, W) frames in chunks of
        frame_chunk_size. Returns CLS features as (N_total, vit_hidden).

        This is the core of optimization (2): instead of one ViT call per
        (study, sequence, chunk), we make one ViT call per chunk across the
        ENTIRE mini-batch of frames.
        """
        if big_pixel_values.size(0) == 0:
            return torch.zeros(
                (0, self.vit_hidden),
                device=big_pixel_values.device,
                dtype=big_pixel_values.dtype,
            )

        vit_ctx = torch.no_grad() if self._vit_is_frozen() else torch.enable_grad()
        chunk = max(1, int(frame_chunk_size))
        out_chunks: List[torch.Tensor] = []

        with vit_ctx:
            for i in range(0, big_pixel_values.size(0), chunk):
                pv = big_pixel_values[i: i + chunk]
                out = self.vit(pixel_values=pv)
                cls = out.last_hidden_state[:, 0, :]
                out_chunks.append(cls)
                del pv, out, cls

        return torch.cat(out_chunks, dim=0)

    def encode_visual_batch(
        self,
        batch_pixel_values: List[List[torch.Tensor]],
        batch_positions: List[List[torch.Tensor]],
        device: torch.device,
        frame_chunk_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        New batched visual forward.

        Inputs:
          batch_pixel_values[study_idx][seq_idx] = Tensor(T_i, 3, H, W) on CPU
          batch_positions[study_idx][seq_idx]    = Tensor(T_i,) long, original frame positions

        Returns:
          study_visuals_tensor: (B, vit_hidden) pooled study embeddings
          seq_tokens          : (B, S_max[+1], vit_hidden) cross-attention memory
          seq_mask            : (B, S_max[+1]) attention mask for cross-attn memory

        Functionally identical to the previous per-sequence loop, but performs all
        ViT work in a single set of chunked calls across the whole mini-batch.
        """
        B = len(batch_pixel_values)

        # Defensive empty-batch handling. Collate already filters empty studies, so
        # this branch should not normally trigger.
        if B == 0:
            empty_study = torch.zeros((0, self.vit_hidden), device=device)
            empty_tokens = torch.zeros((0, 1, self.vit_hidden), device=device)
            empty_mask = torch.zeros((0, 1), device=device, dtype=torch.long)
            return empty_study, empty_tokens, empty_mask

        # ---- Step 1: Flatten frames across the entire mini-batch ----
        per_seq_lengths: List[List[int]] = [[] for _ in range(B)]
        flat_pv_chunks: List[torch.Tensor] = []
        flat_positions: List[torch.Tensor] = []

        for s_idx, study_seqs in enumerate(batch_pixel_values):
            for q_idx, seq_pv in enumerate(study_seqs):
                T = int(seq_pv.size(0))
                per_seq_lengths[s_idx].append(T)
                if T == 0:
                    continue
                flat_pv_chunks.append(seq_pv)
                flat_positions.append(batch_positions[s_idx][q_idx])

        if flat_pv_chunks:
            big_pixel_values = torch.cat(flat_pv_chunks, dim=0).to(device, non_blocking=True)
        else:
            big_pixel_values = torch.zeros((0, 3, 1, 1), device=device)

        # ---- Step 2: One chunked ViT pass over all frames ----
        all_frame_embeds = self._vit_forward_chunked(
            big_pixel_values=big_pixel_values,
            frame_chunk_size=frame_chunk_size,
        )  # (N_total, vit_hidden)

        # ---- Step 3: Scatter back per (study, seq), apply per-frame temporal
        #              encoding, and pool frames -> per-sequence embedding. ----
        per_study_seq_feats: List[torch.Tensor] = []
        cursor = 0
        flat_seq_idx = 0

        for s_idx in range(B):
            seq_feats: List[torch.Tensor] = []

            for q_idx, T in enumerate(per_seq_lengths[s_idx]):
                if T == 0:
                    continue

                seq_embeds = all_frame_embeds[cursor: cursor + T]
                cursor += T

                if self.temporal_on_frames and self.temporal_mode != "none":
                    positions = flat_positions[flat_seq_idx].to(device=device, dtype=torch.long)
                    seq_embeds = self._add_temporal_encoding(
                        seq_embeds,
                        positions=positions,
                        scale=self.frame_temporal_scale,
                    )

                seq_feat = pool_stack(seq_embeds, self.frame_pooling)
                seq_feats.append(seq_feat)
                flat_seq_idx += 1

            if not seq_feats:
                # No sequences for this study survived. Use a zero placeholder so the
                # rest of the model stays well-defined.
                zero_feat = torch.zeros(self.vit_hidden, device=device)
                seq_stack = zero_feat.unsqueeze(0)
            else:
                seq_stack = torch.stack(seq_feats, dim=0)

            per_study_seq_feats.append(seq_stack)

        # ---- Step 4: Per-study sequence-level temporal encoding + pooling ----
        study_visuals: List[torch.Tensor] = []
        for s_idx in range(B):
            seq_stack = per_study_seq_feats[s_idx]
            if self.temporal_on_sequences and self.temporal_mode != "none":
                seq_positions = torch.arange(seq_stack.size(0), device=device, dtype=torch.long)
                seq_stack = self._add_temporal_encoding(
                    seq_stack,
                    positions=seq_positions,
                    scale=self.sequence_temporal_scale,
                )
                per_study_seq_feats[s_idx] = seq_stack

            study_feat = pool_stack(seq_stack, self.sequence_pooling)
            study_visuals.append(study_feat)

        study_visuals_tensor = torch.stack(study_visuals, dim=0)

        # ---- Step 5: Build cross-attention memory (with optional study token) ----
        max_seq = max(x.size(0) for x in per_study_seq_feats)
        token_dim = self.vit_hidden
        prepend = 1 if self.generation_use_study_token else 0

        seq_tokens = torch.zeros(
            (B, max_seq + prepend, token_dim),
            device=device,
            dtype=study_visuals_tensor.dtype,
        )
        seq_mask = torch.zeros(
            (B, max_seq + prepend),
            device=device,
            dtype=torch.long,
        )

        for i in range(B):
            offset = 0
            if self.generation_use_study_token:
                seq_tokens[i, 0] = study_visuals_tensor[i]
                seq_mask[i, 0] = 1
                offset = 1
            seq_stack = per_study_seq_feats[i]
            seq_tokens[i, offset: offset + seq_stack.size(0)] = seq_stack
            seq_mask[i, offset: offset + seq_stack.size(0)] = 1

        return study_visuals_tensor, seq_tokens, seq_mask

    def forward(
        self,
        batch_pixel_values: List[List[torch.Tensor]],
        batch_positions: List[List[torch.Tensor]],
        texts: Optional[List[str]],
        tokenizer,
        device: torch.device,
        frame_chunk_size: int = 16,
        generation_tokenizer=None,
        generation_texts: Optional[List[str]] = None,
        generation_max_length: int = 256,
    ) -> Dict[str, Optional[torch.Tensor]]:
        study_visuals, visual_tokens, visual_attention_mask = self.encode_visual_batch(
            batch_pixel_values=batch_pixel_values,
            batch_positions=batch_positions,
            device=device,
            frame_chunk_size=frame_chunk_size,
        )

        image_embeds = F.normalize(self.vision_proj(study_visuals), dim=-1)

        text_embeds = None
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        if texts is not None:
            tok = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            text_ctx = torch.no_grad() if self._text_is_frozen() else torch.enable_grad()
            with text_ctx:
                tout = self.text_model(**tok)
                tcls = tout.last_hidden_state[:, 0, :]
            text_embeds = F.normalize(self.text_proj(tcls), dim=-1)

        gen_loss = None
        gen_logits = None
        if self.enable_generation and generation_texts is not None:
            if generation_tokenizer is None:
                raise ValueError("generation_tokenizer must be provided when generation_texts is used.")

            gen_enc = generation_tokenizer(
                generation_texts,
                padding=True,
                truncation=True,
                max_length=generation_max_length,
                return_tensors="pt",
            ).to(device)

            labels = gen_enc["input_ids"].clone()
            labels[gen_enc["attention_mask"] == 0] = -100

            decoder_ctx = torch.no_grad() if self._decoder_is_frozen() else torch.enable_grad()
            projected_visual_tokens = self.generation_visual_proj(visual_tokens)

            with decoder_ctx:
                gen_out = self.report_decoder(
                    input_ids=gen_enc["input_ids"],
                    attention_mask=gen_enc["attention_mask"],
                    labels=labels,
                    encoder_hidden_states=projected_visual_tokens,
                    encoder_attention_mask=visual_attention_mask,
                    return_dict=True,
                )
            gen_loss = gen_out.loss
            gen_logits = gen_out.logits

        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logit_scale": logit_scale,
            "gen_loss": gen_loss,
            "gen_logits": gen_logits,
            "visual_tokens": visual_tokens,
            "visual_attention_mask": visual_attention_mask,
        }

    @torch.no_grad()
    def generate_reports(
        self,
        batch_pixel_values: List[List[torch.Tensor]],
        batch_positions: List[List[torch.Tensor]],
        generation_tokenizer,
        device: torch.device,
        frame_chunk_size: int = 16,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        prompt_texts: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.enable_generation or self.report_decoder is None:
            raise RuntimeError("Generation head is not enabled in this checkpoint/model.")

        if num_beams != 1:
            print("[WARN] Manual decoder path currently supports greedy/sampling decode only. Overriding num_beams to 1.")

        study_visuals, visual_tokens, visual_attention_mask = self.encode_visual_batch(
            batch_pixel_values=batch_pixel_values,
            batch_positions=batch_positions,
            device=device,
            frame_chunk_size=frame_chunk_size,
        )
        _ = study_visuals  # kept for symmetry/debugging
        projected_visual_tokens = self.generation_visual_proj(visual_tokens)

        batch_size = projected_visual_tokens.size(0)
        if prompt_texts is None:
            if generation_tokenizer.bos_token is not None:
                prompt_texts = [generation_tokenizer.bos_token] * batch_size
            elif generation_tokenizer.eos_token is not None:
                prompt_texts = [generation_tokenizer.eos_token] * batch_size
            else:
                prompt_texts = [""] * batch_size

        prompt_enc = generation_tokenizer(prompt_texts, padding=True, return_tensors="pt").to(device)

        input_ids = prompt_enc["input_ids"]
        attention_mask = prompt_enc["attention_mask"]

        eos_token_id = generation_tokenizer.eos_token_id
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=device)

        for _step in range(max_new_tokens):
            out = self.report_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=projected_visual_tokens,
                encoder_attention_mask=visual_attention_mask,
                return_dict=True,
            )

            next_token_logits = out.logits[:, -1, :]

            if temperature is not None and temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)

                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False

                    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
                    denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    sorted_probs = sorted_probs / denom

                    sampled = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, sampled)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype),
                ],
                dim=-1,
            )

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if torch.all(finished):
                    break

        return generation_tokenizer.batch_decode(input_ids, skip_special_tokens=True)


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
            w.writerow(["epoch", "step", "batch_size", "loss", "loss_ema", "clip_loss", "gen_loss"])

    return log_path


def _append_loss_row(
    log_path: Path,
    epoch: int,
    step: int,
    batch_size: int,
    loss_val: float,
    loss_ema: float,
    clip_loss_val: Optional[float],
    gen_loss_val: Optional[float],
):
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            epoch,
            step,
            batch_size,
            f"{loss_val:.8f}",
            f"{loss_ema:.8f}",
            "" if clip_loss_val is None else f"{clip_loss_val:.8f}",
            "" if gen_loss_val is None else f"{gen_loss_val:.8f}",
        ])


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




def _shell_quote(arg: str) -> str:
    return shlex.quote(str(arg))


def _build_command_string() -> str:
    return " ".join(_shell_quote(x) for x in [sys.executable] + sys.argv)


def save_training_command(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd_path = run_dir / "train_command.sh"
    cmd_text = _build_command_string() + "\n"
    cmd_path.write_text(cmd_text)
    try:
        current_mode = cmd_path.stat().st_mode
        cmd_path.chmod(current_mode | 0o111)
    except Exception:
        pass
    print(f"[INFO] Saved training command: {cmd_path}")
    return cmd_path

def run_post_training_pipeline(args, run_name: str, run_dir: Path, loss_csv: Path) -> None:
    val_data_dir = Path(args.val_data_dir)
    if not val_data_dir.exists():
        raise SystemExit(f"[ERROR] Validation data dir does not exist: {val_data_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"[ERROR] Run directory does not exist: {run_dir}")

    epoch_ckpts = sorted(run_dir.glob("epoch_*.pt"))
    last_ckpt = run_dir / "last.pt"
    if not epoch_ckpts and not last_ckpt.exists():
        raise SystemExit(f"[ERROR] No checkpoints found in run directory: {run_dir}")

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
        str(run_dir),
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




def _sanitize_text_for_csv(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ").strip()


def run_holdout_report_generation(
    model: "PooledCLIP",
    args,
    processor,
    generation_tokenizer,
    device: torch.device,
    epoch: int,
    run_name: str,
) -> Optional[Path]:
    print(f"\n[INFO] Starting holdout report generation for epoch {epoch}...\n")

    if not args.enable_generation:
        return None

    holdout_meta_csv = Path(args.holdout_meta_csv)
    holdout_base_frames_dir = Path(args.holdout_base_frames_dir)
    if not holdout_meta_csv.exists():
        print(f"[WARN] Holdout meta CSV not found; skipping epoch holdout generation: {holdout_meta_csv}")
        return None
    if not holdout_base_frames_dir.exists():
        print(f"[WARN] Holdout base frames dir not found; skipping epoch holdout generation: {holdout_base_frames_dir}")
        return None

    holdout_dataset = HoldoutStudyDataset(
        meta_csv=holdout_meta_csv,
        base_frames_dir=holdout_base_frames_dir,
        processor=processor,
        vit_image_size=args.vit_image_size,
        anon_col=args.holdout_anon_col,
        sop_col=args.holdout_sop_col,
        min_frames_per_sequence=args.min_frames_per_sequence,
        max_sequences_per_study=args.max_sequences_per_study,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=max(1, int(args.holdout_generation_batch_size)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=holdout_collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    out_dir = Path(args.report_generation_output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"epoch_{epoch}_generated_reports.csv"

    was_training = model.training
    model.eval()

    rows: List[Dict[str, Any]] = []
    pbar = tqdm(holdout_loader, desc=f"Holdout generation epoch {epoch}", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            if len(batch["acc"]) == 0:
                continue

            generated_reports = model.generate_reports(
                batch_pixel_values=batch["pixel_values"],
                batch_positions=batch["positions"],
                generation_tokenizer=generation_tokenizer,
                device=device,
                frame_chunk_size=args.frame_chunk_size,
                max_new_tokens=args.holdout_generation_max_new_tokens,
                num_beams=args.holdout_generation_num_beams,
                do_sample=args.holdout_generation_do_sample,
                top_p=args.holdout_generation_top_p,
                temperature=args.holdout_generation_temperature,
                prompt_texts=None,
            )

            for i, acc in enumerate(batch["acc"]):
                print(f"[GEN] Processing accession {acc} | sequences={batch['num_sequences'][i]}")
                rows.append({
                    "epoch": epoch,
                    "accession": acc,
                    "num_sequences": batch["num_sequences"][i],
                    "num_frames": batch["num_frames"][i],
                    "sop_uids": " | ".join(batch["sop_uids"][i]),
                    "generated_report": _sanitize_text_for_csv(generated_reports[i]),
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[INFO] Finished holdout generation for epoch {epoch}")
    print(f"[INFO] Saved to: {out_csv}\n")

    if was_training:
        model.train()
    return out_csv


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
            if any(p.requires_grad for p in block.parameters()):
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
    print(f"[INFO] Text encoder trainable: {any(p.requires_grad for p in model.text_model.parameters())}")
    print(f"[INFO] vision_proj trainable: {any(p.requires_grad for p in model.vision_proj.parameters())}")
    print(f"[INFO] text_proj trainable: {any(p.requires_grad for p in model.text_proj.parameters())}")
    print(f"[INFO] temporal_mode: {model.temporal_mode}")
    print(f"[INFO] temporal_on_frames: {model.temporal_on_frames} (scale={model.frame_temporal_scale})")
    print(f"[INFO] temporal_on_sequences: {model.temporal_on_sequences} (scale={model.sequence_temporal_scale})")
    print(f"[INFO] enable_generation: {model.enable_generation}")
    if model.enable_generation:
        print(f"[INFO] decoder model: {model.decoder_model_name}")
        print(f"[INFO] decoder trainable: {any(p.requires_grad for p in model.report_decoder.parameters())}")
        print(f"[INFO] generation_visual_proj trainable: {any(p.requires_grad for p in model.generation_visual_proj.parameters())}")


def build_optimizer(model: "PooledCLIP", args) -> torch.optim.Optimizer:
    vision_backbone_params = [p for p in model.vit.parameters() if p.requires_grad]
    text_backbone_params = [p for p in model.text_model.parameters() if p.requires_grad]

    head_params = []
    head_params.extend([p for p in model.vision_proj.parameters() if p.requires_grad])
    head_params.extend([p for p in model.text_proj.parameters() if p.requires_grad])
    if model.logit_scale.requires_grad:
        head_params.append(model.logit_scale)

    generation_head_params = []
    decoder_params = []
    if model.enable_generation:
        generation_head_params.extend([p for p in model.generation_visual_proj.parameters() if p.requires_grad])
        decoder_params.extend([p for p in model.report_decoder.parameters() if p.requires_grad])

    param_groups = []

    if vision_backbone_params:
        param_groups.append({
            "params": vision_backbone_params,
            "lr": args.vision_backbone_lr,
            "weight_decay": args.weight_decay,
        })

    if text_backbone_params:
        param_groups.append({
            "params": text_backbone_params,
            "lr": args.text_lr,
            "weight_decay": args.weight_decay,
        })

    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        })

    if generation_head_params:
        param_groups.append({
            "params": generation_head_params,
            "lr": args.gen_head_lr,
            "weight_decay": args.weight_decay,
        })

    if decoder_params:
        param_groups.append({
            "params": decoder_params,
            "lr": args.decoder_lr,
            "weight_decay": args.weight_decay,
        })

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    print("[INFO] Optimizer parameter groups:")
    if vision_backbone_params:
        print(f"       vision backbone: {sum(p.numel() for p in vision_backbone_params):,} params, lr={args.vision_backbone_lr}")
    if text_backbone_params:
        print(f"       text backbone:   {sum(p.numel() for p in text_backbone_params):,} params, lr={args.text_lr}")
    if head_params:
        print(f"       projection/head: {sum(p.numel() for p in head_params):,} params, lr={args.head_lr}")
    if generation_head_params:
        print(f"       gen vis head:    {sum(p.numel() for p in generation_head_params):,} params, lr={args.gen_head_lr}")
    if decoder_params:
        print(f"       decoder:         {sum(p.numel() for p in decoder_params):,} params, lr={args.decoder_lr}")

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

    save_training_command(run_dir)

    if ckpt_path.exists():
        print(f"[INFO] Existing checkpoint found at: {ckpt_path}")
        print("[INFO] Skipping training and using existing checkpoint.")
        run_post_training_pipeline(args=args, run_name=run_name, run_dir=run_dir, loss_csv=loss_log_path)
        return

    # Build the image processor up front so it can be passed into the dataset for
    # worker-side preprocessing (optimization 1).
    processor = get_vit_processor(args.vit_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    generation_tokenizer = None
    if args.enable_generation:
        generation_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_name)
        if generation_tokenizer.pad_token is None:
            if generation_tokenizer.eos_token is not None:
                generation_tokenizer.pad_token = generation_tokenizer.eos_token
            else:
                generation_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dataset = StudyDataset(
        meta_csv=Path(args.meta_csv),
        reports_csv=Path(args.reports_csv),
        base_frames_dir=Path(args.base_frames_dir),
        processor=processor,
        vit_image_size=args.vit_image_size,
        report_text_col=args.report_text_col,
        anon_col=args.holdout_anon_col,
        sop_col=args.holdout_sop_col,
        report_type_col=args.report_type_col,
        min_frames_per_sequence=args.min_frames_per_sequence,
        max_sequences_per_study=args.max_sequences_per_study,
        max_frames_per_sequence=args.max_frames_per_sequence,
        drop_missing_reports=not args.keep_missing_reports,
        report_sampling=args.report_sampling,
        report_sampling_seed=args.report_sampling_seed,
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

    model = PooledCLIP(
        vit_name=args.vit_name,
        text_model_name=args.bert_name,
        embed_dim=args.embed_dim,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
        frame_pooling=frame_pooling,
        sequence_pooling=sequence_pooling,
        vit_trainable_blocks=args.vit_trainable_blocks,
        vit_unfreeze_patch_embed=args.vit_unfreeze_patch_embed,
        temporal_mode=args.temporal_mode,
        temporal_on_frames=(not args.disable_frame_temporal),
        temporal_on_sequences=args.enable_sequence_temporal,
        frame_temporal_scale=args.frame_temporal_scale,
        sequence_temporal_scale=args.sequence_temporal_scale,
        enable_generation=args.enable_generation,
        decoder_model_name=args.decoder_model_name,
        freeze_decoder=args.freeze_decoder,
        generation_use_study_token=(not args.disable_generation_study_token),
    ).to(device)

    if args.enable_generation and len(generation_tokenizer) > model.report_decoder.get_input_embeddings().num_embeddings:
        model.report_decoder.resize_token_embeddings(len(generation_tokenizer))
    if args.enable_generation and model.report_decoder.config.pad_token_id is None:
        model.report_decoder.config.pad_token_id = generation_tokenizer.pad_token_id

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
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)

        for batch in pbar:
            if len(batch["text"]) == 0:
                continue

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    batch_pixel_values=batch["pixel_values"],
                    batch_positions=batch["positions"],
                    texts=batch["text"] if args.clip_loss_weight > 0 else None,
                    tokenizer=tokenizer,
                    device=device,
                    frame_chunk_size=args.frame_chunk_size,
                    generation_tokenizer=generation_tokenizer,
                    generation_texts=batch["text"] if args.enable_generation and args.gen_loss_weight > 0 else None,
                    generation_max_length=args.generation_max_length,
                )

                clip_loss_val = None
                if args.clip_loss_weight > 0:
                    if outputs["text_embeds"] is None:
                        raise RuntimeError("CLIP loss weight > 0 but text embeddings were not computed.")
                    clip_loss_val = clip_loss_chunked(
                        image_embeds=outputs["image_embeds"],
                        text_embeds=outputs["text_embeds"],
                        logit_scale=outputs["logit_scale"],
                        chunk=args.logits_chunk,
                    )

                gen_loss_val = None
                if args.enable_generation and args.gen_loss_weight > 0:
                    if outputs["gen_loss"] is None:
                        raise RuntimeError("Generation is enabled but gen_loss was not produced.")
                    gen_loss_val = outputs["gen_loss"]

                if clip_loss_val is None and gen_loss_val is None:
                    raise RuntimeError("Both clip_loss and gen_loss are disabled. Nothing to optimize.")

                total_loss = 0.0
                if clip_loss_val is not None:
                    total_loss = total_loss + (args.clip_loss_weight * clip_loss_val)
                if gen_loss_val is not None:
                    total_loss = total_loss + (args.gen_loss_weight * gen_loss_val)

                loss_to_backprop = total_loss / float(grad_accum)

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

            total_loss_float = float(total_loss.detach().cpu().item())
            clip_loss_float = None if clip_loss_val is None else float(clip_loss_val.detach().cpu().item())
            gen_loss_float = None if gen_loss_val is None else float(gen_loss_val.detach().cpu().item())

            if ema is None:
                ema = total_loss_float
            else:
                ema = ema_beta * ema + (1 - ema_beta) * total_loss_float

            _append_loss_row(
                log_path=loss_log_path,
                epoch=epoch + 1,
                step=step,
                batch_size=len(batch["text"]),
                loss_val=total_loss_float,
                loss_ema=float(ema),
                clip_loss_val=clip_loss_float,
                gen_loss_val=gen_loss_float,
            )

            postfix = {"loss": total_loss_float, "loss_ema": float(ema), "bs": len(batch["text"])}
            if clip_loss_float is not None:
                postfix["clip"] = clip_loss_float
            if gen_loss_float is not None:
                postfix["gen"] = gen_loss_float
            pbar.set_postfix(**postfix)

            del outputs, total_loss, loss_to_backprop
            if args.empty_cache_each_step and device.type == "cuda":
                torch.cuda.empty_cache()

        last_ckpt = _save_last_checkpoint(run_dir=run_dir, model=model, opt=opt, step=step, epoch=epoch + 1)
        epoch_ckpt = _save_epoch_checkpoint(run_dir=run_dir, model=model, opt=opt, step=step, epoch=epoch + 1)
        print(f"[INFO] Saved rolling checkpoint: {last_ckpt}")
        print(f"[INFO] Saved epoch checkpoint:   {epoch_ckpt}")

        if args.enable_generation and args.preview_generations > 0:
            model.eval()
            preview_batch = next(iter(loader), None)
            if preview_batch and len(preview_batch["text"]) > 0:
                preview_n = min(args.preview_generations, len(preview_batch["text"]))
                generated_reports = model.generate_reports(
                    batch_pixel_values=preview_batch["pixel_values"][:preview_n],
                    batch_positions=preview_batch["positions"][:preview_n],
                    generation_tokenizer=generation_tokenizer,
                    device=device,
                    frame_chunk_size=args.frame_chunk_size,
                    max_new_tokens=args.generation_preview_max_new_tokens,
                    num_beams=args.generation_preview_num_beams,
                    do_sample=args.generation_preview_do_sample,
                    top_p=args.generation_preview_top_p,
                    temperature=args.generation_preview_temperature,
                )
                print("\n[INFO] Preview generations:")
                for i in range(preview_n):
                    print(f"--- sample {i + 1} / accession={preview_batch['acc'][i]}")
                    print(f"[GT ] {preview_batch['text'][i][:500]}")
                    print(f"[GEN] {generated_reports[i][:500]}")
            model.train()

        if args.enable_generation and args.run_holdout_generation_each_epoch:
            run_holdout_report_generation(
                model=model,
                args=args,
                processor=processor,
                generation_tokenizer=generation_tokenizer,
                device=device,
                epoch=epoch + 1,
                run_name=run_name,
            )

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
    ap.add_argument("--report_type_col", type=str, default="Type")
    ap.add_argument("--report_sampling", type=str, default="uniform", choices=["uniform"])
    ap.add_argument("--report_sampling_seed", type=int, default=42)

    ap.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k")
    ap.add_argument(
        "--bert_name",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Text encoder model name for the CLIP alignment branch.",
    )

    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--vision_backbone_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--text_lr", type=float, default=5e-5)
    ap.add_argument("--gen_head_lr", type=float, default=1e-4)
    ap.add_argument("--decoder_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--frame_chunk_size", type=int, default=16)
    ap.add_argument("--min_frames_per_sequence", type=int, default=1)
    ap.add_argument("--max_sequences_per_study", type=int, default=None)
    ap.add_argument("--max_frames_per_sequence", type=int, default=None)

    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--freeze_text", action="store_true")
    ap.add_argument("--freeze_decoder", action="store_true")

    ap.add_argument("--vit_trainable_blocks", type=int, default=3)
    ap.add_argument("--vit_unfreeze_patch_embed", action="store_true")

    ap.add_argument("--temporal_mode", type=str, default="sinusoidal", choices=TEMPORAL_MODE_CHOICES)
    ap.add_argument("--disable_frame_temporal", action="store_true")
    ap.add_argument("--enable_sequence_temporal", action="store_true")
    ap.add_argument("--frame_temporal_scale", type=float, default=0.25)
    ap.add_argument("--sequence_temporal_scale", type=float, default=0.25)

    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--out_dir", type=str, default="/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints")
    ap.add_argument("--output_dir", type=str, default="/data/Deep_Angiography/AngioVision/fine-tuning/output")
    ap.add_argument(
        "--val_data_dir",
        type=str,
        default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed",
        help="Validation data_dir passed to custom_framework_validate.py",
    )

    ap.add_argument("--keep_missing_reports", action="store_true")

    ap.add_argument("--pooling", type=str, default="max", choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling", type=str, default=None, choices=POOL_CHOICES)
    ap.add_argument("--sequence_pooling", type=str, default=None, choices=POOL_CHOICES)

    ap.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision.")
    ap.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    ap.add_argument("--vit_grad_ckpt", action="store_true", help="Enable ViT gradient checkpointing when vision trainable.")
    ap.add_argument("--logits_chunk", type=int, default=4, help="Chunk size for CLIP loss to avoid full BxB logits.")
    ap.add_argument("--vit_image_size", type=int, default=None)
    ap.add_argument("--empty_cache_each_step", action="store_true")

    ap.add_argument("--validate_script", type=str, default="custom_framework_validate_temporal.py")
    ap.add_argument("--calculate_score_script", type=str, default="calculate_score.py")
    ap.add_argument("--plot_loss_script", type=str, default="plot_loss.py")
    ap.add_argument("--validate_device", type=str, default=None)

    ap.add_argument("--enable_generation", action="store_true", help="Enable the visual-to-report generative decoder branch.")
    ap.add_argument("--decoder_model_name", type=str, default="gpt2", help="Decoder LM used for report generation.")
    ap.add_argument("--generation_max_length", type=int, default=256, help="Max training target length for reports in the generative branch.")
    ap.add_argument("--clip_loss_weight", type=float, default=1.0, help="Weight for the original CLIP-style contrastive loss.")
    ap.add_argument("--gen_loss_weight", type=float, default=1.0, help="Weight for the report generation loss.")
    ap.add_argument("--disable_generation_study_token", action="store_true", help="Do not prepend pooled study token to the decoder cross-attention memory.")

    ap.add_argument("--preview_generations", type=int, default=0, help="After each epoch, print up to N generated reports for quick inspection.")
    ap.add_argument("--generation_preview_max_new_tokens", type=int, default=128)
    ap.add_argument("--generation_preview_num_beams", type=int, default=1)
    ap.add_argument("--generation_preview_do_sample", action="store_true")
    ap.add_argument("--generation_preview_top_p", type=float, default=1.0)
    ap.add_argument("--generation_preview_temperature", type=float, default=1.0)


    ap.add_argument("--run_holdout_generation_each_epoch", action="store_true", help="After each epoch, generate reports on the holdout validation set grouped by accession/study.")
    ap.add_argument("--holdout_meta_csv", type=str, default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_23/consolidated_metadata_ALL_Sequences.csv")
    ap.add_argument("--holdout_base_frames_dir", type=str, default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_23/DICOM_Sequence_Processed")
    ap.add_argument("--report_generation_output_dir", type=str, default="/data/Deep_Angiography/AngioVision/fine-tuning/output/report_generation")
    ap.add_argument("--holdout_generation_batch_size", type=int, default=1)
    ap.add_argument("--holdout_generation_max_new_tokens", type=int, default=256)
    ap.add_argument("--holdout_generation_num_beams", type=int, default=1)
    ap.add_argument("--holdout_generation_do_sample", action="store_true")
    ap.add_argument("--holdout_generation_top_p", type=float, default=1.0)
    ap.add_argument("--holdout_generation_temperature", type=float, default=1.0)

    ap.add_argument("--holdout_anon_col", type=str, default="Anon Acc #")
    ap.add_argument("--holdout_sop_col", type=str, default="SOPInstanceUIDs")

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)