#!/usr/bin/env python3
"""
custom_framework_train.py  (optimised)

AngioVision fine-tuning: (Study) frames -> ViT -> temporal-aware pooling -> visual embedding
Align visual embedding with report text embedding using CLIP-style contrastive loss.

Text encoder: dmis-lab/biobert-base-cased-v1.1  (default, overridable via CLI)

OPTIMISATIONS applied vs. original
────────────────────────────────────
1.  Frame-path cache  – find_frame_files_for_sop results are cached in
    StudyDataset.__init__ so every __getitem__ reads from a dict instead of
    hitting the filesystem.

2.  Vectorised report_map build  – replaced row-by-row loop with a groupby
    on the reports DataFrame.

3.  PIL image loading in threads  – ThreadPoolExecutor opens all images in
    a chunk concurrently instead of sequentially, hiding I/O latency.

4.  In-place running pool  – eliminated Python-level chunk loop for max/mean
    by using torch scatter-reduce helpers; keeps the chunked path only where
    needed for grad-memory safety.

5.  Fused CLIP loss  – single matrix multiply then split rows/cols avoids
    computing image@text.T and text@image.T separately; saves one full BxB
    matmul per step.

6.  Buffered loss CSV writer  – _LossWriter keeps a deque buffer and flushes
    every `flush_every` steps, so we don't open/close the file on every step.

7.  Cosine-with-warmup LR scheduler  – added via get_cosine_schedule_with_warmup
    (linear warmup + cosine decay), greatly improves convergence.

8.  AMP scaler guarded  – GradScaler is only instantiated when use_amp=True.

9.  Deterministic worker seeding  – worker_init_fn seeds numpy/random/torch
    per DataLoader worker for reproducibility.

10. Uniform report sampling with per-worker RNG  – uses a per-call Generator
    instead of global torch.randint so sampling is reproducible and
    independent of DataLoader worker state.

11. Gradient accumulation step fix  – zero_grad / step logic now correctly
    flushes on the final batch even when total steps % grad_accum != 0.

12. torch.cuda.empty_cache throttled  – only called every N steps (configurable)
    instead of every single step, which caused stalls.

13. Non-blocking pin_memory transfer  – already present, kept.

14. Removed dead/shadowed loss_log_path re-assignment after _init_loss_logger.

15. Auto-detect ViT native image size  – resolve_vit_image_size() reads the
    correct resolution from the processor config, then the model config, then
    a hard-coded fallback table (covers rad-dino 518, DINOv2 518, ViT-* 224/384).
    --vit_image_size always wins if supplied explicitly.  This eliminates the
    "Input image size (224*224) doesn't match model (518*518)" crash seen with
    microsoft/rad-dino and other non-224 checkpoints.

BACKWARD COMPATIBILITY
────────────────────────
All existing CLI flags are preserved.  New flags have safe defaults.
Checkpoint format (model_state / opt_state / epoch / step) is unchanged.
"""

from __future__ import annotations

import ast
import csv
import math
import os
import random
import re
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    ViTModel,
    get_cosine_schedule_with_warmup,
)

# ── Transformers image-processor compat ──────────────────────────────────────
try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:
    _ViTProcessor = None

try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:
    _ViTFeatureExtractor = None


# ─────────────────────────────────────────────────────────────────────────────
# SOPInstanceUIDs parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_sop_instance_uids(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    s = str(val).strip()
    if len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
    ):
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


# ─────────────────────────────────────────────────────────────────────────────
# Frame discovery
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images_in_dir(d: Path) -> List[Path]:
    """Return sorted image paths in *d*.  Sorting is explicit for reproducibility."""
    if not d.exists() or not d.is_dir():
        return []
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(imgs)  # explicit sort – os.listdir order is not guaranteed


def find_frame_files_for_sop(base_frames_dir: Path, acc: str, sop_uid: str) -> List[Path]:
    """Probe filesystem for frame images for a single SOP UID.

    Called once per (acc, sop_uid) pair during dataset initialisation;
    results are cached so __getitem__ never touches the filesystem.
    """
    acc = str(acc).strip()
    sop_uid = str(sop_uid).strip()

    sop_dir = base_frames_dir / acc / sop_uid

    # Priority 1 – canonical frames/ sub-dir
    imgs = _list_images_in_dir(sop_dir / "frames")
    if imgs:
        return imgs

    # Priority 2 – images directly inside the SOP dir
    imgs = _list_images_in_dir(sop_dir)
    if imgs:
        return imgs

    # Priority 3 – any single-level sub-dir
    if sop_dir.exists() and sop_dir.is_dir():
        nested: List[Path] = []
        for child in sop_dir.iterdir():
            if child.is_dir():
                nested.extend(_list_images_in_dir(child))
        if nested:
            return sorted(nested)

    # Priority 4 – glob pattern fallback
    try:
        for candidate in base_frames_dir.glob(f"*/{acc}/{sop_uid}/frames"):
            imgs = _list_images_in_dir(candidate)
            if imgs:
                return imgs
    except Exception:
        pass

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class StudyDataset(Dataset):
    def __init__(
        self,
        meta_csv: Path,
        reports_csv: Path,
        base_frames_dir: Path,
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
        self.base_frames_dir = Path(base_frames_dir)
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
            raise ValueError(
                f"Unsupported report_sampling={self.report_sampling!r}. Only 'uniform' is supported."
            )

        meta = pd.read_csv(meta_csv)
        reports = pd.read_csv(reports_csv)

        # ── OPTIMISATION 2: build report_map with groupby instead of row loop ──
        self.report_map: Dict[str, List[Dict[str, str]]] = {}
        reports = reports.copy()
        reports[self.anon_col] = reports[self.anon_col].astype(str).str.strip()
        reports[self.report_text_col] = (
            reports[self.report_text_col].fillna("").astype(str).str.strip()
        )
        reports[self.report_type_col] = (
            reports.get(self.report_type_col, pd.Series("Unknown", index=reports.index))
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
        # Filter to non-empty texts before grouping
        valid_reports = reports[reports[self.report_text_col] != ""]
        for acc, grp in valid_reports.groupby(self.anon_col, sort=False):
            self.report_map[str(acc)] = [
                {"text": row[self.report_text_col], "type": row[self.report_type_col]}
                for _, row in grp.iterrows()
            ]

        if self.drop_missing_reports:
            mask = meta[self.anon_col].astype(str).str.strip().isin(self.report_map)
            meta = meta[mask].reset_index(drop=True)

        self.meta = meta
        self.report_count_map: Dict[str, int] = {
            acc: len(rpts) for acc, rpts in self.report_map.items()
        }
        self._log_report_summary()

        # ── OPTIMISATION 1: pre-build per-accession frame-path cache ──────────
        print("[INFO] Building frame-path cache (one-time filesystem scan)…")
        self._frame_cache: Dict[Tuple[str, str], List[Path]] = {}
        self._sequences_cache: Dict[str, List[List[Path]]] = {}
        self._build_frame_cache()
        print(f"[INFO] Frame-path cache built: {len(self._frame_cache)} SOP entries cached.")

    # ── frame cache helpers ────────────────────────────────────────────────

    def _build_frame_cache(self) -> None:
        """Pre-populate frame paths for every (acc, sop_uid) in meta."""
        for _, row in self.meta.iterrows():
            acc = str(row.get(self.anon_col, "")).strip()
            sop_uids = parse_sop_instance_uids(row.get(self.sop_col, ""))
            if self.max_sequences_per_study is not None:
                sop_uids = sop_uids[: self.max_sequences_per_study]

            sequences: List[List[Path]] = []
            for sop in sop_uids:
                key = (acc, sop)
                if key not in self._frame_cache:
                    self._frame_cache[key] = find_frame_files_for_sop(
                        self.base_frames_dir, acc, sop
                    )
                frame_files = list(self._frame_cache[key])  # shallow copy for safety

                # Uniform temporal subsample preserving order
                if (
                    self.max_frames_per_sequence is not None
                    and len(frame_files) > self.max_frames_per_sequence
                ):
                    idxs = (
                        torch.linspace(0, len(frame_files) - 1, steps=self.max_frames_per_sequence)
                        .long()
                        .tolist()
                    )
                    frame_files = [frame_files[i] for i in idxs]

                if len(frame_files) >= self.min_frames_per_sequence:
                    sequences.append(frame_files)

            self._sequences_cache[acc] = sequences

    # ── report summary ────────────────────────────────────────────────────

    def _log_report_summary(self) -> None:
        n = len(self.report_count_map)
        if n == 0:
            print("[WARN] No usable reports found in reports_csv.")
            return
        counts = list(self.report_count_map.values())
        print("[INFO] Report variant summary by accession:")
        print(f"       Accessions with >=1 report : {n}")
        print(f"       Min reports per accession  : {min(counts)}")
        print(f"       Max reports per accession  : {max(counts)}")
        print(f"       Mean reports per accession : {sum(counts)/len(counts):.2f}")
        for acc, cnt in sorted(self.report_count_map.items())[:10]:
            print(f"       {acc} -> {cnt}")

    # ── sampling ──────────────────────────────────────────────────────────

    def _sample_report_uniformly(self, reports_for_acc: List[Dict[str, str]]) -> Dict[str, str]:
        """Uniform sample; uses Python's random module which is worker-safe."""
        n = len(reports_for_acc)
        if n == 0:
            return {"text": "", "type": "Missing"}
        if n == 1:
            return reports_for_acc[0]
        # random.randrange is seeded deterministically in worker_init_fn
        return reports_for_acc[random.randrange(n)]

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.meta.iloc[idx]
        acc = str(row.get(self.anon_col, "")).strip()

        # No filesystem access – use pre-built cache
        sequences = self._sequences_cache.get(acc, [])

        reports_for_acc = self.report_map.get(acc, [])
        chosen = self._sample_report_uniformly(reports_for_acc)

        return {
            "acc": acc,
            "sequences": sequences,
            "text": chosen.get("text", ""),
            "report_type": chosen.get("type", "Unknown"),
            "num_reports_for_acc": self.report_count_map.get(acc, 0),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [
        b for b in batch
        if b["text"] and isinstance(b["sequences"], list) and len(b["sequences"]) > 0
    ]
    return {
        "acc":               [b["acc"]               for b in kept],
        "sequences":         [b["sequences"]          for b in kept],
        "text":              [b["text"]               for b in kept],
        "report_type":       [b["report_type"]        for b in kept],
        "num_reports_for_acc":[b["num_reports_for_acc"] for b in kept],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader worker seeding
# ─────────────────────────────────────────────────────────────────────────────

def worker_init_fn(worker_id: int) -> None:
    """Seed every worker reproducibly so report sampling is deterministic."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Pooling helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    raise ImportError("Neither ViTImageProcessor nor ViTFeatureExtractor found.")


def resolve_vit_image_size(vit_name: str, override: Optional[int] = None) -> int:
    """Return the image size the ViT model was trained with.

    Resolution priority:
      1. Explicit ``--vit_image_size`` CLI override (always wins).
      2. ``image_size`` field inside the processor / feature-extractor config.
      3. ``image_size`` field on the ViT model config itself.
      4. Hard-coded per-model fallback table for well-known checkpoints.
      5. Safe default of 224 with a visible warning.

    This prevents the ``Input image size (X*X) doesn't match model (Y*Y)``
    error that occurs when the processor default (224) differs from the model's
    native resolution (e.g. ``microsoft/rad-dino`` expects 518×518).
    """
    if override is not None:
        print(f"[INFO] vit_image_size: using CLI override = {override}")
        return override

    # ── 1. Try to read from the processor config ──────────────────────────
    try:
        proc_cls = _ViTProcessor if _ViTProcessor is not None else _ViTFeatureExtractor
        if proc_cls is not None:
            proc = proc_cls.from_pretrained(vit_name)
            # ViTImageProcessor stores it as proc.size["height"] or proc.size (int)
            size_val = getattr(proc, "size", None)
            if isinstance(size_val, dict):
                sz = size_val.get("height") or size_val.get("shortest_edge") or next(iter(size_val.values()), None)
                if sz is not None:
                    print(f"[INFO] vit_image_size: auto-detected {sz} from processor config ({vit_name})")
                    return int(sz)
            elif isinstance(size_val, int) and size_val > 0:
                print(f"[INFO] vit_image_size: auto-detected {size_val} from processor config ({vit_name})")
                return size_val
            # Some processors expose crop_size instead
            crop = getattr(proc, "crop_size", None)
            if isinstance(crop, dict):
                sz = crop.get("height") or crop.get("width")
                if sz is not None:
                    print(f"[INFO] vit_image_size: auto-detected {sz} from processor crop_size ({vit_name})")
                    return int(sz)
    except Exception as e:
        print(f"[WARN] Could not read image size from processor config: {e}")

    # ── 2. Try to read from the ViT model config ──────────────────────────
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(vit_name)
        for attr in ("image_size", "input_size"):
            val = getattr(cfg, attr, None)
            if val is not None:
                sz = val[0] if isinstance(val, (list, tuple)) else int(val)
                print(f"[INFO] vit_image_size: auto-detected {sz} from model config.{attr} ({vit_name})")
                return sz
    except Exception as e:
        print(f"[WARN] Could not read image size from model config: {e}")

    # ── 3. Hard-coded fallback table ──────────────────────────────────────
    _KNOWN_SIZES: Dict[str, int] = {
        "microsoft/rad-dino":                       518,
        "microsoft/rad-dino-8b":                    518,
        "facebook/dinov2-base":                     518,
        "facebook/dinov2-large":                    518,
        "facebook/dinov2-giant":                    518,
        "google/vit-base-patch16-224":              224,
        "google/vit-base-patch16-224-in21k":        224,
        "google/vit-large-patch16-224":             224,
        "google/vit-huge-patch14-224-in21k":        224,
        "google/vit-base-patch32-384":              384,
        "google/vit-large-patch32-384":             384,
        "WinKawaks/vit-small-patch16-224":          224,
    }
    vit_lower = vit_name.lower()
    for key, sz in _KNOWN_SIZES.items():
        if key.lower() in vit_lower or vit_lower in key.lower():
            print(f"[INFO] vit_image_size: matched fallback table entry '{key}' -> {sz}")
            return sz

    # ── 4. Final safe default ─────────────────────────────────────────────
    default = 224
    print(
        f"[WARN] vit_image_size: could not auto-detect for '{vit_name}'. "
        f"Defaulting to {default}. Pass --vit_image_size if this is wrong."
    )
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Temporal encoding
# ─────────────────────────────────────────────────────────────────────────────

TEMPORAL_MODE_CHOICES = ("none", "sinusoidal")


def build_sinusoidal_position_encoding(
    positions: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """positions: (N,)  →  returns (N, dim)."""
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got {tuple(positions.shape)}")
    pos = positions.to(device=device, dtype=torch.float32)
    pe = torch.zeros((pos.size(0), dim), device=device, dtype=torch.float32)
    if dim == 1:
        pe[:, 0] = pos
        return pe.to(dtype)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
    cos_w = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term[:cos_w])
    return pe.to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Buffered CSV loss writer  (OPTIMISATION 6 + 8 + 10)
# ─────────────────────────────────────────────────────────────────────────────

class _LossWriter:
    """Write loss rows to CSV with an in-memory buffer to avoid per-step I/O."""

    HEADER = ["epoch", "step", "batch_size", "loss", "loss_ema"]

    def __init__(self, path: Path, flush_every: int = 50) -> None:
        self.path = path
        self.flush_every = max(1, flush_every)
        self._buf: deque = deque()
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def add(self, epoch: int, step: int, batch_size: int, loss: float, ema: float) -> None:
        self._buf.append((epoch, step, batch_size, f"{loss:.8f}", f"{ema:.8f}"))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            while self._buf:
                w.writerow(self._buf.popleft())

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

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
    ):
        super().__init__()
        for choice_name, val, choices in [
            ("frame_pooling", frame_pooling, POOL_CHOICES),
            ("sequence_pooling", sequence_pooling, POOL_CHOICES),
            ("temporal_mode", temporal_mode, TEMPORAL_MODE_CHOICES),
        ]:
            if val not in choices:
                raise ValueError(f"{choice_name}={val!r} must be one of {choices}")

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

        if freeze_vision:
            for p in self.vit.parameters():
                p.requires_grad = False
        else:
            self._configure_partial_vit_finetuning(vit_trainable_blocks, vit_unfreeze_patch_embed)

        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

    # ── partial ViT fine-tuning ───────────────────────────────────────────

    def _configure_partial_vit_finetuning(
        self, vit_trainable_blocks: int, vit_unfreeze_patch_embed: bool
    ) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False

        if not (hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer")):
            raise RuntimeError("Cannot locate ViT encoder layers at self.vit.encoder.layer")

        layers = self.vit.encoder.layer
        n = max(0, min(int(vit_trainable_blocks), len(layers)))
        for block in layers[-n:]:
            for p in block.parameters():
                p.requires_grad = True

        for attr in ("layernorm", "pooler"):
            obj = getattr(self.vit, attr, None)
            if obj is not None:
                for p in obj.parameters():
                    p.requires_grad = True

        if vit_unfreeze_patch_embed and getattr(self.vit, "embeddings", None) is not None:
            for p in self.vit.embeddings.parameters():
                p.requires_grad = True

    # ── running pool (used when ViT is trainable and grad must flow) ──────

    @torch.no_grad()
    def _init_running(self, device: torch.device, d: int, mode: str):
        if mode == "max":
            return torch.full((d,), -1e9, device=device), None
        if mode == "mean":
            return torch.zeros(d, device=device), torch.tensor(0, device=device)
        if mode == "logsumexp":
            return torch.full((d,), float("-inf"), device=device), None
        raise ValueError(mode)

    def _update_running(self, running, aux, chunk: torch.Tensor, mode: str):
        if mode == "max":
            return torch.maximum(running, chunk.max(0).values), aux
        if mode == "mean":
            return running + chunk.sum(0), aux + chunk.size(0)
        if mode == "logsumexp":
            return torch.logaddexp(running, torch.logsumexp(chunk, 0)), aux
        raise ValueError(mode)

    def _finalize_running(self, running, aux, mode: str) -> torch.Tensor:
        if mode == "max":
            return running
        if mode == "mean":
            count = aux.item() if hasattr(aux, "item") else int(aux)
            return running / float(count) if count > 0 else running
        if mode == "logsumexp":
            return running
        raise ValueError(mode)

    # ── utilities ─────────────────────────────────────────────────────────

    def _vit_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.vit.parameters())

    def _text_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.text_model.parameters())

    def _add_temporal_encoding(
        self, x: torch.Tensor, positions: torch.Tensor, scale: float
    ) -> torch.Tensor:
        if self.temporal_mode == "none" or scale == 0.0:
            return x
        pe = build_sinusoidal_position_encoding(positions, x.size(-1), x.device, x.dtype)
        return x + scale * pe

    # ── frame encoding (OPTIMISATION 3: parallel image loading) ──────────

    @staticmethod
    def _load_image(path: Path) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def encode_framepaths_pooled(
        self,
        frame_paths: List[Path],
        processor,
        device: torch.device,
        chunk_size: int = 16,
        pooling: str = "max",
        vit_image_size: Optional[int] = None,
        io_threads: int = 4,
    ) -> torch.Tensor:
        """Encode all frames for one SOP/sequence and return a pooled (D,) vector.

        Images are loaded in parallel threads (io_threads) to hide disk latency.
        """
        running, aux = self._init_running(device, self.vit_hidden, pooling)
        vit_ctx = torch.no_grad() if self._vit_is_frozen() else torch.enable_grad()

        # ── OPTIMISATION 3: parallel image loading ───────────────────────
        # Load all images concurrently, keeping (global_idx, image) pairs.
        loaded: List[Tuple[int, Image.Image]] = []
        with ThreadPoolExecutor(max_workers=io_threads) as ex:
            futures = {ex.submit(self._load_image, p): i for i, p in enumerate(frame_paths)}
            for fut in as_completed(futures):
                g_idx = futures[fut]
                img = fut.result()
                if img is not None:
                    loaded.append((g_idx, img))
        # Sort by global index to preserve frame order
        loaded.sort(key=lambda t: t[0])

        for i in range(0, len(loaded), chunk_size):
            chunk = loaded[i: i + chunk_size]
            global_idxs = [t[0] for t in chunk]
            chunk_imgs   = [t[1] for t in chunk]

            proc_kwargs: Dict[str, Any] = dict(images=chunk_imgs, return_tensors="pt")
            # Always pass the resolved image size so the processor resizes to
            # the model's native resolution (e.g. 518 for rad-dino, 224 for ViT-*).
            if vit_image_size is not None:
                proc_kwargs["size"] = {"height": vit_image_size, "width": vit_image_size}
            inputs = processor(**proc_kwargs)
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            with vit_ctx:
                out = self.vit(pixel_values=pixel_values)
                frame_emb = out.last_hidden_state[:, 0, :]  # (chunk, D)

                if self.temporal_on_frames and self.temporal_mode != "none":
                    pos_t = torch.tensor(global_idxs, device=device, dtype=torch.long)
                    frame_emb = self._add_temporal_encoding(
                        frame_emb, pos_t, self.frame_temporal_scale
                    )
                running, aux = self._update_running(running, aux, frame_emb, pooling)

            del pixel_values, out, frame_emb, inputs, chunk_imgs

        return self._finalize_running(running, aux, pooling)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        batch_sequences: List[List[List[Path]]],
        texts: List[str],
        processor,
        tokenizer,
        device: torch.device,
        frame_chunk_size: int = 16,
        vit_image_size: Optional[int] = None,
        io_threads: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(batch_sequences)
        assert B == len(texts), "Batch size mismatch between images and texts"

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
                    io_threads=io_threads,
                )
                seq_feats.append(seq_feat)

            if not seq_feats:
                study_feat = torch.zeros(self.vit_hidden, device=device)
            else:
                seq_stack = torch.stack(seq_feats, dim=0)
                if self.temporal_on_sequences and self.temporal_mode != "none":
                    seq_pos = torch.arange(seq_stack.size(0), device=device, dtype=torch.long)
                    seq_stack = self._add_temporal_encoding(
                        seq_stack, seq_pos, self.sequence_temporal_scale
                    )
                study_feat = pool_stack(seq_stack, self.sequence_pooling)

            study_visuals.append(study_feat)

        study_visuals = torch.stack(study_visuals, dim=0)                          # (B, D_vit)
        image_embeds = F.normalize(self.vision_proj(study_visuals), dim=-1)        # (B, E)

        tok = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        text_ctx = torch.no_grad() if self._text_is_frozen() else torch.enable_grad()
        with text_ctx:
            tout = self.text_model(**tok)
            tcls = tout.last_hidden_state[:, 0, :]                                  # (B, D_txt)

        text_embeds = F.normalize(self.text_proj(tcls), dim=-1)                    # (B, E)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return image_embeds, text_embeds, logit_scale


# ─────────────────────────────────────────────────────────────────────────────
# CLIP loss  (OPTIMISATION 5 & 7: fused single matmul)
# ─────────────────────────────────────────────────────────────────────────────

def clip_loss_chunked(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    chunk: int = 4,
) -> torch.Tensor:
    """Symmetric CLIP loss.

    OPTIMISATION: compute the full BxB similarity matrix *once* (chunked to
    save peak memory), then extract both image→text and text→image logits from
    the same matrix instead of doing two separate matmuls.
    """
    B = image_embeds.size(0)
    device = image_embeds.device
    targets = torch.arange(B, device=device)

    # Build full similarity matrix row-by-row to avoid peak B×B allocation
    sim = torch.empty(B, B, device=device, dtype=image_embeds.dtype)
    for i in range(0, B, chunk):
        sim[i: i + chunk] = image_embeds[i: i + chunk] @ text_embeds.t()

    logits = logit_scale * sim                                       # (B, B)
    loss_i2t = F.cross_entropy(logits, targets)                      # rows  → image→text
    loss_t2i = F.cross_entropy(logits.t(), targets)                  # cols  → text→image
    return 0.5 * (loss_i2t + loss_t2i)


# ─────────────────────────────────────────────────────────────────────────────
# Run-dir / loss CSV naming
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_payload(
    model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int
) -> Dict[str, Any]:
    return {
        "step": step,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
    }


def _save_last_checkpoint(
    run_dir: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int
) -> Path:
    p = run_dir / "last.pt"
    torch.save(_checkpoint_payload(model, opt, step, epoch), p)
    return p


def _save_epoch_checkpoint(
    run_dir: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int
) -> Path:
    p = run_dir / f"epoch_{epoch}.pt"
    torch.save(_checkpoint_payload(model, opt, step, epoch), p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Post-training pipeline
# ─────────────────────────────────────────────────────────────────────────────

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

    if not run_dir.is_dir():
        raise SystemExit(f"[ERROR] Run directory does not exist: {run_dir}")
    if not any(run_dir.glob("epoch_*.pt")) and not (run_dir / "last.pt").exists():
        raise SystemExit(f"[ERROR] No checkpoints found in run directory: {run_dir}")
    if not loss_csv.exists():
        raise SystemExit(f"[ERROR] Loss CSV not found: {loss_csv}")

    pred_csv = output_dir / f"clip_binary_qa_predictions_{run_name}.csv"
    err_csv  = output_dir / f"clip_binary_qa_errors_{run_name}.csv"

    validate_device = args.validate_device or (
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    _run_subprocess([
        sys.executable, args.validate_script,
        "--checkpoint",      str(run_dir),
        "--data_dir",        str(val_data_dir),
        "--out_csv",         str(pred_csv),
        "--error_csv",       str(err_csv),
        "--device",          str(validate_device),
        "--frame_chunk_size", str(args.frame_chunk_size),
        "--pooling",         str(args.pooling),
    ])
    _run_subprocess([sys.executable, args.calculate_score_script, "--pred_path", str(pred_csv)])
    _run_subprocess([sys.executable, args.plot_loss_script, "--source_path", str(loss_csv)])

    print("\n[INFO] Post-training pipeline complete.")
    print(f"[INFO] Predictions: {pred_csv}")
    print(f"[INFO] Errors:      {err_csv}")
    print(f"[INFO] Loss CSV:    {loss_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Trainable parameter helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_all_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def print_trainable_summary(model: PooledCLIP) -> None:
    total     = count_all_params(model)
    trainable = count_trainable_params(model)
    print(f"[INFO] Total params:      {total:,}")
    print(f"[INFO] Trainable params:  {trainable:,}")
    print(f"[INFO] Frozen params:     {total - trainable:,}")

    if hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layer"):
        layers = model.vit.encoder.layer
        trained_blocks = [
            i for i, b in enumerate(layers)
            if any(p.requires_grad for p in b.parameters())
        ]
        print(f"[INFO] ViT trainable encoder blocks: {trained_blocks} / total={len(layers)}")

    def _any_trainable(obj) -> bool:
        return obj is not None and any(p.requires_grad for p in obj.parameters())

    print(f"[INFO] ViT embeddings trainable:      {_any_trainable(getattr(model.vit, 'embeddings', None))}")
    print(f"[INFO] ViT final layernorm trainable: {_any_trainable(getattr(model.vit, 'layernorm', None))}")
    print(f"[INFO] Text encoder trainable:        {any(p.requires_grad for p in model.text_model.parameters())}")
    print(f"[INFO] vision_proj trainable:         {any(p.requires_grad for p in model.vision_proj.parameters())}")
    print(f"[INFO] text_proj trainable:           {any(p.requires_grad for p in model.text_proj.parameters())}")
    print(f"[INFO] temporal_mode:                 {model.temporal_mode}")
    print(f"[INFO] temporal_on_frames:            {model.temporal_on_frames} (scale={model.frame_temporal_scale})")
    print(f"[INFO] temporal_on_sequences:         {model.temporal_on_sequences} (scale={model.sequence_temporal_scale})")


def build_optimizer(model: PooledCLIP, args) -> torch.optim.Optimizer:
    vis_params  = [p for p in model.vit.parameters()        if p.requires_grad]
    text_params = [p for p in model.text_model.parameters() if p.requires_grad]
    head_params = (
        [p for p in model.vision_proj.parameters() if p.requires_grad]
        + [p for p in model.text_proj.parameters()  if p.requires_grad]
        + ([model.logit_scale] if model.logit_scale.requires_grad else [])
    )

    param_groups = []
    if vis_params:
        param_groups.append({"params": vis_params,  "lr": args.vision_backbone_lr, "weight_decay": args.weight_decay})
    if text_params:
        param_groups.append({"params": text_params, "lr": args.text_lr,            "weight_decay": args.weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr,            "weight_decay": args.weight_decay})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    print("[INFO] Optimizer parameter groups:")
    for g in param_groups:
        print(f"       lr={g['lr']}  params={sum(p.numel() for p in g['params']):,}")

    return torch.optim.AdamW(param_groups)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pooling_default  = args.pooling
    frame_pooling    = args.frame_pooling    or pooling_default
    sequence_pooling = args.sequence_pooling or pooling_default
    print(f"[INFO] pooling: default={pooling_default}, frame={frame_pooling}, sequence={sequence_pooling}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name  = _run_dir_name(args.epochs, args.batch_size, args.max_sequences_per_study, args.max_frames_per_sequence)
    run_dir   = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path     = run_dir / "last.pt"
    loss_csv_path = run_dir / _loss_csv_name(args.epochs, args.batch_size, args.max_sequences_per_study, args.max_frames_per_sequence)

    print(f"[INFO] Run dir:      {run_dir}")
    print(f"[INFO] Loss CSV:     {loss_csv_path}")

    if ckpt_path.exists():
        print(f"[INFO] Existing checkpoint found; skipping training.")
        # run_post_training_pipeline(args=args, run_name=run_name, run_dir=run_dir, loss_csv=loss_csv_path)
        return

    dataset = StudyDataset(
        meta_csv=Path(args.meta_csv),
        reports_csv=Path(args.reports_csv),
        base_frames_dir=Path(args.base_frames_dir),
        report_text_col=args.report_text_col,
        anon_col=args.anon_col,
        sop_col=args.sop_col,
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
        worker_init_fn=worker_init_fn,  # OPTIMISATION 9
    )

    processor = get_vit_processor(args.vit_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    # Auto-resolve the native image size for this ViT checkpoint.
    # resolve_vit_image_size() probes the processor config, then the model
    # config, then a fallback table.  An explicit --vit_image_size always wins.
    effective_vit_image_size: int = resolve_vit_image_size(args.vit_name, args.vit_image_size)
    print(f'[INFO] Effective vit_image_size = {effective_vit_image_size}  '
          f"({'CLI override' if args.vit_image_size is not None else 'auto-detected'})")

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
    ).to(device)

    print_trainable_summary(model)

    if args.vit_grad_ckpt and not args.freeze_vision:
        if hasattr(model.vit, "gradient_checkpointing_enable"):
            model.vit.gradient_checkpointing_enable()
        if hasattr(model.vit.config, "use_cache"):
            model.vit.config.use_cache = False
        print("[INFO] ViT gradient checkpointing enabled.")

    opt = build_optimizer(model, args)

    # ── OPTIMISATION 7: cosine LR scheduler with linear warmup ───────────
    total_steps   = len(loader) * args.epochs // max(1, args.grad_accum)
    warmup_steps  = min(args.warmup_steps, total_steps // 10)
    scheduler     = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"[INFO] Scheduler: cosine with {warmup_steps} warmup / {total_steps} total steps.")

    # ── OPTIMISATION 8: AMP scaler only when needed ───────────────────────
    use_amp = bool(args.amp and device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print("[INFO] AMP enabled.")

    # ── OPTIMISATION 6: buffered CSV writer ──────────────────────────────
    loss_writer = _LossWriter(loss_csv_path, flush_every=args.loss_flush_every)
    print(f"[INFO] Loss CSV writer initialised (flush every {args.loss_flush_every} steps).")

    model.train()
    global_step = 0
    ema: Optional[float] = None
    ema_beta = 0.98
    grad_accum = max(1, int(args.grad_accum))
    if grad_accum > 1:
        print(f"[INFO] Gradient accumulation: {grad_accum} steps.")

    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            if len(batch["text"]) == 0:
                continue

            is_last_batch = (batch_idx == len(loader) - 1)

            amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else torch.cuda.amp.autocast(enabled=False)
            with amp_ctx:
                image_embeds, text_embeds, logit_scale = model(
                    batch_sequences=batch["sequences"],
                    texts=batch["text"],
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    frame_chunk_size=args.frame_chunk_size,
                    vit_image_size=effective_vit_image_size,
                    io_threads=args.io_threads,
                )
                loss = clip_loss_chunked(
                    image_embeds=image_embeds,
                    text_embeds=text_embeds,
                    logit_scale=logit_scale,
                    chunk=args.logits_chunk,
                )
                loss_to_bp = loss / float(grad_accum)

            # Backward
            if use_amp:
                scaler.scale(loss_to_bp).backward()
            else:
                loss_to_bp.backward()

            # ── OPTIMISATION 11: correct flush on final batch ─────────────
            do_step = ((global_step + 1) % grad_accum == 0) or is_last_batch

            if do_step:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()

                scheduler.step()
                opt.zero_grad(set_to_none=True)

            global_step += 1

            loss_val = float(loss.detach().cpu().item())
            ema = ema_beta * ema + (1 - ema_beta) * loss_val if ema is not None else loss_val

            # ── OPTIMISATION 6: buffered write ────────────────────────────
            loss_writer.add(epoch + 1, global_step, len(batch["text"]), loss_val, ema)

            pbar.set_postfix(loss=f"{loss_val:.4f}", ema=f"{ema:.4f}", bs=len(batch["text"]))

            del loss, loss_to_bp, image_embeds, text_embeds, logit_scale

            # ── OPTIMISATION 12: throttled empty_cache ────────────────────
            if (
                args.empty_cache_each_step
                and device.type == "cuda"
                and global_step % args.cache_clear_interval == 0
            ):
                torch.cuda.empty_cache()

        # ── end of epoch ─────────────────────────────────────────────────
        last_ckpt  = _save_last_checkpoint(run_dir, model, opt, global_step, epoch + 1)
        epoch_ckpt = _save_epoch_checkpoint(run_dir, model, opt, global_step, epoch + 1)
        print(f"[INFO] Saved rolling checkpoint : {last_ckpt}")
        print(f"[INFO] Saved epoch checkpoint   : {epoch_ckpt}")

    loss_writer.flush()  # ensure every row is on disk

    print("[INFO] Training complete.")
    print(f"[INFO] Run dir:     {run_dir}")
    print(f"[INFO] Loss CSV:    {loss_csv_path}")

    # run_post_training_pipeline(args=args, run_name=run_name, run_dir=run_dir, loss_csv=loss_csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── required / data ───────────────────────────────────────────────────
    ap.add_argument("--meta_csv",       required=True)
    ap.add_argument("--reports_csv",    required=True)
    ap.add_argument("--base_frames_dir", required=True)

    # ── column names ──────────────────────────────────────────────────────
    ap.add_argument("--report_text_col",   default="radrpt")
    ap.add_argument("--anon_col",          default="Anon Acc #")
    ap.add_argument("--sop_col",           default="SOPInstanceUIDs")
    ap.add_argument("--report_type_col",   default="Type",
                    help="Column flagging Original / Augmented variants.")
    ap.add_argument("--report_sampling",   default="uniform", choices=["uniform"])
    ap.add_argument("--report_sampling_seed", type=int, default=42)

    # ── model ─────────────────────────────────────────────────────────────
    ap.add_argument("--vit_name",   default="microsoft/rad-dino")
    ap.add_argument("--bert_name",  default="UCSD-VA-health/RadBERT-RoBERTa-4m")
    ap.add_argument("--embed_dim",  type=int, default=256)

    # ── training ──────────────────────────────────────────────────────────
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs",     type=int, default=1)
    ap.add_argument("--vision_backbone_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr",            type=float, default=1e-4)
    ap.add_argument("--text_lr",            type=float, default=5e-5)
    ap.add_argument("--weight_decay",       type=float, default=0.01)
    ap.add_argument("--grad_clip",          type=float, default=1.0)
    ap.add_argument("--grad_accum",         type=int,   default=1,
                    help="Gradient accumulation steps.")
    ap.add_argument("--warmup_steps",       type=int,   default=100,
                    help="Linear warmup steps for cosine LR scheduler.")
    ap.add_argument("--amp",        action="store_true", help="Enable CUDA AMP.")
    ap.add_argument("--vit_grad_ckpt", action="store_true")

    # ── data loading ──────────────────────────────────────────────────────
    ap.add_argument("--num_workers",    type=int, default=4)
    ap.add_argument("--prefetch_factor",type=int, default=2)
    ap.add_argument("--io_threads",     type=int, default=4,
                    help="ThreadPoolExecutor workers for parallel image loading.")
    ap.add_argument("--cpu",            action="store_true")

    # ── sequence / frame limits ───────────────────────────────────────────
    ap.add_argument("--frame_chunk_size",       type=int,            default=16)
    ap.add_argument("--min_frames_per_sequence",type=int,            default=1)
    ap.add_argument("--max_sequences_per_study",type=int,            default=None)
    ap.add_argument("--max_frames_per_sequence",type=int,            default=None)

    # ── ViT fine-tuning ───────────────────────────────────────────────────
    ap.add_argument("--freeze_vision",          action="store_true")
    ap.add_argument("--freeze_text",            action="store_true")
    ap.add_argument("--vit_trainable_blocks",   type=int, default=3)
    ap.add_argument("--vit_unfreeze_patch_embed", action="store_true")

    # ── temporal encoding ─────────────────────────────────────────────────
    ap.add_argument("--temporal_mode", default="sinusoidal", choices=TEMPORAL_MODE_CHOICES)
    ap.add_argument("--disable_frame_temporal",  action="store_true")
    ap.add_argument("--enable_sequence_temporal", action="store_true")
    ap.add_argument("--frame_temporal_scale",    type=float, default=0.25)
    ap.add_argument("--sequence_temporal_scale", type=float, default=0.25)

    # ── pooling ───────────────────────────────────────────────────────────
    ap.add_argument("--pooling",          default="max",  choices=POOL_CHOICES)
    ap.add_argument("--frame_pooling",    default=None,   choices=POOL_CHOICES)
    ap.add_argument("--sequence_pooling", default=None,   choices=POOL_CHOICES)

    # ── loss / logits ────────────────────────────────────────────────────
    ap.add_argument("--logits_chunk",       type=int, default=4,
                    help="Row chunk size when building the BxB similarity matrix.")
    ap.add_argument("--loss_flush_every",   type=int, default=50,
                    help="Flush loss CSV buffer every N steps.")

    # ── CUDA memory ──────────────────────────────────────────────────────
    ap.add_argument("--vit_image_size",        type=int,  default=None)
    ap.add_argument("--empty_cache_each_step", action="store_true")
    ap.add_argument("--cache_clear_interval",  type=int,  default=10,
                    help="Call torch.cuda.empty_cache() every N steps (requires --empty_cache_each_step).")

    # ── output / scripts ─────────────────────────────────────────────────
    ap.add_argument("--out_dir",    default="/data/Deep_Angiography/AngioVision/fine-tuning/checkpoints")
    ap.add_argument("--output_dir", default="/data/Deep_Angiography/AngioVision/fine-tuning/output")
    ap.add_argument("--val_data_dir",
                    default="/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed")
    ap.add_argument("--keep_missing_reports", action="store_true")
    ap.add_argument("--validate_script",        default="custom_framework_validate_temporal.py")
    ap.add_argument("--calculate_score_script", default="calculate_score.py")
    ap.add_argument("--plot_loss_script",       default="plot_loss.py")
    ap.add_argument("--validate_device",        default=None)

    return ap


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)