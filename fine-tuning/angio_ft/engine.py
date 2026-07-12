"""
angio_ft.engine
────────────────
Training loop, optimizer construction, checkpoint I/O, buffered loss logging and
the optional post-training validate/score/plot pipeline.

All numerics (cosine-warmup schedule, AMP, gradient accumulation, gradient
clipping, throttled cache clearing, checkpoint payload format) are preserved
from ``custom_framework_train_temporal.py``.  The only additions are:

  • ``--arch`` dispatch between the CLIP and SigLIP objectives, and
  • ablation-aware run-dir naming so ``clip/siglip x temporal`` runs with the
    same hyper-parameters land in distinct directories instead of overwriting
    each other.

Checkpoint payload is unchanged: ``{step, epoch, model_state, opt_state}``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shlex
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .common import get_vit_processor, resolve_vit_image_size
from .constants import ARCH_DEFAULT_VIT, arch_uses_sigmoid
from .contrastive_accum import gradcache_contrastive_step
from .data import StudyDataset, collate_fn, worker_init_fn
from .losses import clip_loss_chunked, siglip_loss
from .models import PooledCLIP, print_trainable_summary


# ─────────────────────────────────────────────────────────────────────────────
# Buffered CSV loss writer
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
# Per-epoch metrics CSV (train loss, val loss, QA metrics + baselines)
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_GROUPS = ("ORIGINAL", "FLIPPED", "ALL_YES", "ALL_NO", "RANDOM")
_METRIC_FIELDS = ("ACCURACY", "PRECISION", "RECALL", "F1", "TP", "TN", "FP", "FN")


class _EpochMetricsWriter:
    """One row per epoch, appended and flushed immediately so progress is
    preserved across interruptions (and across --resume, which appends)."""

    HEADER = ["epoch", "train_loss", "val_loss"] + [
        f"{g}_{f}" for g in _METRIC_GROUPS for f in _METRIC_FIELDS
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, int):
            return str(v)
        return f"{float(v):.8f}"

    def add(
        self,
        epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        qa_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        qa_metrics = qa_metrics or {}
        row = [str(epoch), self._fmt(train_loss), self._fmt(val_loss)]
        for g in _METRIC_GROUPS:
            for f in _METRIC_FIELDS:
                row.append(self._fmt(qa_metrics.get(f"{g}_{f}")))
        # Open-append-close per epoch: incremental by construction.
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
            f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Run-dir / loss CSV naming (ablation-aware)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_tag(args) -> str:
    """Compact tag describing whether temporal encoding is active for this run."""
    active = args.temporal_mode != "none" and (
        (not args.disable_frame_temporal) or args.enable_sequence_temporal
    )
    return "tempON" if active else "tempOFF"


# Args that do NOT change training numerics/results (bookkeeping, throughput,
# device placement). Everything else participates in the run-config hash so
# runs that differ in any result-affecting flag land in distinct directories.
_HASH_EXCLUDE = {
    "out_dir", "output_dir", "val_data_dir", "run_name", "resume", "force",
    "run_post_pipeline", "validate_script", "calculate_score_script",
    "plot_loss_script", "validate_device",
    "num_workers", "prefetch_factor", "io_threads", "cpu",
    "frame_chunk_size", "logits_chunk", "vit_grad_ckpt",
    "loss_flush_every", "empty_cache_each_step", "cache_clear_interval",
    "enable_generation",
    # Evaluation-only additions: they do not change training numerics.
    "epoch_qa_eval", "validation_csv",
}


def config_hash(args) -> str:
    """Short stable hash over every result-affecting CLI arg."""
    items = sorted(
        (k, repr(v)) for k, v in vars(args).items() if k not in _HASH_EXCLUDE
    )
    return hashlib.sha1(repr(items).encode("utf-8")).hexdigest()[:8]


def run_dir_name(
    arch: str,
    tmp_tag: str,
    epochs: int,
    batch_size: int,
    max_sequences_per_study: Optional[int],
    max_frames_per_sequence: Optional[int],
    cfg_hash: str = "",
) -> str:
    ms = "None" if max_sequences_per_study is None else str(max_sequences_per_study)
    mf = "None" if max_frames_per_sequence is None else str(max_frames_per_sequence)
    base = f"{arch}_{tmp_tag}_{epochs}_{batch_size}_{ms}_{mf}"
    return f"{base}_h{cfg_hash}" if cfg_hash else base


def seed_everything(seed: int) -> None:
    """Seed torch / numpy / random. DataLoader workers derive their seeds from
    torch.initial_seed() in worker_init_fn, so this makes report sampling and
    shuffling reproducible too."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Global seed set to {seed}")


def _save_train_command(run_dir: Path, args: argparse.Namespace) -> None:
    """Persist the exact CLI invocation + full resolved config for reproducibility."""
    cmd = " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv)
    sh = run_dir / "train_cmd.sh"
    sh.write_text(f"#!/usr/bin/env bash\n{cmd}\n")
    try:
        sh.chmod(sh.stat().st_mode | 0o111)
    except Exception:
        pass
    print(f"[INFO] Train command saved : {sh}")

    cfg_lines = [f"{k} = {v!r}" for k, v in sorted(vars(args).items())]
    cfg_path = run_dir / "train_config.txt"
    cfg_path.write_text("\n".join(cfg_lines) + "\n")
    print(f"[INFO] Full config saved   : {cfg_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_payload(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
    }
    # Additive, backward-compatible: old loaders read only `model_state`.
    if config is not None:
        payload["config"] = config
    return payload


def _save_last_checkpoint(run_dir: Path, model, opt, step: int, epoch: int, config=None) -> Path:
    p = run_dir / "last.pt"
    torch.save(_checkpoint_payload(model, opt, step, epoch, config), p)
    return p


def _save_epoch_checkpoint(run_dir: Path, model, opt, step: int, epoch: int, config=None) -> Path:
    p = run_dir / f"epoch_{epoch}.pt"
    torch.save(_checkpoint_payload(model, opt, step, epoch, config), p)
    return p


def build_model_config(args, frame_pooling: str, sequence_pooling: str, vit_image_size: int) -> Dict[str, Any]:
    """The exact set of fields needed to reconstruct PooledCLIP at eval time."""
    return {
        "arch": args.arch,
        "vit_name": args.vit_name,
        "bert_name": args.bert_name,
        "embed_dim": args.embed_dim,
        "frame_pooling": frame_pooling,
        "sequence_pooling": sequence_pooling,
        "temporal_mode": args.temporal_mode,
        "temporal_on_frames": (not args.disable_frame_temporal),
        "temporal_on_sequences": args.enable_sequence_temporal,
        "frame_temporal_scale": args.frame_temporal_scale,
        "sequence_temporal_scale": args.sequence_temporal_scale,
        "vit_image_size": vit_image_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Post-training pipeline (validate -> score -> plot)
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
    err_csv = output_dir / f"clip_binary_qa_errors_{run_name}.csv"

    validate_device = args.validate_device or (
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    validate_cmd = [
        sys.executable, args.validate_script,
        "--checkpoint", str(run_dir),
        "--data_dir", str(val_data_dir),
        "--out_csv", str(pred_csv),
        "--error_csv", str(err_csv),
        "--device", str(validate_device),
        "--frame_chunk_size", str(args.frame_chunk_size),
        "--pooling", str(args.pooling),
        "--arch", str(args.arch),
        "--vit_name", str(args.vit_name),
        "--bert_name", str(args.bert_name),
        "--embed_dim", str(args.embed_dim),
        "--temporal_mode", str(args.temporal_mode),
        "--frame_temporal_scale", str(args.frame_temporal_scale),
        "--sequence_temporal_scale", str(args.sequence_temporal_scale),
        "--calculate_score_script", str(args.calculate_score_script),
    ]
    # Forward the exact structural flags so validation rebuilds the same model.
    if args.disable_frame_temporal:
        validate_cmd.append("--disable_frame_temporal")
    if args.enable_sequence_temporal:
        validate_cmd.append("--enable_sequence_temporal")
    if args.vit_image_size is not None:
        validate_cmd += ["--vit_image_size", str(args.vit_image_size)]
    if args.frame_pooling:
        validate_cmd += ["--frame_pooling", str(args.frame_pooling)]
    if args.sequence_pooling:
        validate_cmd += ["--sequence_pooling", str(args.sequence_pooling)]
    _run_subprocess(validate_cmd)
    _run_subprocess([sys.executable, args.calculate_score_script, "--pred_path", str(pred_csv)])
    _run_subprocess([sys.executable, args.plot_loss_script, "--source_path", str(loss_csv)])

    print("\n[INFO] Post-training pipeline complete.")
    print(f"[INFO] Predictions: {pred_csv}")
    print(f"[INFO] Errors:      {err_csv}")
    print(f"[INFO] Loss CSV:    {loss_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: PooledCLIP, args) -> torch.optim.Optimizer:
    vis_params = [p for p in model.vit.parameters() if p.requires_grad]
    text_params = [p for p in model.text_model.parameters() if p.requires_grad]
    head_params = (
        [p for p in model.vision_proj.parameters() if p.requires_grad]
        + [p for p in model.text_proj.parameters() if p.requires_grad]
        + ([model.logit_scale] if model.logit_scale.requires_grad else [])
        + ([model.siglip_bias] if (model.siglip_bias is not None and model.siglip_bias.requires_grad) else [])
    )

    param_groups = []
    if vis_params:
        param_groups.append({"params": vis_params, "lr": args.vision_backbone_lr, "weight_decay": args.weight_decay})
    if text_params:
        param_groups.append({"params": text_params, "lr": args.text_lr, "weight_decay": args.weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay})

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
    print(f"[INFO] arch   = {args.arch}")

    # Resolve the per-arch default vision tower BEFORE the config hash is
    # computed so the run directory reflects the actual model trained.
    if args.vit_name is None:
        args.vit_name = ARCH_DEFAULT_VIT.get(args.arch, "microsoft/rad-dino")
        print(f"[INFO] --vit_name not given; defaulting to {args.vit_name} for --arch {args.arch}.")

    seed_everything(int(args.seed))

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pooling_default = args.pooling
    frame_pooling = args.frame_pooling or pooling_default
    sequence_pooling = args.sequence_pooling or pooling_default
    print(f"[INFO] pooling: default={pooling_default}, frame={frame_pooling}, sequence={sequence_pooling}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_tag = temporal_tag(args)
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = run_dir_name(
            args.arch, tmp_tag, args.epochs, args.batch_size,
            args.max_sequences_per_study, args.max_frames_per_sequence,
            cfg_hash=config_hash(args),
        )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _save_train_command(run_dir, args)

    ckpt_path = run_dir / "last.pt"
    loss_csv_path = run_dir / f"{run_name}_loss.csv"

    print(f"[INFO] Run dir:      {run_dir}")
    print(f"[INFO] Loss CSV:     {loss_csv_path}")

    resume_payload: Optional[Dict[str, Any]] = None
    if ckpt_path.exists():
        if args.resume:
            print(f"[INFO] --resume: loading state from {ckpt_path}")
            resume_payload = torch.load(ckpt_path, map_location="cpu")
        elif args.force:
            # Purge ALL stale checkpoints, not just last.pt. Otherwise a shorter
            # re-run (e.g. --epochs 3 over a previous --epochs 5 dir) leaves
            # epoch_4/epoch_5 from the OLD training, and validate.py would mix
            # two runs in its best-checkpoint comparison.
            stale = sorted(run_dir.glob("epoch_*.pt")) + [ckpt_path]
            stale = [p for p in stale if p.exists()]
            for p in stale:
                p.unlink()
            print(f"[WARN] --force: removed {len(stale)} stale checkpoint(s) in {run_dir}.")
        else:
            raise SystemExit(
                f"[ERROR] Run directory already contains checkpoints: {ckpt_path}\n"
                "        Use --resume to continue training, --force to retrain from "
                "scratch (overwrites), or --run_name to start a fresh run directory."
            )

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
        frame_keep_fraction=0.2 if getattr(args, "frames_20pct", False) else 1.0,
        frame_keep_seed=args.seed,
    )
    if getattr(args, "frames_20pct", False):
        print("[INFO] --20% active: training on a random 20% of frames per sequence.")

    # ── optional study-level train/val split (for per-epoch validation loss) ──
    train_data: torch.utils.data.Dataset = dataset
    val_loader: Optional[DataLoader] = None
    val_fraction = float(getattr(args, "val_fraction", 0.0) or 0.0)
    if val_fraction > 0.0:
        if not (0.0 < val_fraction < 1.0):
            raise SystemExit(f"[ERROR] --val_fraction must be in (0, 1), got {val_fraction}")
        n = len(dataset)
        n_val = int(round(n * val_fraction))
        n_val = max(1, min(n_val, n - 1))
        gen = torch.Generator().manual_seed(int(args.seed))
        perm = torch.randperm(n, generator=gen).tolist()
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_data = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        print(f"[INFO] Train/val split (seeded by --seed={args.seed}): "
              f"{len(train_idx)} train / {len(val_idx)} val studies.")

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )

    processor = get_vit_processor(args.vit_name)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)

    effective_vit_image_size: int = resolve_vit_image_size(args.vit_name, args.vit_image_size)
    print(f"[INFO] Effective vit_image_size = {effective_vit_image_size}  "
          f"({'CLI override' if args.vit_image_size is not None else 'auto-detected'})")

    model = PooledCLIP(
        vit_name=args.vit_name,
        text_model_name=args.bert_name,
        embed_dim=args.embed_dim,
        arch=args.arch,
        freeze_vision=args.freeze_vision,
        freeze_text=args.freeze_text,
        freeze_vision_proj=args.freeze_vision_proj,
        freeze_text_proj=args.freeze_text_proj,
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

    train_config = build_model_config(args, frame_pooling, sequence_pooling, effective_vit_image_size)

    if args.vit_grad_ckpt and not args.freeze_vision:
        if hasattr(model.vit, "gradient_checkpointing_enable"):
            model.vit.gradient_checkpointing_enable()
        if hasattr(model.vit.config, "use_cache"):
            model.vit.config.use_cache = False
        print("[INFO] ViT gradient checkpointing enabled.")

    opt = build_optimizer(model, args)

    total_steps = len(loader) * args.epochs // max(1, args.grad_accum)
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"[INFO] Scheduler: cosine with {warmup_steps} warmup / {total_steps} total steps.")

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print("[INFO] AMP enabled.")

    loss_writer = _LossWriter(loss_csv_path, flush_every=args.loss_flush_every)
    print(f"[INFO] Loss CSV writer initialised (flush every {args.loss_flush_every} steps).")

    # ── per-epoch metrics CSV (incremental; resume-safe append) ────────────
    epoch_metrics_csv = run_dir / f"{run_name}_epoch_metrics.csv"
    epoch_metrics = _EpochMetricsWriter(epoch_metrics_csv)
    print(f"[INFO] Epoch metrics CSV: {epoch_metrics_csv}")

    qa_eval_ctx: Optional[Dict[str, Any]] = None
    if getattr(args, "epoch_qa_eval", False):
        from . import qa_eval as _qa_mod
        val_gt_csv = getattr(args, "validation_csv", "") or _qa_mod.VALIDATION_CSV_FROM_SETTINGS
        if not val_gt_csv:
            raise SystemExit(
                "[ERROR] --epoch_qa_eval requires --validation_csv "
                "(or a central settings.py providing VALIDATION_CSV)."
            )
        qa_data_dir = Path(args.val_data_dir)
        if not qa_data_dir.exists():
            raise SystemExit(f"[ERROR] --epoch_qa_eval: validation data dir does not exist: {qa_data_dir}")
        qa_eval_ctx = {
            "module": _qa_mod,
            "gt_csv": str(val_gt_csv),
            "data_dir": str(qa_data_dir),
            "lookup": _qa_mod.load_validation_question_lookup(str(val_gt_csv)),
        }
        (run_dir / "epoch_eval").mkdir(exist_ok=True)
        print(f"[INFO] Per-epoch QA eval enabled: data={qa_data_dir} gt={val_gt_csv}")

    model.train()
    global_step = 0
    start_epoch = 0
    ema: Optional[float] = None
    ema_beta = 0.98
    grad_accum = max(1, int(args.grad_accum))
    if grad_accum > 1:
        print(f"[INFO] Gradient accumulation: {grad_accum} steps.")

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        opt.load_state_dict(resume_payload["opt_state"])
        global_step = int(resume_payload.get("step", 0))
        start_epoch = int(resume_payload.get("epoch", 0))
        # Fast-forward the LR schedule to where the optimizer left off.
        for _ in range(global_step // grad_accum):
            scheduler.step()
        print(f"[INFO] Resumed at epoch {start_epoch}, global step {global_step}.")
        if start_epoch >= args.epochs:
            print(f"[INFO] Checkpoint already covers {start_epoch} epochs "
                  f">= --epochs {args.epochs}; nothing to train.")
        del resume_payload

    def _amp_ctx():
        return torch.cuda.amp.autocast(enabled=use_amp)

    def _forward_fn(b):
        return model(
            batch_sequences=b["sequences"],
            texts=b["text"],
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            frame_chunk_size=args.frame_chunk_size,
            vit_image_size=effective_vit_image_size,
            io_threads=args.io_threads,
        )

    def _loss_fn(img, txt, logit_scale):
        if arch_uses_sigmoid(args.arch):
            return siglip_loss(
                image_embeds=img, text_embeds=txt, logit_scale=logit_scale,
                siglip_bias=model.siglip_bias, chunk=args.logits_chunk,
            )
        return clip_loss_chunked(
            image_embeds=img, text_embeds=txt, logit_scale=logit_scale,
            chunk=args.logits_chunk,
        )

    contrastive_accum = bool(getattr(args, "contrastive_accum", False))
    if contrastive_accum:
        print(f"[INFO] Contrastive accumulation ON (GradCache): effective negatives "
              f"≈ batch_size*grad_accum = {args.batch_size * grad_accum}.")

    opt.zero_grad(set_to_none=True)

    early_stop_patience = int(getattr(args, "early_stop_patience", 0) or 0)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
        gc_buffer: List[dict] = []
        epoch_loss_sum = 0.0
        epoch_loss_n = 0

        for batch_idx, batch in enumerate(pbar):
            if len(batch["text"]) == 0:
                continue

            is_last_batch = (batch_idx == len(loader) - 1)

            # ── GradCache path: batch_size*grad_accum acts as ONE big batch ──
            if contrastive_accum:
                gc_buffer.append(batch)
                if not ((len(gc_buffer) == grad_accum) or is_last_batch):
                    continue

                loss = gradcache_contrastive_step(
                    micro_batches=gc_buffer,
                    forward_fn=_forward_fn,
                    loss_fn=_loss_fn,
                    device=device,
                    amp_ctx_factory=_amp_ctx,
                    scaler=(scaler if use_amp else None),
                )

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

                total_bs = sum(len(b["text"]) for b in gc_buffer)
                global_step += len(gc_buffer)  # keep global_step = micro-batch count
                gc_buffer = []

                loss_val = float(loss.detach().cpu().item())
                ema = ema_beta * ema + (1 - ema_beta) * loss_val if ema is not None else loss_val
                loss_writer.add(epoch + 1, global_step, total_bs, loss_val, ema)
                epoch_loss_sum += loss_val * total_bs
                epoch_loss_n += total_bs
                pbar.set_postfix(loss=f"{loss_val:.4f}", ema=f"{ema:.4f}", bs=total_bs)

                del loss
                if (
                    args.empty_cache_each_step
                    and device.type == "cuda"
                    and global_step % args.cache_clear_interval == 0
                ):
                    torch.cuda.empty_cache()
                continue

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
                if arch_uses_sigmoid(args.arch):
                    loss = siglip_loss(
                        image_embeds=image_embeds,
                        text_embeds=text_embeds,
                        logit_scale=logit_scale,
                        siglip_bias=model.siglip_bias,
                        chunk=args.logits_chunk,
                    )
                else:
                    loss = clip_loss_chunked(
                        image_embeds=image_embeds,
                        text_embeds=text_embeds,
                        logit_scale=logit_scale,
                        chunk=args.logits_chunk,
                    )
                loss_to_bp = loss / float(grad_accum)

            if use_amp:
                scaler.scale(loss_to_bp).backward()
            else:
                loss_to_bp.backward()

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

            loss_writer.add(epoch + 1, global_step, len(batch["text"]), loss_val, ema)
            epoch_loss_sum += loss_val * len(batch["text"])
            epoch_loss_n += len(batch["text"])
            pbar.set_postfix(loss=f"{loss_val:.4f}", ema=f"{ema:.4f}", bs=len(batch["text"]))

            del loss, loss_to_bp, image_embeds, text_embeds, logit_scale

            if (
                args.empty_cache_each_step
                and device.type == "cuda"
                and global_step % args.cache_clear_interval == 0
            ):
                torch.cuda.empty_cache()

        last_ckpt = _save_last_checkpoint(run_dir, model, opt, global_step, epoch + 1, train_config)
        epoch_ckpt = _save_epoch_checkpoint(run_dir, model, opt, global_step, epoch + 1, train_config)
        print(f"[INFO] Saved rolling checkpoint : {last_ckpt}")
        print(f"[INFO] Saved epoch checkpoint   : {epoch_ckpt}")

        # ── per-epoch evaluation: validation loss + optional QA metrics ────
        loss_writer.flush()  # keep step-level CSV current before the eval pause
        train_loss_epoch = (epoch_loss_sum / epoch_loss_n) if epoch_loss_n else None

        val_loss_epoch: Optional[float] = None
        if val_loader is not None:
            model.eval()
            v_sum, v_n = 0.0, 0
            # The model uses nullcontext (not enable_grad) around its trainable
            # towers, so this no_grad() fully suppresses graph construction —
            # no activation retention during the validation pass.
            with torch.no_grad():
                for vbatch in val_loader:
                    if len(vbatch["text"]) == 0:
                        continue
                    with _amp_ctx():
                        v_img, v_txt, v_scale = _forward_fn(vbatch)
                        v_loss = _loss_fn(v_img, v_txt, v_scale)
                    v_sum += float(v_loss.detach().cpu().item()) * len(vbatch["text"])
                    v_n += len(vbatch["text"])
                    del v_loss, v_img, v_txt, v_scale
            model.train()
            val_loss_epoch = (v_sum / v_n) if v_n else None

        def _fmt_loss(v: Optional[float]) -> str:
            return "N/A" if v is None else f"{v:.6f}"
        print(f"[INFO] Epoch {epoch+1}: train_loss={_fmt_loss(train_loss_epoch)} "
              f"val_loss={_fmt_loss(val_loss_epoch)}")

        qa_metrics: Optional[Dict[str, Any]] = None
        if qa_eval_ctx is not None:
            eval_dir = run_dir / "epoch_eval"
            qa_metrics = qa_eval_ctx["module"].predict_and_score(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=device,
                data_dir_path=qa_eval_ctx["data_dir"],
                validation_lookup=qa_eval_ctx["lookup"],
                score_gt_csv=qa_eval_ctx["gt_csv"],
                out_csv=eval_dir / f"preds_epoch_{epoch + 1}.csv",
                error_csv=eval_dir / f"errors_epoch_{epoch + 1}.csv",
                label=f"epoch_{epoch + 1}",
                frame_chunk_size=args.frame_chunk_size,
                vit_image_size=effective_vit_image_size,
                calculate_score_script=args.calculate_score_script,
                random_seed=int(args.seed),
            )

        epoch_metrics.add(epoch + 1, train_loss_epoch, val_loss_epoch, qa_metrics)
        print(f"[INFO] Epoch metrics row appended: {epoch_metrics_csv}")

        if early_stop_patience and val_loss_epoch is not None:
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"[INFO] No val_loss improvement for {epochs_no_improve} epoch(s) "
                      f"(best={best_val_loss:.6f}, patience={early_stop_patience}).")
                if epochs_no_improve >= early_stop_patience:
                    print(f"[INFO] Early stopping at epoch {epoch + 1}.")
                    break

    loss_writer.flush()

    print("[INFO] Training complete.")
    print(f"[INFO] Run dir:     {run_dir}")
    print(f"[INFO] Loss CSV:    {loss_csv_path}")

    if getattr(args, "run_post_pipeline", False):
        run_post_training_pipeline(args=args, run_name=run_name, run_dir=run_dir, loss_csv=loss_csv_path)
