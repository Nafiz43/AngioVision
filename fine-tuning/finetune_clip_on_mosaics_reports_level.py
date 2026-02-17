#!/usr/bin/env python3
"""
finetune_clip_on_study_mosaic_grids.py

Train CLIP on STUDY-level merged mosaics:
  For each study (Anon Acc #):
    - multiple sequences (SOPInstanceUIDs)
    - each sequence has one mosaic:  DICOM_Sequence_Processed/<Anon Acc #>/<SOPInstanceUID>/mosaic.png
    - one report text per study

We merge the sequence mosaics into one big grid image and train CLIP with that + report.

Outputs:
  <output_dir>/
    resolved_pairs.csv
    train_log.csv
    loss_curve.png
    merged_mosaics/
    last/
    best/ (if val exists)
"""

import argparse
import ast
import csv
import hashlib
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt


# -----------------------------
# Misc utils
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_nonempty_str(x: Any) -> bool:
    if x is None:
        return False
    s = str(x).strip()
    return s != "" and s.lower() != "nan"


# -----------------------------
# Robust SOPInstanceUIDs parser
# -----------------------------
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
            pass  # fall back

    # Comma-separated
    return [tok.strip() for tok in s.split(",") if tok.strip()]


# -----------------------------
# Mosaic resolution (FIXED for your layout)
# -----------------------------
@dataclass
class ResolvedMosaic:
    uid: str
    acc: str
    mosaic_path: Optional[Path]
    ok: bool
    error: Optional[str] = None


def resolve_mosaic_for_acc_uid(
    base_path: Path,
    acc: str,
    uid: str,
    mosaic_name: str = "mosaic.png",
    mosaic_relative_dir: str = "",
    allow_global_uid_rglob: bool = False,
) -> ResolvedMosaic:
    """
    Matches your screenshot layout FIRST:
      base/<Anon Acc #>/<SOPInstanceUID>/mosaic.png

    Fallbacks:
      base/<Anon Acc #>/<SOPInstanceUID>/**/mosaic.png
      optional global rglob by UID (OFF by default; enable with --allow_global_uid_rglob)
    """
    acc = str(acc).strip()
    uid = str(uid).strip()

    if not acc or not uid:
        return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=None, ok=False, error="Empty acc/uid")

    # 1) EXACT expected path (your screenshot)
    uid_dir = base_path / acc / uid
    if mosaic_relative_dir:
        cand = uid_dir / mosaic_relative_dir / mosaic_name
    else:
        cand = uid_dir / mosaic_name

    if cand.exists():
        return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=cand, ok=True)

    # 2) If mosaic is nested somewhere under that UID dir (rare, but safe)
    if uid_dir.exists() and uid_dir.is_dir():
        try:
            hits = list(uid_dir.rglob(mosaic_name))
            if hits:
                return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=hits[0], ok=True)
        except Exception as e:
            return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=None, ok=False, error=f"rglob under uid_dir failed: {e}")

    # 3) Optional expensive global search (OFF unless requested)
    if allow_global_uid_rglob:
        try:
            for p in base_path.rglob(uid):
                if p.is_dir() and p.name == uid:
                    cand2 = (p / mosaic_relative_dir / mosaic_name) if mosaic_relative_dir else (p / mosaic_name)
                    if cand2.exists():
                        return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=cand2, ok=True)
                    hits = list(p.rglob(mosaic_name))
                    if hits:
                        return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=hits[0], ok=True)
        except Exception as e:
            return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=None, ok=False, error=f"global uid rglob failed: {e}")

    return ResolvedMosaic(uid=uid, acc=acc, mosaic_path=None, ok=False, error="Missing mosaic")


# -----------------------------
# Mosaic merging (grid)
# -----------------------------
def _stable_uid_hash(uids: List[str]) -> str:
    s = "|".join([str(u).strip() for u in uids if str(u).strip()])
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def merge_mosaics_grid(
    mosaic_paths: List[Path],
    tile_size: int = 256,
    grid_cols: Optional[int] = None,
    padding: int = 8,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    imgs: List[Image.Image] = []
    for p in mosaic_paths:
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((tile_size, tile_size))
            imgs.append(im)
        except Exception:
            continue

    if not imgs:
        return Image.new("RGB", (tile_size, tile_size), color=background)

    n = len(imgs)
    if grid_cols is None or grid_cols <= 0:
        grid_cols = int(math.ceil(math.sqrt(n)))
    grid_rows = int(math.ceil(n / grid_cols))

    out_w = grid_cols * tile_size + (grid_cols + 1) * padding
    out_h = grid_rows * tile_size + (grid_rows + 1) * padding
    canvas = Image.new("RGB", (out_w, out_h), color=background)

    for i, im in enumerate(imgs):
        r = i // grid_cols
        c = i % grid_cols
        x = padding + c * (tile_size + padding)
        y = padding + r * (tile_size + padding)
        canvas.paste(im, (x, y))

    return canvas


# -----------------------------
# Dataset
# -----------------------------
class StudyMosaicReportDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, processor: CLIPProcessor, image_size: Optional[int] = None):
        self.df = pairs_df.reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = Path(row["merged_mosaic_path"])
        text = str(row["text"])

        img = Image.open(img_path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size))

        return {"image": img, "text": text}


def collate_clip(batch: List[Dict[str, Any]], processor: CLIPProcessor) -> Dict[str, torch.Tensor]:
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return enc


# -----------------------------
# Training helpers
# -----------------------------
def clip_contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    scale = logit_scale.exp()
    logits_per_image = scale * (image_embeds @ text_embeds.t())
    logits_per_text = logits_per_image.t()

    n = image_embeds.size(0)
    targets = torch.arange(n, device=image_embeds.device)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2.0


@torch.no_grad()
def evaluate(model: CLIPModel, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(**batch, return_dict=True)
            loss = clip_contrastive_loss(out.image_embeds, out.text_embeds, model.logit_scale)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(model: CLIPModel, processor: CLIPProcessor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


# -----------------------------
# Build study-level pairs
# -----------------------------
def build_study_pairs_df(
    meta_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    base_path: Path,
    anon_col: str,
    sop_col: str,
    report_text_col: str,
    mosaic_name: str,
    mosaic_relative_dir: str,
    text_template: str,
    max_rows: Optional[int],
    drop_missing_text: bool,
    max_sequences_per_study: Optional[int],
    min_mosaics_per_study: int,
    cache_dir: Path,
    tile_size: int,
    grid_cols: Optional[int],
    padding: int,
    allow_global_uid_rglob: bool,
    verbose_errors_csv: Optional[Path],
) -> pd.DataFrame:
    # map acc -> report text
    report_map: Dict[str, str] = {}
    for _, r in reports_df.iterrows():
        acc = str(r.get(anon_col, "")).strip()
        txt = r.get(report_text_col, "")
        if isinstance(txt, float) and pd.isna(txt):
            txt = ""
        report_map[acc] = str(txt)

    err_writer = None
    err_fh = None
    if verbose_errors_csv is not None:
        verbose_errors_csv.parent.mkdir(parents=True, exist_ok=True)
        err_fh = open(verbose_errors_csv, "w", newline="")
        err_writer = csv.writer(err_fh)
        err_writer.writerow(["acc", "uid", "stage", "error"])

    def log_err(acc: str, uid: str, stage: str, error: str):
        if err_writer is not None:
            err_writer.writerow([acc, uid, stage, error])

    rows = []
    it = list(meta_df.iterrows())
    if max_rows is not None:
        it = it[:max_rows]

    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(it, desc="Building study mosaics", unit="study"):
        acc = str(row.get(anon_col, "")).strip()
        if not is_nonempty_str(acc):
            continue

        raw_report = report_map.get(acc, "")
        if drop_missing_text and (not is_nonempty_str(raw_report)):
            continue

        sop_uids = parse_sop_instance_uids(row.get(sop_col, ""))
        if max_sequences_per_study is not None:
            sop_uids = sop_uids[:max_sequences_per_study]
        if not sop_uids:
            log_err(acc, "", "parse_sops", "No SOPInstanceUIDs")
            continue

        resolved_paths: List[Path] = []
        used_uids: List[str] = []
        for uid in sop_uids:
            res = resolve_mosaic_for_acc_uid(
                base_path=base_path,
                acc=acc,
                uid=uid,
                mosaic_name=mosaic_name,
                mosaic_relative_dir=mosaic_relative_dir,
                allow_global_uid_rglob=allow_global_uid_rglob,
            )
            if res.ok and res.mosaic_path:
                resolved_paths.append(res.mosaic_path)
                used_uids.append(uid)
            else:
                log_err(acc, uid, "resolve_mosaic", res.error or "Unknown error")

        if len(resolved_paths) < min_mosaics_per_study:
            log_err(acc, "", "min_mosaics", f"Found {len(resolved_paths)} < min_mosaics_per_study={min_mosaics_per_study}")
            continue

        uid_hash = _stable_uid_hash(used_uids)
        merge_key = f"{acc}_{uid_hash}_t{tile_size}_c{grid_cols if grid_cols else 'auto'}_p{padding}.png"
        merged_path = cache_dir / merge_key

        if not merged_path.exists():
            try:
                big_img = merge_mosaics_grid(
                    mosaic_paths=resolved_paths,
                    tile_size=tile_size,
                    grid_cols=grid_cols,
                    padding=padding,
                )
                big_img.save(merged_path)
            except Exception as e:
                log_err(acc, "", "merge_save", str(e))
                continue

        txt = "" if raw_report is None else str(raw_report).strip()
        try:
            rendered = text_template.format(
                **{c: row.get(c, "") for c in meta_df.columns},
                TEXT=txt,
                ACC=acc,
            )
        except Exception:
            rendered = txt

        if drop_missing_text and (not is_nonempty_str(rendered)):
            log_err(acc, "", "render_text", "Rendered text empty")
            continue

        rows.append(
            {
                "ACC": acc,
                "merged_mosaic_path": str(merged_path),
                "text": rendered,
                "num_mosaics": int(len(resolved_paths)),
                "used_uids": "|".join(used_uids),
                "input_row_index": int(idx),
            }
        )

    if err_fh is not None:
        err_fh.close()

    return pd.DataFrame(rows)


def split_train_val(pairs: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(pairs) < 2:
        return pairs, pairs.iloc[0:0].copy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n_val = int(round(val_ratio * len(pairs)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_df = pairs.iloc[train_idx].reset_index(drop=True)
    val_df = pairs.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


# -----------------------------
# Plotting
# -----------------------------
def plot_loss_curve(log_csv: Path, out_png: Path) -> None:
    if not log_csv.exists() or log_csv.stat().st_size == 0:
        return

    df = pd.read_csv(log_csv)
    if "epoch" not in df.columns or "train_loss" not in df.columns:
        return

    df = df.sort_values("epoch")
    x = df["epoch"].tolist()
    y_train = df["train_loss"].tolist()
    y_val = df["val_loss"].tolist() if "val_loss" in df.columns else None

    plt.figure()
    plt.plot(x, y_train, label="train_loss")
    if y_val is not None and not all(pd.isna(y_val)):
        plt.plot(x, y_val, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CLIP fine-tuning loss (study merged mosaics)")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", type=Path, default=Path("/data/Deep_Angiography/DICOM_Sequence_Processed"))
    parser.add_argument("--meta_csv", type=Path, required=True)
    parser.add_argument("--reports_csv", type=Path, required=True)

    parser.add_argument("--anon_col", type=str, default="Anon Acc #")
    parser.add_argument("--sop_col", type=str, default="SOPInstanceUIDs")
    parser.add_argument("--report_text_col", type=str, default="radrpt")

    parser.add_argument("--mosaic_name", type=str, default="mosaic.png")
    parser.add_argument("--mosaic_relative_dir", type=str, default="")

    parser.add_argument("--text_template", type=str, default="{TEXT}")

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--drop_missing_text", action="store_true")
    parser.add_argument("--max_sequences_per_study", type=int, default=None)
    parser.add_argument("--min_mosaics_per_study", type=int, default=1)

    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--grid_cols", type=int, default=None)
    parser.add_argument("--grid_padding", type=int, default=8)

    parser.add_argument("--allow_global_uid_rglob", action="store_true")

    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume_from", type=Path, default=None)

    parser.add_argument("--errors_csv", type=Path, default=None)

    args = parser.parse_args()

    if not args.meta_csv.exists():
        raise FileNotFoundError(f"Meta CSV not found: {args.meta_csv}")
    if not args.reports_csv.exists():
        raise FileNotFoundError(f"Reports CSV not found: {args.reports_csv}")
    if not args.base_path.exists():
        raise FileNotFoundError(f"Base path not found: {args.base_path}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    if args.output_dir is None:
        output_root = Path(f"{args.base_path}_CLIP_STUDY_MOSAIC_Output")
    else:
        output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    merged_cache_dir = output_root / "merged_mosaics"

    meta_df = pd.read_csv(args.meta_csv)
    reports_df = pd.read_csv(args.reports_csv)

    for col in [args.anon_col, args.sop_col]:
        if col not in meta_df.columns:
            raise ValueError(f"Meta CSV missing required column '{col}'. Found: {list(meta_df.columns)}")
    if args.anon_col not in reports_df.columns:
        raise ValueError(f"Reports CSV missing anon_col '{args.anon_col}'. Found: {list(reports_df.columns)}")
    if args.report_text_col not in reports_df.columns:
        raise ValueError(f"Reports CSV missing report_text_col '{args.report_text_col}'. Found: {list(reports_df.columns)}")

    if args.resume_from is not None:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from}")
        processor = CLIPProcessor.from_pretrained(args.resume_from)
        model = CLIPModel.from_pretrained(args.resume_from)
        model_id_for_log = str(args.resume_from)
    else:
        processor = CLIPProcessor.from_pretrained(args.model_name)
        model = CLIPModel.from_pretrained(args.model_name)
        model_id_for_log = args.model_name

    model.to(device)

    if args.freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False
    if args.freeze_text:
        for p in model.text_model.parameters():
            p.requires_grad = False

    pairs = build_study_pairs_df(
        meta_df=meta_df,
        reports_df=reports_df,
        base_path=args.base_path,
        anon_col=args.anon_col,
        sop_col=args.sop_col,
        report_text_col=args.report_text_col,
        mosaic_name=args.mosaic_name,
        mosaic_relative_dir=args.mosaic_relative_dir,
        text_template=args.text_template,
        max_rows=args.max_rows,
        drop_missing_text=args.drop_missing_text,
        max_sequences_per_study=args.max_sequences_per_study,
        min_mosaics_per_study=args.min_mosaics_per_study,
        cache_dir=merged_cache_dir,
        tile_size=args.tile_size,
        grid_cols=args.grid_cols,
        padding=args.grid_padding,
        allow_global_uid_rglob=args.allow_global_uid_rglob,
        verbose_errors_csv=args.errors_csv,
    )

    if len(pairs) == 0:
        raise RuntimeError(
            "No training pairs were built. Common causes:\n"
            "- ACC/SOP columns mismatch (set --anon_col / --sop_col)\n"
            "- report text missing/empty and you used --drop_missing_text\n"
            "- mosaics not found under base_path (check --base_path, --mosaic_name)\n"
            "- min_mosaics_per_study too high\n"
        )

    resolved_csv = output_root / "resolved_pairs.csv"
    pairs.to_csv(resolved_csv, index=False)
    print(f"[INFO] Resolved pairs saved: {resolved_csv}")
    print(f"[INFO] Total usable studies: {len(pairs)}")
    print(f"[INFO] Merged mosaic cache dir: {merged_cache_dir}")

    train_df, val_df = split_train_val(pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] Train size: {len(train_df)} | Val size: {len(val_df)}")

    train_ds = StudyMosaicReportDataset(train_df, processor=processor, image_size=args.image_size)
    val_ds = StudyMosaicReportDataset(val_df, processor=processor, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_clip(b, processor),
        drop_last=True,
    )

    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_clip(b, processor),
            drop_last=False,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = output_root / "train_log.csv"
    if not log_path.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "batch_size",
                "grad_accum",
                "model_name",
                "meta_csv",
                "reports_csv",
                "anon_col",
                "sop_col",
                "report_text_col",
                "mosaic_name",
                "tile_size",
                "grid_cols",
                "grid_padding",
                "max_sequences_per_study",
                "min_mosaics_per_study",
                "text_template",
            ]
        ).to_csv(log_path, index=False)

    best_val = float("inf")

    print(f"[INFO] Device: {device} | AMP: {use_amp}")
    print(f"[INFO] Output: {output_root}")
    print(f"[INFO] Model: {model_id_for_log}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_losses: List[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(**batch, return_dict=True)
                loss = clip_contrastive_loss(out.image_embeds, out.text_embeds, model.logit_scale)
                loss = loss / max(1, args.grad_accum)

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_losses.append(float(loss.item() * max(1, args.grad_accum)))
            pbar.set_postfix(loss=float(np.mean(running_losses)), lr=float(scheduler.get_last_lr()[0]))

        train_loss = float(np.mean(running_losses)) if running_losses else float("inf")

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device=device, use_amp=use_amp)

        save_checkpoint(model, processor, output_root / "last")

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, processor, output_root / "best")

        log_row = {
            "timestamp": utc_timestamp(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "model_name": model_id_for_log,
            "meta_csv": str(args.meta_csv),
            "reports_csv": str(args.reports_csv),
            "anon_col": args.anon_col,
            "sop_col": args.sop_col,
            "report_text_col": args.report_text_col,
            "mosaic_name": args.mosaic_name,
            "tile_size": args.tile_size,
            "grid_cols": args.grid_cols if args.grid_cols is not None else "",
            "grid_padding": args.grid_padding,
            "max_sequences_per_study": args.max_sequences_per_study if args.max_sequences_per_study is not None else "",
            "min_mosaics_per_study": args.min_mosaics_per_study,
            "text_template": args.text_template,
        }
        pd.DataFrame([log_row]).to_csv(log_path, mode="a", header=False, index=False)

        plot_loss_curve(log_path, output_root / "loss_curve.png")

        msg = f"[EPOCH {epoch}] train_loss={train_loss:.4f}"
        if val_loader is not None:
            msg += f" val_loss={val_loss:.4f} best_val={best_val:.4f}"
        print(msg)

    plot_loss_curve(log_path, output_root / "loss_curve.png")

    print("\nDone ✔")
    print(f"- Resolved pairs: {output_root / 'resolved_pairs.csv'}")
    print(f"- Log CSV: {log_path}")
    print(f"- Loss figure: {output_root / 'loss_curve.png'}")
    print(f"- Last checkpoint: {output_root / 'last'}")
    if (output_root / "best").exists():
        print(f"- Best checkpoint: {output_root / 'best'}")
    if args.errors_csv is not None:
        print(f"- Errors CSV: {args.errors_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial checkpoints preserved.", file=sys.stderr)
        raise


# python3 finetune_clip_on_mosaics_reports_level.py \
#   --base_path /data/Deep_Angiography/DICOM_Sequence_Processed \
#   --meta_csv /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_GT.csv \
#   --reports_csv /data/Deep_Angiography/Reports/Report_List_v01_01.csv \
#   --mosaic_name mosaic.png \
#   --report_text_col radrpt \
#   --epochs 10 --batch_size 32 --lr 5e-6 \
#   --drop_missing_text \
#   --errors_csv ./checkpoints/study_mosaic_errors.csv
