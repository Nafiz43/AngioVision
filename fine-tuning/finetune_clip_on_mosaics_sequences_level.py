#!/usr/bin/env python3
"""
finetune_clip_on_mosaics.py

Goal
-----
Fine-tune a vanilla CLIP model on angiography MOSAICS using contrastive learning,
with ground-truth text supervision coming from CSV column: "Answer".

Also generates a train/validation loss figure from the logged CSV.

Assumptions
-----------
- Your mosaics are under:
    /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed
  in UID folders (possibly nested), and each UID has a mosaic image (default: mosaic.png).
- Your CSV has at least:
    - UID column (default: "UID")
    - Answer column (default: "Answer")  <-- ground truth supervision

Outputs
-------
Creates:
  <base_path>_CLIP_FT_Output/
    - resolved_pairs.csv
    - train_log.csv
    - loss_curve.png
    - last/ (checkpoint)
    - best/ (checkpoint, if val exists)

Example
-------
python finetune_clip_on_mosaics.py \
  --base_path /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/DICOM_Sequence_Processed \
  --in_csv /data/Deep_Angiography/Validation_Data/Validation_Data_2026_02_01/Test_Data_2026_02_01_v01.csv \
  --epochs 10 --batch_size 32 --lr 5e-6 \
  --drop_missing_text

Notes
-----
- Contrastive CLIP fine-tuning is only as good as your supervision text. If Answer is short
  ("Yes"/"No"), consider using --text_template to make it more descriptive.
"""

import argparse
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

# Matplotlib for loss curve figure
import matplotlib.pyplot as plt


# -----------------------------
# Utils
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
# UID -> mosaic resolution
# -----------------------------
@dataclass
class ResolvedMosaic:
    uid: str
    seq_dir: Optional[Path]
    mosaic_path: Optional[Path]
    ok: bool
    error: Optional[str] = None


def resolve_uid_dir(base_path: Path, uid: str) -> Optional[Path]:
    direct = base_path / uid
    if direct.exists() and direct.is_dir():
        return direct

    # fallback nested search
    try:
        for p in base_path.rglob("*"):
            if p.is_dir() and p.name == uid:
                return p
    except Exception:
        return None
    return None


def resolve_mosaic_for_uid(
    base_path: Path,
    uid: str,
    mosaic_name: str,
    mosaic_relative_dir: str = "",
) -> ResolvedMosaic:
    uid_dir = resolve_uid_dir(base_path, uid)
    if not uid_dir:
        return ResolvedMosaic(uid=uid, seq_dir=None, mosaic_path=None, ok=False, error="UID directory not found")

    candidate = uid_dir / mosaic_relative_dir / mosaic_name if mosaic_relative_dir else uid_dir / mosaic_name
    if candidate.exists():
        return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=candidate, ok=True, error=None)

    try:
        hits = list(uid_dir.rglob(mosaic_name))
        if hits:
            return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=hits[0], ok=True, error=None)
    except Exception:
        pass

    return ResolvedMosaic(uid=uid, seq_dir=uid_dir, mosaic_path=None, ok=False, error="Missing mosaic")


# -----------------------------
# Dataset
# -----------------------------
class MosaicTextDataset(Dataset):
    def __init__(self, pairs_df: pd.DataFrame, processor: CLIPProcessor, image_size: Optional[int] = None):
        self.df = pairs_df.reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = Path(row["mosaic_path"])
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
def clip_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
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
# Build training pairs
# -----------------------------
def build_pairs_df(
    df: pd.DataFrame,
    base_path: Path,
    uid_col: str,
    text_col: str,
    mosaic_name: str,
    mosaic_relative_dir: str,
    text_template: str,
    max_rows: Optional[int],
    drop_missing: bool,
) -> pd.DataFrame:
    rows = []
    it = list(df.iterrows())
    if max_rows is not None:
        it = it[:max_rows]

    for idx, row in tqdm(it, desc="Resolving mosaics", unit="row"):
        uid = str(row.get(uid_col, "")).strip()
        raw_text = row.get(text_col, "")

        if not is_nonempty_str(uid):
            continue
        if drop_missing and (not is_nonempty_str(raw_text)):
            continue

        resolved = resolve_mosaic_for_uid(
            base_path=base_path,
            uid=uid,
            mosaic_name=mosaic_name,
            mosaic_relative_dir=mosaic_relative_dir,
        )
        if not resolved.ok or not resolved.mosaic_path:
            continue

        txt = "" if raw_text is None else str(raw_text).strip()

        # Template can use any column keys + TEXT + UID
        try:
            rendered = text_template.format(**{c: row.get(c, "") for c in df.columns}, TEXT=txt, UID=uid)
        except Exception:
            rendered = txt

        if drop_missing and (not is_nonempty_str(rendered)):
            continue

        rows.append(
            {
                "UID": uid,
                "mosaic_path": str(resolved.mosaic_path),
                "text": rendered,
                "input_row_index": int(idx),
            }
        )

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
    plt.title("CLIP fine-tuning loss")
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

    parser.add_argument(
        "--base_path",
        type=Path,
        default=Path("/data/Deep_Angiography/DICOM_Sequence_Processed"),
    )
    parser.add_argument(
        "--in_csv",
        type=Path,
        default=Path("/data/Deep_Angiography/Reports/Report_List_v01_01.csv"),

    )

    parser.add_argument("--uid_col", type=str, default="UID")

    # Ground truth column is Answer (default)
    parser.add_argument(
        "--text_col",
        type=str,
        default="Answer",
        help="Column used as text supervision (default: Answer).",
    )

    parser.add_argument("--mosaic_name", type=str, default="mosaic.png")
    parser.add_argument("--mosaic_relative_dir", type=str, default="")

    parser.add_argument(
        "--text_template",
        type=str,
        default="{TEXT}",
        help="Format string. Uses row columns as keys + TEXT + UID. Default: '{TEXT}'.",
    )

    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", type=Path, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Options
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--drop_missing_text", action="store_true", help="Drop rows where Answer is empty/NaN")
    parser.add_argument("--image_size", type=int, default=None, help="Optional manual resize (e.g., 224)")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume_from", type=Path, default=None, help="Path to a saved checkpoint folder")

    args = parser.parse_args()

    if not args.in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.in_csv}")
    if not args.base_path.exists():
        raise FileNotFoundError(f"Base path not found: {args.base_path}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    # Output dir
    if args.output_dir is None:
        output_root = Path(f"{args.base_path}_CLIP_FT_Output")
    else:
        output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.in_csv)
    if args.uid_col not in df.columns:
        raise ValueError(f"Missing UID column '{args.uid_col}' in CSV.")
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text supervision column '{args.text_col}' in CSV.")

    # Load model/processor
    if args.resume_from is not None:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from}")
        processor = CLIPProcessor.from_pretrained(args.resume_from)
        model = CLIPModel.from_pretrained(args.resume_from)
    else:
        processor = CLIPProcessor.from_pretrained(args.model_name)
        model = CLIPModel.from_pretrained(args.model_name)

    model.to(device)

    # Freeze options
    if args.freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False
    if args.freeze_text:
        for p in model.text_model.parameters():
            p.requires_grad = False

    # Build resolved pairs
    pairs = build_pairs_df(
        df=df,
        base_path=args.base_path,
        uid_col=args.uid_col,
        text_col=args.text_col,  # default Answer
        mosaic_name=args.mosaic_name,
        mosaic_relative_dir=args.mosaic_relative_dir,
        text_template=args.text_template,
        max_rows=args.max_rows,
        drop_missing=args.drop_missing_text,
    )

    if len(pairs) == 0:
        raise RuntimeError(
            "No training pairs were built. Common causes:\n"
            "- UID folders not found under base_path\n"
            "- mosaic_name does not exist\n"
            "- Answer is empty/NaN and you used --drop_missing_text\n"
        )

    # Save resolved pairs
    resolved_csv = output_root / "resolved_pairs.csv"
    pairs.to_csv(resolved_csv, index=False)
    print(f"[INFO] Resolved pairs saved: {resolved_csv}")
    print(f"[INFO] Total usable pairs: {len(pairs)}")

    # Split
    train_df, val_df = split_train_val(pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[INFO] Train size: {len(train_df)} | Val size: {len(val_df)}")

    train_ds = MosaicTextDataset(train_df, processor=processor, image_size=args.image_size)
    val_ds = MosaicTextDataset(val_df, processor=processor, image_size=args.image_size)

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

    # Optimizer/scheduler
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

    # Logging
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
                "text_col",
                "text_template",
            ]
        ).to_csv(log_path, index=False)

    best_val = float("inf")

    print(f"[INFO] Device: {device} | AMP: {use_amp}")
    print(f"[INFO] Output: {output_root}")
    print(f"[INFO] Model: {args.model_name}")
    print(f"[INFO] Text supervision column: {args.text_col}")

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

        # Save last
        last_dir = output_root / "last"
        save_checkpoint(model, processor, last_dir)

        # Save best
        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            best_dir = output_root / "best"
            save_checkpoint(model, processor, best_dir)

        # Append log row
        log_row = {
            "timestamp": utc_timestamp(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "model_name": args.model_name if args.resume_from is None else str(args.resume_from),
            "text_col": args.text_col,
            "text_template": args.text_template,
        }
        pd.DataFrame([log_row]).to_csv(log_path, mode="a", header=False, index=False)

        # Update loss plot each epoch
        loss_png = output_root / "loss_curve.png"
        plot_loss_curve(log_path, loss_png)

        msg = f"[EPOCH {epoch}] train_loss={train_loss:.4f}"
        if val_loader is not None:
            msg += f" val_loss={val_loss:.4f} best_val={best_val:.4f}"
        print(msg)

    # Final plot (ensures it exists even if epochs=0 somehow)
    plot_loss_curve(log_path, output_root / "loss_curve.png")

    print("\nDone ✔")
    print(f"- Resolved pairs: {resolved_csv}")
    print(f"- Log CSV: {log_path}")
    print(f"- Loss figure: {output_root / 'loss_curve.png'}")
    print(f"- Last checkpoint: {output_root / 'last'}")
    if (output_root / "best").exists():
        print(f"- Best checkpoint: {output_root / 'best'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial checkpoints preserved.", file=sys.stderr)
        raise
