#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", type=str, required=True, help="Same as --out_dir used in training")
    ap.add_argument("--use", type=str, default="loss_ema", choices=["loss", "loss_ema"], help="What to plot")
    ap.add_argument("--max_points", type=int, default=0, help="If >0, downsample to at most this many points")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    log_path = ckpt_dir / "loss_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path}. Re-run training with loss logging enabled.")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError("loss_log.csv exists but is empty.")

    ycol = args.use
    if ycol not in df.columns:
        raise ValueError(f"{ycol} not found in columns: {list(df.columns)}")

    x = df["step"].values
    y = df[ycol].astype(float).values

    # Optional downsampling (simple stride)
    if args.max_points and len(df) > args.max_points:
        stride = max(1, len(df) // args.max_points)
        x = x[::stride]
        y = y[::stride]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Step")
    plt.ylabel("Contrastive loss" + (" (EMA)" if ycol == "loss_ema" else ""))
    plt.title("CLIP-style contrastive loss over training")

    out_png = ckpt_dir / "contrastive_loss_curve.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[OK] Saved: {out_png}")

if __name__ == "__main__":
    main()
