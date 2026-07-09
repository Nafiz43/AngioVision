#!/usr/bin/env python3
"""
split_gt.py - split the binary-QA ground-truth CSV into dev / test halves.

Splits by unique SOPInstanceUID (all questions of one sequence stay on the same
side) so that best-checkpoint selection (dev) and final reporting (test) never
see the same sequences:

    python3 split_gt.py --gt_path /path/to/gt.csv --dev_frac 0.5 --seed 42

Writes <gt>_dev.csv and <gt>_test.csv next to the input (override with
--out_dev / --out_test), then evaluate with:

    python3 validate.py --checkpoint <run_dir> \
        --selection_csv  /path/to/gt_dev.csv \
        --validation_csv /path/to/gt_test.csv \
        --out_csv preds.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Split a QA ground-truth CSV into dev/test by SOPInstanceUID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--gt_path", required=True, help="Input GT CSV (must have a SOPInstanceUID column).")
    ap.add_argument("--dev_frac", type=float, default=0.5, help="Fraction of SOPs assigned to the dev split.")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (fixed split for a given seed).")
    ap.add_argument("--out_dev", default=None, help="Output dev CSV (default: <gt>_dev.csv).")
    ap.add_argument("--out_test", default=None, help="Output test CSV (default: <gt>_test.csv).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_path)
    if not gt_path.exists():
        raise SystemExit(f"[ERROR] GT CSV does not exist: {gt_path}")
    if not 0.0 < args.dev_frac < 1.0:
        raise SystemExit(f"[ERROR] --dev_frac must be in (0, 1), got {args.dev_frac}")

    df = pd.read_csv(gt_path)
    if "SOPInstanceUID" not in df.columns:
        raise SystemExit(
            f"[ERROR] Column 'SOPInstanceUID' not found. Available: {list(df.columns)}"
        )

    sops = sorted(df["SOPInstanceUID"].astype(str).str.strip().unique())
    rng = random.Random(args.seed)
    rng.shuffle(sops)

    n_dev = max(1, min(len(sops) - 1, round(len(sops) * args.dev_frac)))
    dev_sops = set(sops[:n_dev])

    key = df["SOPInstanceUID"].astype(str).str.strip()
    dev_df = df[key.isin(dev_sops)]
    test_df = df[~key.isin(dev_sops)]

    out_dev = Path(args.out_dev) if args.out_dev else gt_path.with_name(f"{gt_path.stem}_dev.csv")
    out_test = Path(args.out_test) if args.out_test else gt_path.with_name(f"{gt_path.stem}_test.csv")
    dev_df.to_csv(out_dev, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"[INFO] Input : {gt_path}  ({len(df)} rows, {len(sops)} unique SOPs)")
    print(f"[INFO] Dev   : {out_dev}  ({len(dev_df)} rows, {n_dev} SOPs)")
    print(f"[INFO] Test  : {out_test}  ({len(test_df)} rows, {len(sops) - n_dev} SOPs)")
    print("[INFO] Rerun with the same --seed to reproduce the identical split.")


if __name__ == "__main__":
    main()
