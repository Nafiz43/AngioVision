"""
Step 02 — Statistics generation (02_stats_gen).

Two analyses share one distribution-report helper (this replaces the
near-duplicate plotting code that lived in 02_dicom_frame_stats.py and
08_sequence_dist.py):

    1. run()               — frame-count stats over the extracted sequences
                             (frame_statistics.csv + histogram/boxplot/Q-Q/
                             stats txt). Runs at pipeline position 02.
    2. analyze_instances() — 'Number of Instances' distribution from the
                             consolidated metadata CSV (ported from
                             08_sequence_dist.py). The consolidated CSV only
                             exists after step 04, so step 04 calls this
                             right after consolidating; outputs still land
                             under 02_stats_gen/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from urllib.parse import quote

import numpy as np
import pandas as pd

from vdp.common import (
    IMAGE_EXTENSIONS, distribution_report, find_sequence_dirs, write_csv,
)

STEP_DIRNAME = "02_stats_gen"
FRAME_THRESHOLD = 32
INSTANCES_THRESHOLD = 16


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / STEP_DIRNAME
    output_root = Path(cfg.dsa_sequences_root())  # potential-DSA subset (step 06)

    rows = []
    for seq_dir in find_sequence_dirs(output_root):
        frames_dir = seq_dir / "frames"
        n_frames = sum(
            1 for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        rows.append({
            "outer_dir_name": seq_dir.parent.name,
            "inner_dir_name": seq_dir.name,
            "number_of_frames": n_frames,
            "frames_dir_path": str(frames_dir),
            "frames_dir_url": "file://" + quote(str(frames_dir)),
        })
    rows.sort(key=lambda r: r["number_of_frames"], reverse=True)

    write_csv(step_dir / "frame_statistics.csv",
              ["outer_dir_name", "inner_dir_name", "number_of_frames",
               "frames_dir_path", "frames_dir_url"], rows)

    summary: Dict = {"sequences": len(rows)}
    if rows:
        values = np.array([r["number_of_frames"] for r in rows], dtype=float)
        distribution_report(values, "number_of_frames", step_dir,
                            "number_of_frames", threshold=FRAME_THRESHOLD)
        summary.update({
            "mean_frames": round(float(np.mean(values)), 2),
            "min_frames": int(np.min(values)),
            "max_frames": int(np.max(values)),
        })

    print(f"[02] {summary}")
    return summary


def analyze_instances(cfg, run_dir: Path, consolidated_csv: Path) -> Dict:
    """Instances-per-study distribution (from 08_sequence_dist.py)."""
    step_dir = run_dir / STEP_DIRNAME

    df = pd.read_csv(consolidated_csv)
    values = pd.to_numeric(df["Number of Instances"], errors="coerce").dropna()
    if values.empty:
        return {"instances_analysis": "no data"}

    values_np = values.to_numpy(dtype=float)
    bins = min(int(np.ceil(np.log2(len(values_np)))) + 1, 50) if len(values_np) > 1 else 10
    distribution_report(values_np, "Number of Instances", step_dir,
                        "number_of_instances",
                        threshold=INSTANCES_THRESHOLD, bins=bins)
    return {
        "instances_analyzed": len(values_np),
        "mean_instances": round(float(np.mean(values_np)), 2),
    }
