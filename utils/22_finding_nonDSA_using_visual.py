"""
find_non_dsa_sequences_parallel.py
---------------------------------
Parallelized version using ProcessPoolExecutor
"""

import argparse
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Frame loading
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_gray(path):
    try:
        if path.suffix.lower() == ".dcm" and PYDICOM_AVAILABLE:
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
        else:
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)

        return arr
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Frame stats
# ─────────────────────────────────────────────────────────────────────────────

def frame_stats(arr, border_pct):
    h, w = arr.shape
    bh = max(1, int(h * border_pct))
    bw = max(1, int(w * border_pct))
    crop = arr[bh:h - bh, bw:w - bw]

    if crop.size == 0:
        return None

    mean_val = float(crop.mean())
    bright_frac = float((crop >= 180).sum() / crop.size)

    return mean_val, bright_frac


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (MUST be top-level)
# ─────────────────────────────────────────────────────────────────────────────

def process_study(args):
    (
        study_dir,
        mean_thresh,
        frac_thresh,
        ratio_thresh,
        z_thresh,
        top_k,
        border_pct,
    ) = args

    frames_dir = study_dir / "frames"

    result = {
        "study": study_dir.name,
        "frames_dir": str(frames_dir),
        "frame_count": 0,
        "max_mean": 0.0,
        "median_mean": 0.0,
        "topk_mean": 0.0,
        "brightness_ratio": 0.0,
        "z_score": 0.0,
        "max_white_frac": 0.0,
        "has_white_frame": False,
        "trigger": "",
        "error": "",
    }

    if not frames_dir.exists():
        result["error"] = "no frames/"
        return result

    frame_files = sorted(f for f in frames_dir.iterdir() if f.is_file())
    result["frame_count"] = len(frame_files)

    means, fracs = [], []

    for f in frame_files:
        arr = load_frame_gray(f)
        if arr is None:
            continue

        stats = frame_stats(arr, border_pct)
        if stats is None:
            continue

        means.append(stats[0])
        fracs.append(stats[1])

    if not means:
        result["error"] = "no readable frames"
        return result

    means_arr = np.array(means)

    max_mean = float(means_arr.max())
    median_mean = float(np.median(means_arr))

    k = min(top_k, len(means_arr))
    topk_mean = float(np.sort(means_arr)[-k:].mean())

    ratio = topk_mean / median_mean if median_mean > 0 else 0.0

    std = float(np.std(means_arr))
    z_score = (max_mean - median_mean) / (std + 1e-6)

    max_frac = float(max(fracs))

    result.update({
        "max_mean": max_mean,
        "median_mean": median_mean,
        "topk_mean": topk_mean,
        "brightness_ratio": ratio,
        "z_score": z_score,
        "max_white_frac": max_frac,
    })

    # decision logic
    strong = 0
    weak = 0
    triggers = []

    if ratio >= ratio_thresh:
        strong += 1
        triggers.append(f"ratio≥{ratio_thresh:.2f}({ratio:.2f})")

    if z_score >= z_thresh:
        strong += 1
        triggers.append(f"z≥{z_thresh:.1f}({z_score:.2f})")

    if max_mean >= mean_thresh:
        weak += 1
        triggers.append(f"mean≥{mean_thresh}")

    if max_frac >= frac_thresh:
        weak += 1
        triggers.append(f"frac≥{frac_thresh}")

    if strong >= 1 or weak >= 2:
        result["has_white_frame"] = True
        result["trigger"] = " | ".join(triggers)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--mean_thresh", type=float, default=200)
    parser.add_argument("--frac_thresh", type=float, default=0.40)
    parser.add_argument("--ratio_thresh", type=float, default=1.8)
    parser.add_argument("--z_thresh", type=float, default=2.5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--border_pct", type=float, default=0.07)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="non_dsa_list.csv")

    args = parser.parse_args()

    root = Path(args.root)
    study_dirs = sorted(set(p.parent for p in root.rglob("frames")))

    print(f"\nFound {len(study_dirs)} sequences\n")

    all_results = []
    non_dsa = []

    task_args = [
        (
            study,
            args.mean_thresh,
            args.frac_thresh,
            args.ratio_thresh,
            args.z_thresh,
            args.top_k,
            args.border_pct,
        )
        for study in study_dirs
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_study, t) for t in task_args]

        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            all_results.append(res)

            if not res["error"] and not res["has_white_frame"]:
                non_dsa.append(res)

    print("\n==============================")
    print(f"Total: {len(all_results)}")
    print(f"Non-DSA: {len(non_dsa)}")
    print("==============================\n")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()