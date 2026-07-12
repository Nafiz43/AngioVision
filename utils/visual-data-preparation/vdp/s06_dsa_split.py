"""
Step 06 — Frame-based DSA identification + split.

Port of utils/05_dsa_identification_based_on_frame_v2.py (phases 1 and 2)
into the pipeline. The experiment-specific evaluation phases (1b/1c,
hand-labelled validation/test sets) were intentionally NOT ported — run
the original script for benchmark numbers.

Phase 1 — Calibration
    Learns detection thresholds from known-positive DSA sequences
    (cfg.dsa_calibration_roots, comma-separated). All statistics are
    computed on NON-BLACK pixels only (> BLACK_THRESHOLD). Per sequence,
    the frame with the lowest active-pixel std is the mask candidate;
    thresholds come from the candidate distributions with a 15% margin.

    Thresholds are cached in cfg.dsa_thresholds_json (stable across runs).
    When a valid cache exists — complete threshold set, calibrated on the
    same dsa_calibration_roots — calibration is SKIPPED and the cached
    values are reused. Recalibration happens when the cache is missing,
    incomplete, was built from different roots, or cfg.dsa_recalibrate
    is set.

Phase 2 — Detection + split
    Every extracted sequence under cfg.output_root is classified and
    copied (frames/ + metadata.csv + mosaic.png, whole dir) into
        <dsa_split_root>/00_potential_dsas/<accession>/<sop>/
        <dsa_split_root>/01_potential_non_dsas/<accession>/<sop>/
    The split is exhaustive: anything not flagged potential_dsa
    (including unreadable/all-black sequences) lands in 01_potential_non_dsas.

Reports (under <run_dir>/06_dsa_split/):
    calibration_stats.csv    per-frame active-pixel stats
    calibration_summary.txt  derived thresholds
    dsa_split_report.csv     per-sequence verdict + copy status
"""

from __future__ import annotations

import datetime
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import POTENTIAL_DSA_DIRNAME, POTENTIAL_NON_DSA_DIRNAME
from vdp.common import IMAGE_EXTENSIONS, find_sequence_dirs, write_csv

# Pixels at or below this value are treated as black border and excluded.
BLACK_THRESHOLD = 10
# 15% headroom on top of raw percentiles so borderline frames are not missed.
MARGIN = 1.15

THRESHOLD_KEYS = (
    "MASK_MAX_STD", "MASK_MAX_MEAN", "MASK_MIN_MEAN",
    "MASK_MAX_BRIGHT_FRAC", "MASK_MAX_ENTROPY",
)


# =========================================================
# Active-pixel statistics (workers — must stay module-level)
# =========================================================
def _active_pixels(arr: np.ndarray) -> np.ndarray:
    return arr[arr > BLACK_THRESHOLD]


def _pixel_entropy(pixels: np.ndarray) -> float:
    """Shannon entropy (bits); DSA mask frames are near-uniform → low entropy."""
    if pixels.size == 0:
        return 0.0
    counts, _ = np.histogram(pixels, bins=256, range=(0, 255))
    counts = counts[counts > 0].astype(np.float64)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _frame_stats(img_path_str: str) -> Optional[Dict]:
    try:
        arr = np.array(Image.open(img_path_str).convert("L"), dtype=np.float32)
        active = _active_pixels(arr)
        if active.size == 0:
            return None
        return {
            "path": img_path_str,
            "mean": float(np.mean(active)),
            "std": float(np.std(active)),
            "bright_frac": float(np.sum(active > 220)) / active.size,
            "entropy": _pixel_entropy(active),
            "active_frac": float(active.size) / arr.size,
        }
    except Exception:
        return None


def _is_mask_frame(img_path_str: str, thresholds: Dict[str, float]) -> bool:
    try:
        arr = np.array(Image.open(img_path_str).convert("L"), dtype=np.float32)
        active = _active_pixels(arr)
        if active.size == 0:
            return False
        mean_val = float(np.mean(active))
        std_val = float(np.std(active))
        bright_frac = float(np.sum(active > 220)) / active.size
        entropy_val = _pixel_entropy(active)
        return (
            thresholds["MASK_MIN_MEAN"] <= mean_val <= thresholds["MASK_MAX_MEAN"]
            and std_val <= thresholds["MASK_MAX_STD"]
            and bright_frac <= thresholds["MASK_MAX_BRIGHT_FRAC"]
            and entropy_val <= thresholds["MASK_MAX_ENTROPY"]
        )
    except Exception:
        return False


def _analyse_sequence(args: Tuple[str, Dict[str, float]]) -> Dict:
    """
    Worker: classify one sequence dir. Mirrors calibration exactly —
    lowest active-pixel-std frame is the mask candidate; only it is tested.
    """
    seq_dir_str, thresholds = args
    frames_dir = Path(seq_dir_str) / "frames"

    result = {
        "sequence_dir": seq_dir_str,
        "total_frames": 0,
        "candidate_frame": "",
        "candidate_std": "",
        "candidate_mean": "",
        "candidate_bright": "",
        "candidate_entropy": "",
        "verdict": "no_mask_detected",
        "split_dir": "",
        "copied": False,
    }

    if not frames_dir.is_dir():
        result["verdict"] = "skipped_no_frames_dir"
        return result

    frame_files = sorted(
        str(f) for f in frames_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    result["total_frames"] = len(frame_files)
    if not frame_files:
        result["verdict"] = "skipped_no_frames"
        return result

    stats_list = [s for p in frame_files if (s := _frame_stats(p)) is not None]
    if not stats_list:
        result["verdict"] = "skipped_all_black"
        return result

    candidate = min(stats_list, key=lambda s: s["std"])
    result["candidate_frame"] = Path(candidate["path"]).name
    result["candidate_std"] = round(candidate["std"], 4)
    result["candidate_mean"] = round(candidate["mean"], 4)
    result["candidate_bright"] = round(candidate["bright_frac"], 4)
    result["candidate_entropy"] = round(candidate["entropy"], 4)

    if _is_mask_frame(candidate["path"], thresholds):
        result["verdict"] = "potential_dsa"
    return result


# =========================================================
# Threshold cache — skip calibration when already calibrated
# =========================================================
def _normalized_roots(raw: str) -> List[str]:
    """Configured calibration roots as a sorted list (existence NOT checked)."""
    return sorted({chunk.strip() for chunk in raw.split(",") if chunk.strip()})


def _load_cached_thresholds(cache_path: Path,
                            configured_roots: List[str]) -> Optional[Dict[str, float]]:
    """
    Return the cached thresholds if — and only if — the cache is usable:
    file exists, parses as JSON, holds every threshold key as a number,
    and was calibrated on exactly the currently configured roots.
    Any other condition means "not calibrated" -> None.
    """
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[06] WARN unreadable thresholds cache ({e}) — recalibrating.")
        return None

    thresholds = cached.get("thresholds", {})
    if not all(isinstance(thresholds.get(k), (int, float)) for k in THRESHOLD_KEYS):
        print(f"[06] Thresholds cache incomplete ({cache_path}) — recalibrating.")
        return None
    if cached.get("calibration_roots") != configured_roots:
        print("[06] Thresholds cache was calibrated on different "
              "dsa_calibration_roots — recalibrating.")
        return None
    return {k: float(thresholds[k]) for k in THRESHOLD_KEYS}


def _save_thresholds_cache(cache_path: Path, thresholds: Dict[str, float],
                           configured_roots: List[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "calibrated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "calibration_roots": configured_roots,
        "black_threshold": BLACK_THRESHOLD,
        "margin": MARGIN,
        "thresholds": thresholds,
    }, indent=2) + "\n", encoding="utf-8")


# =========================================================
# Phase 1 — calibration
# =========================================================
def _parse_calibration_roots(raw: str) -> List[Path]:
    roots = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        root = Path(chunk)
        if root.is_dir():
            roots.append(root)
        else:
            print(f"[06] WARN calibration root not found, skipping: {root}")
    return roots


def _collect_frame_paths(root: Path) -> List[str]:
    paths = []
    for frames_dir in root.rglob("frames"):
        if frames_dir.is_dir():
            for f in frames_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(f))
    return paths


def _calibrate(roots: List[Path], workers: int, step_dir: Path) -> Dict[str, float]:
    frame_paths: List[str] = []
    for root in roots:
        found = _collect_frame_paths(root)
        print(f"[06]   {len(found):>8,} frames  <- {root}")
        frame_paths.extend(found)
    if not frame_paths:
        raise RuntimeError(
            "No image frames found under any dsa_calibration_roots — "
            "cannot learn DSA thresholds. Fix dsa_calibration_roots "
            "(comma-separated dirs, each holding <seq>/frames/*.png)."
        )

    all_stats: List[Dict] = []
    with tqdm(total=len(frame_paths), unit="frame",
              desc="[06] Calibrating") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_frame_stats, p) for p in frame_paths]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    all_stats.append(res)
                pbar.update(1)

    write_csv(step_dir / "calibration_stats.csv",
              ["path", "mean", "std", "bright_frac", "entropy", "active_frac"],
              all_stats)

    # One mask candidate per sequence = frame with lowest active-pixel std.
    groups: Dict[str, List[Dict]] = {}
    for s in all_stats:
        groups.setdefault(str(Path(s["path"]).parent), []).append(s)
    if not groups:
        raise RuntimeError("All calibration frames were unreadable or fully black.")
    candidates = [min(frames, key=lambda s: s["std"]) for frames in groups.values()]

    stds = np.array([c["std"] for c in candidates])
    means = np.array([c["mean"] for c in candidates])
    bright_fracs = np.array([c["bright_frac"] for c in candidates])
    entropies = np.array([c["entropy"] for c in candidates])

    thresholds = {
        "MASK_MAX_STD": round(float(np.percentile(stds, 100)) * MARGIN, 2),
        "MASK_MAX_MEAN": round(min(255.0, float(np.percentile(means, 100)) * MARGIN), 2),
        "MASK_MIN_MEAN": round(float(np.percentile(means, 3)) / MARGIN, 2),
        "MASK_MAX_BRIGHT_FRAC": round(
            min(1.0, float(np.percentile(bright_fracs, 100)) * MARGIN), 4),
        "MASK_MAX_ENTROPY": round(float(np.percentile(entropies, 100)) * MARGIN, 4),
    }

    lines = [
        "Step 06 — DSA split calibration summary",
        "=" * 55,
        f"Black-pixel threshold : pixel value <= {BLACK_THRESHOLD} (excluded)",
        "Calibration roots:",
        *(f"    {r}" for r in roots),
        f"Frames scanned  : {len(all_stats):,}",
        f"Sequences       : {len(groups):,}",
        f"Margin applied  : {MARGIN}x",
        "",
        "Derived thresholds",
        "-" * 40,
        *(f"  {k:22s} = {v}" for k, v in thresholds.items()),
    ]
    (step_dir / "calibration_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    for k, v in thresholds.items():
        print(f"[06]   {k:22s} = {v}")
    return thresholds


# =========================================================
# Phase 2 — detection + split copy
# =========================================================
def _copy_sequence(seq_dir: Path, dest_root: Path) -> Tuple[bool, str]:
    dest_dir = dest_root / seq_dir.parent.name / seq_dir.name
    try:
        if not dest_dir.exists():
            shutil.copytree(seq_dir, dest_dir)
        return True, str(dest_dir)
    except Exception as e:
        print(f"[06] COPY ERROR {seq_dir}: {e}")
        return False, str(dest_dir)


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / "06_dsa_split"
    step_dir.mkdir(parents=True, exist_ok=True)

    output_root = Path(cfg.output_root).resolve()
    split_root = Path(cfg.dsa_split_root).resolve()
    if split_root == output_root or output_root in split_root.parents:
        raise ValueError(
            f"dsa_split_root ({split_root}) must not be inside "
            f"output_root ({output_root}) — the copies would be re-scanned "
            "as sequences on the next run."
        )

    # ── Phase 1: calibration (skipped when already calibrated) ───────────
    cache_path = Path(cfg.dsa_thresholds_json)
    configured_roots = _normalized_roots(cfg.dsa_calibration_roots)

    thresholds = None
    if getattr(cfg, "dsa_recalibrate", False):
        print("[06] dsa_recalibrate set — ignoring thresholds cache.")
    else:
        thresholds = _load_cached_thresholds(cache_path, configured_roots)

    if thresholds is not None:
        thresholds_source = f"cached ({cache_path})"
        print(f"[06] Already calibrated — reusing thresholds from {cache_path}")
        for k in THRESHOLD_KEYS:
            print(f"[06]   {k:22s} = {thresholds[k]}")
    else:
        roots = _parse_calibration_roots(cfg.dsa_calibration_roots)
        if not roots:
            raise RuntimeError(
                "Not calibrated (no usable thresholds cache) and none of the "
                "dsa_calibration_roots exist on this machine. Set them via "
                "--set dsa_calibration_roots=/path/a,/path/b "
                "(dirs holding known-DSA sequences with frames/ subdirs)."
            )
        thresholds = _calibrate(roots, cfg.workers, step_dir)
        _save_thresholds_cache(cache_path, thresholds, configured_roots)
        thresholds_source = f"calibrated (saved to {cache_path})"
        print(f"[06] Thresholds cached -> {cache_path}")

    # ── Phase 2: classify every extracted sequence ───────────────────────
    seq_dirs = find_sequence_dirs(output_root)
    if not seq_dirs:
        print(f"[06] No sequences found under {output_root} — nothing to split.")
        return {"thresholds_source": thresholds_source, "sequences": 0,
                "potential_dsas": 0, "potential_non_dsas": 0}

    results: List[Dict] = []
    with tqdm(total=len(seq_dirs), unit="seq", desc="[06] Detecting") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = {
                ex.submit(_analyse_sequence, (str(d), thresholds)): str(d)
                for d in seq_dirs
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({
                        "sequence_dir": futures[fut], "total_frames": 0,
                        "candidate_frame": "", "candidate_std": "",
                        "candidate_mean": "", "candidate_bright": "",
                        "candidate_entropy": "",
                        "verdict": f"worker_error: {e}",
                        "split_dir": "", "copied": False,
                    })
                pbar.update(1)

    # ── Split copy: exhaustive — every sequence lands in exactly one dir ─
    dsa_root = split_root / POTENTIAL_DSA_DIRNAME
    non_dsa_root = split_root / POTENTIAL_NON_DSA_DIRNAME
    dsa_root.mkdir(parents=True, exist_ok=True)
    non_dsa_root.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(results), unit="seq", desc="[06] Copying") as pbar:
        for res in results:
            dest_root = (dsa_root if res["verdict"] == "potential_dsa"
                         else non_dsa_root)
            ok, dest = _copy_sequence(Path(res["sequence_dir"]), dest_root)
            res["copied"] = ok
            res["split_dir"] = dest
            pbar.update(1)

    results.sort(key=lambda r: r["sequence_dir"])
    write_csv(step_dir / "dsa_split_report.csv",
              ["sequence_dir", "total_frames", "candidate_frame",
               "candidate_std", "candidate_mean", "candidate_bright",
               "candidate_entropy", "verdict", "split_dir", "copied"],
              results)

    n_dsa = sum(1 for r in results if r["verdict"] == "potential_dsa")
    verdict_breakdown: Dict[str, int] = {}

    def _nf(r) -> int:
        try:
            return int(r.get("total_frames") or 0)
        except (ValueError, TypeError):
            return 0

    total_frames = frames_dsa = 0
    for r in results:
        v = r["verdict"]
        verdict_breakdown[v] = verdict_breakdown.get(v, 0) + 1
        nf = _nf(r)
        total_frames += nf
        if v == "potential_dsa":
            frames_dsa += nf
    summary = {
        "thresholds_source": thresholds_source,
        "sequences": len(results),
        "frames": total_frames,
        "potential_dsas": n_dsa,
        "potential_dsa_frames": frames_dsa,
        "potential_non_dsas": len(results) - n_dsa,
        "potential_non_dsa_frames": total_frames - frames_dsa,
        "verdict_breakdown": verdict_breakdown,
        "copy_failures": sum(1 for r in results if not r["copied"]),
        "dsa_dir": str(dsa_root),
        "non_dsa_dir": str(non_dsa_root),
    }
    print(f"[06] {summary}")
    return summary
