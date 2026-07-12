"""
Step 01 — DICOM sequence extraction (frames + metadata.csv per SOP).

Filter mode (strict/relaxed) comes from config. Output layout:
    output_root/<AccessionNumber>/<SOPInstanceUID>/frames/*.png
    output_root/<AccessionNumber>/<SOPInstanceUID>/metadata.csv
Failed extractions are fully cleaned up (no partial dirs left behind).
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pydicom
from tqdm import tqdm

from vdp.common import (
    NA_VALUE, collect_leaf_dirs, dest_already_exists, extract_metadata_pairs,
    get_tag_str, is_probably_dicom, output_dir_for, passes_eligibility_filter,
    safe_str, sanitize_dirname, save_frames, write_csv,
)


def _process_leaf_dir(
    leaf_str: str, output_root_str: str, min_frames: int,
    skip_existing: bool, mode: str,
) -> Dict[str, List[Dict]]:
    output_root = Path(output_root_str)
    out: Dict[str, List[Dict]] = {
        "processed": [], "filtered": [], "skipped": [], "errors": [],
    }

    for f in Path(leaf_str).iterdir():
        if not f.is_file() or not is_probably_dicom(f):
            continue

        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception as e:
            out["errors"].append({"file": str(f), "stage": "header_read",
                                  "reason": f"{type(e).__name__}: {e}"})
            continue

        uid = get_tag_str(ds, "SOPInstanceUID") or None
        acc = get_tag_str(ds, "AccessionNumber")

        # Evaluate the eligibility filter FIRST — before the skip-existing
        # check — so every file gets a filter verdict for funnel accounting,
        # even on a re-run where the frames are already extracted.
        try:
            ok, reason = passes_eligibility_filter(ds, min_frames, mode)
        except Exception as fe:
            ok, reason = False, "filter_error"
            out["errors"].append({"file": str(f), "stage": "filter_eval",
                                  "reason": f"{type(fe).__name__}: {fe}"})

        if not ok:
            out["filtered"].append({
                "file": str(f), "reason": reason,
                "accession_number": acc, "sop_instance_uid": uid or "",
                "radiation": get_tag_str(ds, "RadiationSetting"),
                "series_desc": get_tag_str(ds, "SeriesDescription"),
                "positioner": get_tag_str(ds, "PositionerMotion"),
                "num_frames": get_tag_str(ds, "NumberOfFrames"),
            })
            del ds
            continue

        # Passed the filter — skip re-extraction if already on disk.
        if skip_existing and uid and dest_already_exists(output_root, acc, uid):
            out["skipped"].append({"file": str(f), "accession_number": acc,
                                   "sop_instance_uid": uid})
            del ds
            continue

        per_dicom_dir = output_dir_for(output_root, acc, uid or "NO_UID")
        frames_dir = per_dicom_dir / "frames"
        metadata_csv = per_dicom_dir / "metadata.csv"
        metadata_tmp = per_dicom_dir / "metadata.csv.tmp"

        try:
            ds_full = pydicom.dcmread(f, force=True)
            base_name = uid if uid else sanitize_dirname(f.stem)
            frame_count = save_frames(ds_full, frames_dir, base_name)
            del ds_full

            metadata_rows = extract_metadata_pairs(ds)
            metadata_rows.extend([
                {"Information": "source_file", "Value": safe_str(f.name)},
                {"Information": "source_path", "Value": safe_str(str(f))},
                {"Information": "frame_count", "Value": safe_str(frame_count)},
                {"Information": "accession_number", "Value": safe_str(acc)},
                {"Information": "sop_instance_uid", "Value": safe_str(uid)},
                {"Information": "filter_mode", "Value": safe_str(mode)},
            ])
            df = pd.DataFrame(metadata_rows)
            df["Information"] = df["Information"].fillna(NA_VALUE).map(safe_str)
            df["Value"] = df["Value"].fillna(NA_VALUE).map(safe_str)
            df.to_csv(metadata_tmp, index=False, encoding="utf-8")
            os.replace(metadata_tmp, metadata_csv)

            out["processed"].append({"file": str(f), "accession_number": acc,
                                     "sop_instance_uid": uid or "",
                                     "frame_count": frame_count})
        except Exception as e:
            out["errors"].append({"file": str(f), "stage": "processing",
                                  "reason": f"{type(e).__name__}: {e}"})
            shutil.rmtree(per_dicom_dir, ignore_errors=True)
        finally:
            del ds

    return out


def run(cfg, run_dir: Path) -> Dict:
    step_dir = run_dir / "01_process_sequences"
    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    leaf_dirs = collect_leaf_dirs(input_root)
    merged: Dict[str, List[Dict]] = {
        "processed": [], "filtered": [], "skipped": [], "errors": [],
    }

    with tqdm(total=len(leaf_dirs), unit="dir", desc=f"[01] Extracting ({cfg.mode})") as pbar:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = [
                ex.submit(_process_leaf_dir, str(d), str(output_root),
                          cfg.min_frames, cfg.skip_existing, cfg.mode)
                for d in leaf_dirs
            ]
            for fut in as_completed(futures):
                result = fut.result()
                for key in merged:
                    merged[key].extend(result[key])
                pbar.update(1)

    write_csv(step_dir / "processed.csv",
              ["file", "accession_number", "sop_instance_uid", "frame_count"],
              merged["processed"])
    write_csv(step_dir / "filtered.csv",
              ["file", "reason", "accession_number", "sop_instance_uid",
               "radiation", "series_desc", "positioner", "num_frames"],
              merged["filtered"])
    write_csv(step_dir / "skipped.csv",
              ["file", "accession_number", "sop_instance_uid"], merged["skipped"])
    write_csv(step_dir / "errors.csv", ["file", "stage", "reason"], merged["errors"])

    # Per-reason funnel breakdown (short-circuit filter → each file has exactly
    # one reason = the first gate it failed).
    def _nframes(s: str) -> int:
        try:
            return int(str(s).strip()) if str(s).strip() else 1
        except (ValueError, TypeError):
            return 1

    filtered_by_reason: Dict[str, int] = {}
    filtered_frames_by_reason: Dict[str, int] = {}
    for r in merged["filtered"]:
        reason = r["reason"]
        filtered_by_reason[reason] = filtered_by_reason.get(reason, 0) + 1
        filtered_frames_by_reason[reason] = (
            filtered_frames_by_reason.get(reason, 0) + _nframes(r.get("num_frames"))
        )

    extracted_frames = sum(int(r.get("frame_count", 0) or 0) for r in merged["processed"])

    summary = {
        "mode": cfg.mode,
        "examined": sum(len(merged[k]) for k in ("processed", "filtered", "skipped", "errors")),
        "processed": len(merged["processed"]),
        "extracted_frames": extracted_frames,
        "filtered": len(merged["filtered"]),
        "filtered_by_reason": filtered_by_reason,
        "filtered_frames_by_reason": filtered_frames_by_reason,
        "skipped_existing": len(merged["skipped"]),
        "errors": len(merged["errors"]),
        "output_root": str(output_root),
    }
    print(f"[01] {summary}")
    return summary
