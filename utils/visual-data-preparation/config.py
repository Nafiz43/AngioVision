#!/usr/bin/env python3
"""
Pipeline configuration.

This file holds the tracked DEFAULTS. Anything the user changes at runtime
(via the interactive prompt in run_pipeline.py, or --set key=value) is saved
to config.local.json next to this file — that file is gitignored, so local
paths never dirty the repo.

Precedence: config.local.json > defaults below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = PIPELINE_DIR / "config.local.json"

# Directory names step 06 splits sequences into (mirror s06_dsa_split).
POTENTIAL_DSA_DIRNAME = "00_potential_dsas"
POTENTIAL_NON_DSA_DIRNAME = "01_potential_non_dsas"


def dequote(raw: str) -> str:
    """Strip one pair of surrounding quotes, e.g. --set p="'/a/b'" -> /a/b.

    Without this a quoted path is not absolute (starts with a quote), so
    Path() silently joins it onto the pipeline dir.
    """
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()
    return raw


@dataclass
class PipelineConfig:
    # ── Data type ─────────────────────────────────────────────────────────
    # "training" (default) uses the core paths below. "validation" swaps
    # input/output/split roots to the val_* paths further down (see
    # apply_data_type) so the two never share output directories.
    data_type: str = "training"

    # ── Core paths (training) ─────────────────────────────────────────────
    # Raw DICOM tree to ingest.
    input_root: str = "/data/Deep_Angiography/rawdata"
    # Where extracted sequences (frames/ + metadata.csv per SOP) are written.
    output_root: str = str(PIPELINE_DIR / "runs" / "extracted_sequences")

    # ── Validation-mode paths (used when data_type == "validation") ───────
    # Independent of the training paths so a validation run never writes into
    # the training output tree. Override individually with --set val_*.
    val_input_root: str = (
        "/data/Deep_Angiography/VLM_Validation_Data_2026_06_17/"
        "Validation_DICOM_Data_2026_06_17_v01"
    )
    val_output_root: str = "/data/Deep_Angiography/Validation_VDP/DICOM_Sequence_Processed"
    val_dsa_split_root: str = "/data/Deep_Angiography/Validation_VDP/DSA_Split"

    # ── Step 01: extraction filter ────────────────────────────────────────
    # "strict"  = RadiationSetting GR + SeriesDescription DSA/CO 2
    #             + PositionerMotion STATIC + NumberOfFrames > min_frames
    # "relaxed" = same, without the SeriesDescription check
    # mode: str = "strict"
    mode: str = "relaxed"
    min_frames: int = 2
    skip_existing: bool = True

    # DICOM pixel-data decoding backend for step 01 (see vdp.common.save_frames).
    # "pydicom" = whatever non-GDCM plugin pydicom ships with (pylibjpeg/pillow/
    # native) — today's default. "gdcm" forces the GDCM plugin. Per-invocation
    # like data_type: never persisted to config.local.json.
    dicom_backend: str = "pydicom"

    # ── Step 00 post-QA: frame-count outlier review ──────────────────────
    # Sequences with frame counts outside [low, high] get their mosaic +
    # metadata copied to 00_consistency_check/outliers/ for review.
    outlier_low_thresh: int = 10
    outlier_high_thresh: int = 60

    # ── Step 03: mosaics ─────────────────────────────────────────────────
    mosaic_max_frames: int = 500
    mosaic_tile_size: int = 512
    mosaic_max_cols: int = 6
    overwrite_mosaic: bool = False

    # ── Step 05: accession cross-check ───────────────────────────────────
    # CSV listing the accessions you EXPECT to have data for. Leave empty
    # to skip step 05 gracefully.
    reports_csv: str = ""
    reports_accession_column: str = "Anon Acc #"

    # ── Step 06: frame-based DSA split ───────────────────────────────────
    # Auto-calibrated DSA mask-frame detector (ported from
    # utils/05_dsa_identification_based_on_frame_v2.py). Thresholds are
    # learned from known-positive DSA sequences, then every extracted
    # sequence under output_root is classified and copied into
    #   <dsa_split_root>/00_potential_dsas/<accession>/<sop>/
    #   <dsa_split_root>/01_potential_non_dsas/<accession>/<sop>/
    # Comma-separated list of roots holding known-DSA sequences
    # (each sequence = a dir owning a frames/ subdir). The default is the
    # consolidated examples dir populated by utils/copy_dsa_calibration_examples.py
    # (which gathered the 10 original roots listed below).
    # dsa_calibration_roots: str = ",".join([
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/0AwEV1kXtf",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/0BH55V6rIB",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/2C9rBTcczL",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/5NUyFXc5Ai",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/5o3Mxk1lx7",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/6kpsDZBHAH",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/1cZA9m5qti",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/P2ykm7rSF8",
    #     "/data/Deep_Angiography/DICOM_Sequence_Processed/1MPUcLN3XP/2.16.840.1.113883.3.16.245346042915223951797304877264329724942",
    #     "/data/Deep_Angiography/Deep_Angio_DB_v02/example_dsa_cases",
    # ])
    dsa_calibration_roots: str = (
        "/data/Deep_Angiography/frame_identification_algo_calibration_examples"
    )
    # Where the two split dirs are created. Must NOT live inside
    # output_root (would pollute the sequence tree on re-runs).
    dsa_split_root: str = str(PIPELINE_DIR / "runs" / "dsa_split")
    # Calibrated-threshold cache (stable across runs, like runs/ itself).
    # If this file exists, is complete, and was calibrated on the same
    # dsa_calibration_roots, step 06 SKIPS calibration and reuses it.
    # Delete the file or --set dsa_recalibrate=true to force recalibration.
    dsa_thresholds_json: str = str(PIPELINE_DIR / "runs" / "dsa_calibration.json")
    dsa_recalibrate: bool = False

    # ── Parallelism ──────────────────────────────────────────────────────
    workers: int = max(1, (os.cpu_count() or 8) - 1)

    def apply_data_type(self) -> None:
        """When data_type == 'validation', swap input/output/split roots to the
        val_* paths so validation outputs land in a separate directory tree.
        No-op for training. Calibration (dsa_calibration_roots / thresholds
        cache) is intentionally shared — the DSA detector uses one calibration
        for both data types."""
        if self.data_type == "validation":
            self.input_root = self.val_input_root
            self.output_root = self.val_output_root
            self.dsa_split_root = self.val_dsa_split_root

    def dsa_sequences_root(self) -> str:
        """Potential-DSA subset produced by step 06 — the working set that
        every post-split step (02/03/04) analyses instead of the full
        output_root."""
        return str(Path(self.dsa_split_root) / POTENTIAL_DSA_DIRNAME)

    # Fields the interactive editor offers to change, in display order.
    EDITABLE: tuple = field(
        default=(
            "input_root", "output_root", "mode", "min_frames",
            "reports_csv", "reports_accession_column",
            "dsa_calibration_roots", "dsa_split_root", "workers",
        ),
        repr=False,
    )


def load_config() -> PipelineConfig:
    cfg = PipelineConfig()
    if LOCAL_CONFIG_PATH.exists():
        overrides = json.loads(LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
        valid = {f.name for f in fields(PipelineConfig)}
        for key, value in overrides.items():
            if key in valid:
                if isinstance(value, str):
                    value = dequote(value)  # heal any quote-poisoned saved paths
                setattr(cfg, key, value)
    return cfg


def save_local_overrides(cfg: PipelineConfig) -> None:
    """Persist only the values that differ from the tracked defaults."""
    defaults = asdict(PipelineConfig())
    current = asdict(cfg)
    diff = {
        k: v for k, v in current.items()
        if k != "EDITABLE" and v != defaults.get(k)
    }
    LOCAL_CONFIG_PATH.write_text(
        json.dumps(diff, indent=2) + "\n", encoding="utf-8"
    )
