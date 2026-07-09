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


@dataclass
class PipelineConfig:
    # ── Core paths ────────────────────────────────────────────────────────
    # Raw DICOM tree to ingest.
    input_root: str = "/data/Deep_Angiography/rawdata"
    # Where extracted sequences (frames/ + metadata.csv per SOP) are written.
    output_root: str = str(PIPELINE_DIR / "runs" / "extracted_sequences")

    # ── Step 01: extraction filter ────────────────────────────────────────
    # "strict"  = RadiationSetting GR + SeriesDescription DSA/CO 2
    #             + PositionerMotion STATIC + NumberOfFrames > min_frames
    # "relaxed" = same, without the SeriesDescription check
    mode: str = "strict"
    min_frames: int = 2
    skip_existing: bool = True

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

    # ── Parallelism ──────────────────────────────────────────────────────
    workers: int = max(1, (os.cpu_count() or 8) - 1)

    # Fields the interactive editor offers to change, in display order.
    EDITABLE: tuple = field(
        default=(
            "input_root", "output_root", "mode", "min_frames",
            "reports_csv", "reports_accession_column", "workers",
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
