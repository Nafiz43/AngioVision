#!/usr/bin/env python3
"""
Pipeline configuration (text-data-preparation).

Tracked DEFAULTS live here. Runtime changes (interactive prompt or
--set key=value) are saved to config.local.json next to this file —
gitignored, so local paths never dirty the repo.

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
    # ── Input ─────────────────────────────────────────────────────────────
    # Raw reports CSV. Must contain the accession + report-text columns.
    input_csv: str = "/data/Deep_Angiography/Reports/Report_List_v01_01_merged_raw.csv"
    accession_column: str = "Anon Acc #"
    # Report-text column; empty string = auto-detect from common names.
    report_column: str = "radrpt"

    # ── Step 00: cleaning ─────────────────────────────────────────────────
    # PHI removal needs presidio-analyzer/-anonymizer + a spaCy model.
    # If unavailable, the step warns and skips ONLY the PHI sub-step.
    enable_phi_removal: bool = True
    # How many reports to show in the Original|Cleaned reviewer docx.
    num_docx_samples: int = 20
    # None-like 0 = clean all rows; set a positive number to limit.
    max_reports: int = 0

    # ── Step 01: augmentation (Ollama) ────────────────────────────────────
    model: str = "thewindmom/llama3-med42-8b"
    n_augmentations: int = 4
    max_retries: int = 3
    retry_sleep: float = 1.5
    ollama_timeout: int = 300

    # ── Parallelism (step 00 cleaning) ────────────────────────────────────
    workers: int = max(1, (os.cpu_count() or 8) - 1)

    # Fields the interactive editor offers to change, in display order.
    EDITABLE: tuple = field(
        default=(
            "input_csv", "accession_column", "report_column",
            "enable_phi_removal", "max_reports", "model",
            "n_augmentations", "workers",
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
