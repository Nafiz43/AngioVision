#!/usr/bin/env python3
"""
benchmarking-models pipeline configuration.

Tracked DEFAULTS live here. Runtime changes (interactive prompt in
run_pipeline.py, or --set key=value) are saved to config.local.json next to
this file — gitignored, so lab-server paths never dirty the repo.

Precedence: config.local.json > defaults below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG_PATH = PIPELINE_DIR / "config.local.json"


def dequote(raw: str) -> str:
    """Strip accidental wrapping quotes from user-entered values.

    A pasted path like '/data/foo' (quotes included) would otherwise be taken
    literally and create a directory named `'` — this happened once.
    """
    s = raw.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


@dataclass
class PipelineConfig:
    # ── Shared inputs ─────────────────────────────────────────────────────
    # Ground-truth validation CSV (SOPInstanceUID, Question, Answer[, AccessionNumber]).
    # Validation_VDP snapshot (has generated mosaics) — keep in sync with
    # configs/settings.py:VALIDATION_CSV.
    validation_csv: str = (
        "/data/Deep_Angiography/Validation_VDP/"
        "VLM_Validation_Data_2026_06_17_v01_filtered.csv"
    )
    # Validation root; build_sequence_index walks it recursively for sequence
    # dirs (mosaic.png + metadata.csv). Mosaics live under DSA_Split/<split>/
    # <accession>/<SOP>/.
    mosaics_root: str = "/data/Deep_Angiography/Validation_VDP"

    # ── Step 01: VLM baselines (Ollama) ──────────────────────────────────
    # Comma-separated Ollama model tags to benchmark.
    vlm_models: str = "qwen3-vl:8b,qwen3-vl:32b,llava:34b,gemma3:27b,llama3.2-vision:11b"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_timeout_s: int = 180
    # Skip (SOPInstanceUID, Question) pairs already present in a model's
    # predictions CSV — makes long GPU runs resumable.
    skip_existing: bool = True
    # Where per-model prediction/metric CSVs accumulate. Stable across runs
    # (like text-data-preparation's runs/report_data) so resume works.
    baselines_dir: str = str(PIPELINE_DIR / "runs" / "baselines")

    # ── Step 02: zero-shot CLIP baselines (naive, not fine-tuned) ────────
    # Comma-separated HF checkpoints. Default matches the starting checkpoint
    # of fine-tuning/finetune_clip_on_mosaics_*.py so naive-vs-fine-tuned
    # isolates the effect of fine-tuning.
    clip_models: str = "openai/clip-vit-base-patch32"
    # "" = auto (cuda if available), or "cuda"/"cpu"/"cuda:1".
    clip_device: str = ""

    # ── Step 05: zero-shot SigLIP-family baselines (naive, not fine-tuned) ─
    # Comma-separated HF checkpoints for the SigLIP tower family the
    # fine-tuning track uses: SigLIP, SigLIP2, MedSigLIP. medsiglip-448 is
    # gated — needs a HF token (~/.cache/huggingface/token). Shares clip_device.
    siglip_models: str = (
        "google/siglip-base-patch16-224,"
        "google/siglip2-base-patch16-224,"
        "google/medsiglip-448"
    )

    # ── Step 07: zero-shot X-CLIP baseline (video tower; mosaic as clip) ──
    # X-CLIP has no single-image path — the mosaic is replicated to its 8-frame
    # clip length. Forced but consistent with the mosaic system-level task.
    xclip_model: str = "microsoft/xclip-base-patch32"

    # ── Step 06: comprehensive per-question evaluation suite ─────────────
    # Fine-tuned frozen-probe per-question CSV (from the fine-tuning probe;
    # cols checkpoint,group,question,n,yes_rate,F1_BASELINE,F1_A1_pre,...).
    # The suite joins per-question zero-shot/VLM F1 + constant controls onto it.
    probe_perq_csv: str = (
        "/data/Deep_Angiography/AngioVision/fine-tuning/reports/"
        "perq_leaderboard_2026-07-13.csv"
    )
    # Weighted-F1 twin of probe_perq_csv (same checkpoints/questions/OOF preds,
    # F1 columns are weighted-F1 instead of macro). Used when f1_average=weighted.
    probe_perq_csv_weighted: str = (
        "/data/Deep_Angiography/AngioVision/fine-tuning/reports/"
        "perq_leaderboard_weighted.csv"
    )
    # Per-question F1 averaging for step 06: "macro" (default) or "weighted".
    # "weighted" weights each class's F1 by its support; writes a parallel set of
    # *_weighted.csv/html so the macro outputs are left untouched.
    f1_average: str = "macro"
    # Stable dir the suite mirrors its latest CSV+HTML into (easy to find/pull).
    eval_out_dir: str = str(PIPELINE_DIR / "runs" / "eval_suite")

    # ── Step 03: AWS Bedrock VLM baselines ───────────────────────────────
    # Comma-separated Bedrock model ids (Converse API). Auth via boto3's
    # default credential chain — no keys in this repo.
    bedrock_models: str = "anthropic.claude-sonnet-4-6"
    bedrock_region: str = "us-west-2"
    bedrock_max_tokens: int = 16

    # ── Step 04: statistics (McNemar + bootstrap) ────────────────────────
    # Directory of fine-tuned model predictions (*predictions*.csv), columns:
    # AccessionNumber, SOPInstanceUID, Question, Answer.
    ft_predictions_dir: str = "/data/Deep_Angiography/AngioVision/fine-tuning/output"
    alpha: float = 0.15
    qa_limit: int = 0  # 0 = all QA rows; >0 = head(n) for smoke tests
    n_bootstrap: int = 2000
    bootstrap_seed: int = 42
    random_baseline_seed: int = 12345

    # Fields the interactive editor offers to change, in display order.
    EDITABLE: tuple = field(
        default=(
            "validation_csv", "mosaics_root", "vlm_models", "ollama_url",
            "clip_models", "clip_device", "siglip_models", "xclip_model",
            "bedrock_models", "bedrock_region",
            "baselines_dir", "ft_predictions_dir", "probe_perq_csv",
            "probe_perq_csv_weighted", "f1_average", "eval_out_dir",
            "alpha", "qa_limit",
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
                setattr(cfg, key, dequote(value) if isinstance(value, str) else value)
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
