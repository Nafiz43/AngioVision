#!/usr/bin/env python3
"""
benchmarking-models pipeline
============================

Benchmarks the fine-tuned AngioVision model against local vision-language
models on the validation mosaic QA task (system-level comparison: our pipeline
consumes frame sequences, baseline VLMs consume the per-sequence mosaic).

    01  VLM baselines       (each Ollama VLM answers every validation
                             question per mosaic; predictions + metrics,
                             resumable)
    02  CLIP zero-shot      (naive, NOT fine-tuned CLIP answers the same
                             questions via yes/no prompt similarity)
    03  Bedrock VLMs        (AWS Bedrock-hosted Claude models answer the same
                             questions via the Converse API)
    04  statistics          (McNemar + bootstrap: every baseline — plus
                             random / all-yes / all-no controls — vs every
                             fine-tuned predictions file)

Fine-tuned predictions are NOT produced here — generate them with
fine-tuning/validate.py; step 02 picks up *predictions*.csv from
cfg.ft_predictions_dir.

On start you're offered the chance to fix the config — changes are saved to
config.local.json (gitignored). Baseline predictions accumulate in
runs/baselines/ (stable, resumable); per-run statistics land under
runs/run_<timestamp>/.

Usage
-----
    python run_pipeline.py                      # interactive, all steps
    python run_pipeline.py --yes                # non-interactive, all steps
    python run_pipeline.py --only 04            # statistics only
    python run_pipeline.py --yes --set vlm_models=qwen3-vl:32b
    python run_pipeline.py --yes --only 03 --set bedrock_models=anthropic.claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import asdict
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from config import PipelineConfig, dequote, load_config, save_local_overrides  # noqa: E402
from bmk import (  # noqa: E402
    s01_vlm_baselines, s02_clip_zeroshot, s03_bedrock,
    s05_siglip_zeroshot, xclip_zeroshot, s06_per_question, s04_statistics,
)

STEPS = [
    ("01", "VLM baselines (Ollama)", s01_vlm_baselines.run),
    ("02", "CLIP zero-shot baseline (naive, not fine-tuned)", s02_clip_zeroshot.run),
    ("03", "Bedrock VLM baselines (AWS)", s03_bedrock.run),
    ("05", "SigLIP-family zero-shot baselines (SigLIP/SigLIP2/MedSigLIP)", s05_siglip_zeroshot.run),
    ("07", "X-CLIP zero-shot baseline (video tower, mosaic-as-clip)", xclip_zeroshot.run),
    ("04", "Statistics (McNemar + bootstrap)", s04_statistics.run),
    ("06", "Per-question comprehensive eval suite (F1 + collapsible HTML)", s06_per_question.run),
]
STEP_IDS = [s[0] for s in STEPS]


def _coerce(current, raw: str):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return dequote(raw)


def interactive_config_fix(cfg: PipelineConfig) -> PipelineConfig:
    print("\nCurrent configuration:")
    for name in cfg.EDITABLE:
        print(f"  {name:<20} = {getattr(cfg, name)!r}")

    answer = input("\nEdit config? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        return cfg

    print("(press Enter to keep the current value)")
    for name in cfg.EDITABLE:
        current = getattr(cfg, name)
        raw = input(f"  {name} [{current}]: ").strip()
        if raw:
            try:
                setattr(cfg, name, _coerce(current, raw))
            except ValueError:
                print(f"    invalid value for {name}, keeping {current!r}")

    save_local_overrides(cfg)
    print(f"Saved overrides to {PIPELINE_DIR / 'config.local.json'} (gitignored).\n")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="benchmarking-models pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(f"{s[0]} ({s[1]})" for s in STEPS),
    )
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Non-interactive: accept config as-is, run selected steps")
    parser.add_argument("--skip", nargs="+", default=[], choices=STEP_IDS,
                        metavar="STEP", help="Step ids to skip")
    parser.add_argument("--only", nargs="+", default=None, choices=STEP_IDS,
                        metavar="STEP", help="Run ONLY these step ids")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config values (saved to config.local.json)")
    parser.add_argument("--f1-average", choices=["macro", "weighted"], default=None,
                        dest="f1_average",
                        help="Per-question F1 averaging for step 06 (default from "
                             "config: macro). 'weighted' writes parallel "
                             "*_weighted.{csv,html} without touching the macro outputs.")
    args = parser.parse_args()

    cfg = load_config()
    if args.f1_average:
        cfg.f1_average = args.f1_average

    for pair in args.set:
        key, _, raw = pair.partition("=")
        if not hasattr(cfg, key) or key == "EDITABLE":
            parser.error(f"unknown config key: {key}")
        setattr(cfg, key, _coerce(getattr(cfg, key), raw))
    if args.set:
        save_local_overrides(cfg)

    if not args.yes:
        cfg = interactive_config_fix(cfg)

    if args.only:
        selected = set(args.only)
    else:
        selected = set(STEP_IDS) - set(args.skip)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PIPELINE_DIR / "runs" / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  Validation CSV : {cfg.validation_csv}")
    print(f"  Mosaics root   : {cfg.mosaics_root}")
    print(f"  VLM models     : {cfg.vlm_models}")
    print(f"  Baselines dir  : {cfg.baselines_dir}")
    print(f"  FT predictions : {cfg.ft_predictions_dir}")
    print(f"  Run dir        : {run_dir}")
    print(f"  Steps          : {', '.join(s for s in STEP_IDS if s in selected) or 'none'}")
    print(f"{'=' * 64}\n")

    manifest = {
        "started": run_ts,
        "config": {k: v for k, v in asdict(cfg).items() if k != "EDITABLE"},
        "steps": {},
    }

    exit_code = 0
    for step_id, title, run_fn in STEPS:
        if step_id not in selected:
            manifest["steps"][step_id] = {"status": "opted_out"}
            continue
        print(f"── Step {step_id}: {title} " + "─" * max(0, 44 - len(title)))
        try:
            summary = run_fn(cfg, run_dir)
            manifest["steps"][step_id] = {"status": "ok", "summary": summary}
        except Exception as e:
            manifest["steps"][step_id] = {
                "status": "failed", "error": f"{type(e).__name__}: {e}",
            }
            print(f"[{step_id}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            exit_code = 1
        print()

    (run_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{'=' * 64}")
    for step_id, _, _ in STEPS:
        info = manifest["steps"].get(step_id, {})
        print(f"  {step_id}  {info.get('status', '?'):<10} {info.get('error', '')}")
    print(f"  Manifest: {run_dir / 'pipeline_manifest.json'}")
    print(f"{'=' * 64}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
