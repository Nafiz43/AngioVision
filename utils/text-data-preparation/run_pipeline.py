#!/usr/bin/env python3
"""
text-data-preparation pipeline
==============================

Runs the report text-preparation stages sequentially:

    00  clean reports            (PHI removal, abbreviation expansion,
                                  sentence casing → cleaned CSV + docx)
    01  augment reports          (Ollama rephrasing, resumable)
    02  comparison docx          (Original vs Augmented reviewer table)

On start you're offered the chance to fix the config (input CSV, columns,
model, ...) — changes are saved to config.local.json (gitignored). Reviewer
artifacts land under runs/run_<timestamp>/; the cleaned/augmented CSVs live
at the stable path runs/report_data/ so step 01's resume logic works across
runs. Everything under runs/ is gitignored.

By default ALL steps run. Opt out with --skip, run a subset with --only,
and use --yes for non-interactive runs.

Usage
-----
    python run_pipeline.py                     # interactive, all steps
    python run_pipeline.py --yes               # non-interactive, all steps
    python run_pipeline.py --skip 01           # skip augmentation
    python run_pipeline.py --only 00           # just cleaning
    python run_pipeline.py --yes --set input_csv=/path/to/reports.csv
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

from config import PipelineConfig, load_config, save_local_overrides  # noqa: E402
from tdp import s00_clean_reports, s01_augment_reports, s02_comparison_docx  # noqa: E402

STEPS = [
    ("00", "Clean reports", s00_clean_reports.run),
    ("01", "Augment reports (Ollama)", s01_augment_reports.run),
    ("02", "Original-vs-Augmented docx", s02_comparison_docx.run),
]
STEP_IDS = [s[0] for s in STEPS]

DATA_DIR = PIPELINE_DIR / "runs" / "report_data"


def _coerce(current, raw: str):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def interactive_config_fix(cfg: PipelineConfig) -> PipelineConfig:
    print("\nCurrent configuration:")
    for name in cfg.EDITABLE:
        print(f"  {name:<28} = {getattr(cfg, name)!r}")

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


def interactive_step_selection() -> set:
    print("Select steps to run (all run by default):")
    selected = set()
    for step_id, title, _ in STEPS:
        answer = input(f"  Run {step_id} — {title}? [Y/n] ").strip().lower()
        if answer not in ("n", "no"):
            selected.add(step_id)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="text-data-preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(f"{s[0]} ({s[1]})" for s in STEPS),
    )
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Non-interactive: accept config as-is, run selected steps")
    parser.add_argument("--skip", nargs="+", default=[], choices=STEP_IDS,
                        metavar="STEP", help="Step ids to skip (e.g. --skip 01)")
    parser.add_argument("--only", nargs="+", default=None, choices=STEP_IDS,
                        metavar="STEP", help="Run ONLY these step ids")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config values (saved to config.local.json)")
    args = parser.parse_args()

    cfg = load_config()

    for pair in args.set:
        key, _, raw = pair.partition("=")
        if not hasattr(cfg, key) or key == "EDITABLE":
            parser.error(f"unknown config key: {key}")
        setattr(cfg, key, _coerce(getattr(cfg, key), raw))
    if args.set:
        save_local_overrides(cfg)

    if not args.yes:
        cfg = interactive_config_fix(cfg)

    if not Path(cfg.input_csv).exists():
        print(f"[ERROR] input_csv does not exist: {cfg.input_csv}", file=sys.stderr)
        return 2

    if args.only:
        selected = set(args.only)
    else:
        selected = set(STEP_IDS) - set(args.skip)
        if not args.yes and not args.skip:
            selected = interactive_step_selection()

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PIPELINE_DIR / "runs" / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  Run dir  : {run_dir}")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Steps    : {', '.join(s for s in STEP_IDS if s in selected) or 'none'}")
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
            summary = run_fn(cfg, run_dir, DATA_DIR)
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
