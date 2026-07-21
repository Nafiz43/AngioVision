#!/usr/bin/env python3
"""
visual-data-preparation pipeline
================================

Runs the DICOM visual-data-preparation stages sequentially:

    00  consistency check        (read-only QA on the raw input tree)
    01  process sequences        (extract frames + metadata.csv per SOP)
    06  DSA split                (frame-based mask detection; copies every
                                  sequence into 00_potential_dsas/ or
                                  01_potential_non_dsas/ under dsa_split_root)
    02  frame stats              (frame-count distribution + plots)
    03  mosaics                  (mosaic.png per sequence + sizes CSV)
    04  consolidate metadata     (hub CSV + instances distribution)
    05  accession check          (reports list vs consolidated metadata)

Steps 02-05 run on the potential-DSA subset produced by 06, not the full
extracted tree — the split happens immediately after extraction so every
downstream stage only touches likely-DSA sequences.

On start you're offered the chance to fix the config (input dir, output
dir, ...) — changes are saved to config.local.json (gitignored). Every
report/log lands under runs/run_<timestamp>/ inside this directory, which
is also gitignored.

By default ALL steps run (end-to-end). Opt out of specific steps with
--skip, or run a subset with --only. --yes skips all interactive prompts.

Usage
-----
    python run_pipeline.py                     # interactive, all steps
    python run_pipeline.py --yes               # non-interactive, all steps
    python run_pipeline.py --skip 03 05        # everything except 03 and 05
    python run_pipeline.py --only 00 01        # just these two
    python run_pipeline.py --yes --set input_root=/some/dicom/tree
    python run_pipeline.py --funnel-only runs/training/run_<ts>   # re-print a run's story
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import sys
from dataclasses import asdict
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from config import PipelineConfig, dequote, load_config, save_local_overrides  # noqa: E402
from vdp import (  # noqa: E402
    funnel, s00_consistency_check, s01_process_sequences, s02_stats_gen,
    s03_mosaics, s04_consolidate, s05_accession_check, s06_dsa_split,
)

# Execution order (not id order): 06 runs right after 01 so the DSA split
# happens on the freshly extracted sequences, and 02-05 then analyse only the
# potential-DSA subset (see cfg.dsa_sequences_root()).
STEPS = [
    ("00", "DICOM consistency check", s00_consistency_check.run),
    ("01", "Process DICOM sequences", s01_process_sequences.run),
    ("06", "DSA split (frame-based mask detection)", s06_dsa_split.run),
    ("02", "Statistics generation", s02_stats_gen.run),
    ("03", "Mosaics + sizes", s03_mosaics.run),
    ("04", "Consolidate metadata (+ instances stats)", s04_consolidate.run),
    ("05", "Accession cross-check", s05_accession_check.run),
    # 00-post (outlier mosaics + no-visual accessions) runs automatically
    # after the steps above whenever step 00 is selected — its inputs are
    # produced by steps 02/03/05, so it cannot run at position 00.
]
STEP_IDS = [s[0] for s in STEPS]


def _coerce(current, raw: str):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y")
    if isinstance(current, int):
        return int(raw)
    return dequote(raw)


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


def rebuild_funnel(run_dir: Path) -> int:
    """--funnel-only: re-render the funnel story for an existing run dir from
    its manifest. Complete AND interrupted runs work (the manifest is flushed
    after every step); steps that never ran simply render blank."""
    manifest_path = run_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] no pipeline_manifest.json in {run_dir} — this run "
              "predates incremental manifests or isn't a run dir.", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from types import SimpleNamespace
    cfg = SimpleNamespace(**manifest.get("config", {}))
    manifest["funnel"] = funnel.build(manifest.get("steps", {}), cfg, run_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    # Use 'spawn' for every step's ProcessPoolExecutor. The default 'fork' on
    # Linux can deadlock workers on a lock held in the parent at fork time
    # (e.g. the tqdm bar opened just before the pool), leaving all workers idle
    # in futex_wait while the parent blocks on the pool pipe. spawn starts clean
    # interpreters with no inherited locks.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. re-entry)

    parser = argparse.ArgumentParser(
        description="visual-data-preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(f"{s[0]} ({s[1]})" for s in STEPS),
    )
    parser.add_argument("--data_type", choices=["training", "validation"],
                        default="training",
                        help="Which dataset to process. 'validation' swaps input/output/"
                             "split roots to the val_* paths so it runs independently and "
                             "stores results in a separate directory tree from training.")
    parser.add_argument("--dicom-backend", choices=["pydicom", "gdcm"],
                        default="pydicom",
                        help="Pixel-data decoding backend for step 01 (default: pydicom). "
                             "'gdcm' requires python-gdcm installed.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Non-interactive: accept config as-is, run selected steps")
    parser.add_argument("--skip", nargs="+", default=[], choices=STEP_IDS,
                        metavar="STEP", help="Step ids to skip (e.g. --skip 03 05)")
    parser.add_argument("--only", nargs="+", default=None, choices=STEP_IDS,
                        metavar="STEP", help="Run ONLY these step ids")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config values (saved to config.local.json)")
    parser.add_argument("--funnel-only", metavar="RUN_DIR", default=None,
                        help="Rebuild + print the funnel story for an EXISTING run dir "
                             "from its pipeline_manifest.json (no steps are run). Works "
                             "on interrupted runs too, since the manifest is now written "
                             "incrementally after every step.")
    args = parser.parse_args()

    if args.funnel_only:
        return rebuild_funnel(Path(args.funnel_only))

    cfg = load_config()

    # CLI overrides
    for pair in args.set:
        key, _, raw = pair.partition("=")
        if not hasattr(cfg, key) or key == "EDITABLE":
            parser.error(f"unknown config key: {key}")
        setattr(cfg, key, _coerce(getattr(cfg, key), raw))
    if args.set:
        save_local_overrides(cfg)

    # Data type and DICOM backend are per-invocation (NOT persisted to
    # config.local.json). Applied after --set so val_* overrides are honoured,
    # then data_type swaps the roots.
    cfg.data_type = args.data_type
    cfg.apply_data_type()
    cfg.dicom_backend = args.dicom_backend

    # A non-default backend gets its own output/split dirs so it can never
    # collide with (or skip-existing against) another backend's run — unless
    # the user already pinned those paths explicitly via --set.
    explicitly_set = {pair.partition("=")[0] for pair in args.set}
    if cfg.dicom_backend != "pydicom":
        if "output_root" not in explicitly_set:
            cfg.output_root = f"{cfg.output_root.rstrip('/')}_{cfg.dicom_backend}"
        if "dsa_split_root" not in explicitly_set:
            cfg.dsa_split_root = f"{cfg.dsa_split_root.rstrip('/')}_{cfg.dicom_backend}"

    if not args.yes:
        cfg = interactive_config_fix(cfg)

    if not Path(cfg.input_root).exists():
        print(f"[ERROR] input_root does not exist: {cfg.input_root}", file=sys.stderr)
        return 2

    # Step selection: --only > --skip > interactive > all
    if args.only:
        selected = set(args.only)
    else:
        selected = set(STEP_IDS) - set(args.skip)
        if not args.yes and not args.skip:
            selected = interactive_step_selection()

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PIPELINE_DIR / "runs" / cfg.data_type / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  Data type   : {cfg.data_type.upper()}")
    print(f"  DICOM backend: {cfg.dicom_backend}")
    print(f"  Input root  : {cfg.input_root}")
    print(f"  Output root : {cfg.output_root}")
    print(f"  Split root  : {cfg.dsa_split_root}")
    print(f"  Run dir     : {run_dir}")
    print(f"  Steps       : {', '.join(s for s in STEP_IDS if s in selected) or 'none'}")
    print(f"{'=' * 64}\n")

    manifest = {
        "started": run_ts,
        "config": {k: v for k, v in asdict(cfg).items() if k != "EDITABLE"},
        "steps": {},
    }

    def flush_manifest() -> None:
        # Written after EVERY step so an interrupted/killed run still leaves a
        # usable manifest behind (--funnel-only can then tell a partial story).
        (run_dir / "pipeline_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    exit_code = 0
    for step_id, title, run_fn in STEPS:
        if step_id not in selected:
            manifest["steps"][step_id] = {"status": "opted_out"}
            flush_manifest()
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
        flush_manifest()
        print()

    # Post-QA phase of step 00 — depends on outputs of steps 02/03/05
    if "00" in selected:
        print("── Step 00-post: Outlier + no-visual QA " + "─" * 24)
        try:
            summary = s00_consistency_check.run_post(cfg, run_dir)
            manifest["steps"]["00-post"] = {"status": "ok", "summary": summary}
        except Exception as e:
            manifest["steps"]["00-post"] = {
                "status": "failed", "error": f"{type(e).__name__}: {e}",
            }
            print(f"[00-post] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            exit_code = 1
        flush_manifest()
        print()

    # Story-like filtering funnel across all stages (reads the step summaries).
    try:
        manifest["funnel"] = funnel.build(manifest["steps"], cfg, run_dir)
        flush_manifest()
    except Exception as e:
        print(f"[funnel] FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"{'=' * 64}")
    for step_id in [s[0] for s in STEPS] + ["00-post"]:
        info = manifest["steps"].get(step_id, {})
        print(f"  {step_id}  {info.get('status', '?'):<10}"
              f" {info.get('error', '')}")
    print(f"  Manifest: {run_dir / 'pipeline_manifest.json'}")
    print(f"{'=' * 64}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
