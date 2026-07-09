# visual-data-preparation

Modular, self-contained pipeline that turns a raw DICOM tree into a clean,
analyzed visual dataset. Consolidates six standalone `utils/` scripts into
one orchestrated run with shared code (`vdp/common.py`) and zero hardcoded
lab-server paths.

## Steps (run sequentially; each can be opted out)

| Step | Module | What it does |
|------|--------|--------------|
| 00 | `s00_consistency_check` | Read-only QA on the raw tree: unreadable files, missing AccessionNumber, duplicate SOPInstanceUID, accession→study mismatches |
| 01 | `s01_process_sequences` | Extract frames + `metadata.csv` per SOP (strict or relaxed filter via config; default **relaxed** — no SeriesDescription keyword check, DSA identification is deferred to step 06's frame-based detector) |
| 02 | `s02_stats_gen` | Frame-count CSV + full distribution report (histogram/boxplot/Q-Q/stats); also owns the instances-per-study distribution, triggered by step 04 once the consolidated CSV exists |
| 03 | `s03_mosaics` | `mosaic.png` per sequence (reuses valid existing ones) + sizes CSV |
| 04 | `s04_consolidate` | Hub CSV (one row per study, SOPs chronologically ordered) + unmatched SOPs; then calls step 02's instances analysis |
| 05 | `s05_accession_check` | Reports-list accessions missing from consolidated metadata (skips if no reports CSV configured) |
| 06 | `s06_dsa_split` | Frame-based DSA identification (port of `utils/05_dsa_identification_based_on_frame_v2.py`, phases 1+2): calibrates mask-frame thresholds from known-DSA sequences (`dsa_calibration_roots`), then copies every extracted sequence into `<dsa_split_root>/00_potential_dsas/` or `<dsa_split_root>/01_potential_non_dsas/` (exhaustive split, `<accession>/<sop>` layout preserved). Thresholds are cached in `dsa_thresholds_json` (default `runs/dsa_calibration.json`, stable across runs) — when a valid cache exists for the same calibration roots, calibration is skipped; delete the file or `--set dsa_recalibrate=true` to force it. The evaluation phases (1b/1c) were not ported — use the original script for benchmark numbers |
| 00-post | `s00_consistency_check.run_post` | Runs automatically after the steps above when step 00 is selected (its inputs come from steps 02/03/05): copies mosaics+metadata of frame-count outliers to `00_consistency_check/outliers/`, and re-scans the raw tree to split step 05's missing accessions into "no DICOM at all" vs "DICOM without pixel data" |

## Usage

```bash
python run_pipeline.py                 # interactive: fix config, choose steps
python run_pipeline.py --yes           # non-interactive, all steps
python run_pipeline.py --skip 03 05    # opt out of specific steps
python run_pipeline.py --only 00       # run a single step
python run_pipeline.py --yes --set input_root=/path/to/dicom reports_csv=/path/to/reports.csv
```

## Config

Defaults live in `config.py` (tracked). Anything you change — interactively
or via `--set` — is saved to `config.local.json` (gitignored), so local
paths never dirty the repo.

## Outputs

All pipeline metadata goes under `runs/run_<timestamp>/` in this directory
(gitignored), one subfolder per step, plus `pipeline_manifest.json`
recording the config and per-step summaries. The only outputs outside the
run dir are the extracted sequences (`output_root`, defaults to
`runs/extracted_sequences/`), the `mosaic.png` files placed inside each
sequence directory, and step 06's split copies (`dsa_split_root`, defaults
to `runs/dsa_split/` — deliberately outside `output_root`, and the step
refuses to run if you point it inside).

## Smoke test on dummy data

```bash
python tests/make_dummy_data.py
python run_pipeline.py --yes \
    --set input_root=sample_data/raw reports_csv=sample_data/reports.csv \
          output_root=/tmp/vdp_smoke/extracted_sequences \
          dsa_calibration_roots=/tmp/vdp_smoke/extracted_sequences \
          dsa_split_root=/tmp/vdp_smoke/dsa_split
```

The dummy set (8 tiny 16×16 DICOMs) exercises every step: strict-filter
passes and rejections, a relaxed-only sequence, a duplicate SOPInstanceUID,
a missing accession, and a reports-only accession for step 05 to flag.
Step 06's default `dsa_calibration_roots` are lab-server paths, so for a
local smoke test point them at the extracted output itself (in-sample
calibration — every noise sequence lands in `00_potential_dsas`, which is
fine for wiring checks; don't read anything into the verdicts).

## Dependencies

`pydicom numpy pandas pillow matplotlib scipy tqdm`
