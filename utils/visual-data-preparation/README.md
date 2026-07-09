# visual-data-preparation

Modular, self-contained pipeline that turns a raw DICOM tree into a clean,
analyzed visual dataset. Consolidates six standalone `utils/` scripts into
one orchestrated run with shared code (`vdp/common.py`) and zero hardcoded
lab-server paths.

## Steps (run sequentially; each can be opted out)

| Step | Module | What it does |
|------|--------|--------------|
| 00 | `s00_consistency_check` | Read-only QA on the raw tree: unreadable files, missing AccessionNumber, duplicate SOPInstanceUID, accession→study mismatches |
| 01 | `s01_process_sequences` | Extract frames + `metadata.csv` per SOP (strict or relaxed filter via config) |
| 02 | `s02_stats_gen` | Frame-count CSV + full distribution report (histogram/boxplot/Q-Q/stats); also owns the instances-per-study distribution, triggered by step 04 once the consolidated CSV exists |
| 03 | `s03_mosaics` | `mosaic.png` per sequence (reuses valid existing ones) + sizes CSV |
| 04 | `s04_consolidate` | Hub CSV (one row per study, SOPs chronologically ordered) + unmatched SOPs; then calls step 02's instances analysis |
| 05 | `s05_accession_check` | Reports-list accessions missing from consolidated metadata (skips if no reports CSV configured) |
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
`runs/extracted_sequences/`) and the `mosaic.png` files placed inside each
sequence directory.

## Smoke test on dummy data

```bash
python tests/make_dummy_data.py
python run_pipeline.py --yes \
    --set input_root=sample_data/raw reports_csv=sample_data/reports.csv
```

The dummy set (8 tiny 16×16 DICOMs) exercises every step: strict-filter
passes and rejections, a relaxed-only sequence, a duplicate SOPInstanceUID,
a missing accession, and a reports-only accession for step 05 to flag.

## Dependencies

`pydicom numpy pandas pillow matplotlib scipy tqdm`
