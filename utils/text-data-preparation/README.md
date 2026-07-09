# text-data-preparation

Modular, self-contained pipeline that turns raw angiographic report text
into a cleaned, augmented training corpus with reviewer artifacts.
Consolidates three standalone `utils/` scripts (`21_cleaning_reports.py`,
`17_report_augmentor.py`, `18_text_report_comparison.py`) with shared code
(`tdp/common.py`) and no hardcoded lab-server paths.

## Steps (run sequentially; each can be opted out)

| Step | Module | What it does |
|------|--------|--------------|
| 00 | `s00_clean_reports` | Unicode normalization, PHI removal (Presidio — skipped with a warning if not installed), attestation/header/HISTORY stripping, abbreviation + shorthand expansion, sentence casing → `cleaned_reports.csv` + Original\|Cleaned reviewer docx |
| 01 | `s01_augment_reports` | N conservative Ollama rephrasings per report (Type column: Original / Augmented 1..N). Incremental + resume-safe; failed generations fall back to the original text (logged) |
| 02 | `s02_comparison_docx` | 3-column reviewer docx: Acc ID \| Original \| Augmented, one row per variant |

## Usage

```bash
python run_pipeline.py                 # interactive: fix config, choose steps
python run_pipeline.py --yes           # non-interactive, all steps
python run_pipeline.py --skip 01       # skip the (slow) augmentation step
python run_pipeline.py --yes --set input_csv=/path/to/reports.csv model=llama3
```

## Config

Defaults live in `config.py` (tracked). Anything you change — interactively
or via `--set` — is saved to `config.local.json` (gitignored).

## Outputs

- `runs/report_data/` (stable, gitignored): `cleaned_reports.csv`,
  `augmented_reports.csv`, `augment.log`. Stable so step 01's resume logic
  can skip already-augmented accessions across runs.
- `runs/run_<timestamp>/` (per run, gitignored): reviewer docx files per
  step + `pipeline_manifest.json` with config and per-step summaries.

## Smoke test on dummy data

```bash
python tests/make_dummy_reports.py
python run_pipeline.py --yes \
    --set input_csv=sample_data/reports_raw.csv enable_phi_removal=false \
          max_retries=1 retry_sleep=0
```

(3 tiny synthetic reports; without Ollama installed, step 01 logs failures
and falls back to originals, which still exercises the full flow.)

## Dependencies

Core: `pandas tqdm python-docx`
Step 00 PHI removal (optional): `presidio-analyzer presidio-anonymizer` +
`python -m spacy download en_core_web_lg`
Step 01: [Ollama](https://ollama.com) with the configured model pulled
(default `thewindmom/llama3-med42-8b`)
