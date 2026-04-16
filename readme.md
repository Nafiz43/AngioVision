# AngioVision

AngioVision is a research codebase for **end-to-end angiography understanding** across three connected tracks:

1. **Vision track**: process DICOM studies into sequence-level frames/mosaics and run VLM-based question answering.
2. **Text track**: extract sequence-specific findings from consolidated radiology reports and convert them into structured labels.
3. **Alignment track**: train and validate multimodal models that align sequence visuals with report text and evaluate binary clinical QA.

> ⚠️ This repository is currently organized as research pipelines and utilities, not as a packaged Python library. Most scripts are executed directly with `python <script>.py ...` and assume local dataset paths.

---

## Deep-dive overview

### Why this repo exists

The core challenge in this project is a **many-to-one supervision mismatch**:

- A single study can contain **multiple DICOM sequences/runs**.
- The corresponding radiology report is often **one consolidated narrative** for the entire study.
- Reliable clinical QA needs **sequence-level attribution** so that a model’s answer can be traced back to specific imaging evidence.

The repository tackles this with a staged workflow: preprocess DICOMs → build sequence-level visual artifacts → extract/normalize text supervision → run inference/fine-tuning → score/evaluate.

### Data assumptions encoded in the code

Across scripts, the expected data shape is broadly:

- Root with per-study/per-sequence folders that eventually include:
  - `frames/` (PNG frame exports)
  - `metadata.csv` (DICOM metadata key/value pairs)
  - optionally `mosaic.png` (sampled tiled sequence summary image)
- CSV files for:
  - report text (`radrpt`, accession IDs)
  - validation QA labels (`SOPInstanceUID`, `Question`, `Answer`)

Defaults in several scripts point to local paths under `/data/Deep_Angiography/...`; these should be overridden by CLI args for portability.

---

## Repository map (what each area does)

### `utils/`
General preprocessing and dataset utilities:

- DICOM to frame + metadata extraction (`01_process_dicom_sequences_multi_thread.py`)
- metadata consolidation and sanity checks
- augmentation and reporting helpers
- evaluation/statistics helpers (distribution checks, McNemar test, comparisons)

### `frame-processing/`
Sequence-level visual preparation and VLM inference:

- frame cleanup (`00_trim_black_borders_from_frames.py`)
- mosaic creation (`00_mosaic_creation.py`)
- QA extraction from frames or mosaics using multiple backends:
  - Ollama-hosted VLMs
  - CLIP / BiomedCLIP / MedCLIP variants
- report generation prototypes from mosaics (RadFM, Med-Flamingo, Ollama models)

### `text-report-processing/`
Report decomposition + sequence QA supervision extraction:

- split consolidated reports into inferred sequence chunks (`00_extract_sequences_from_reports.py`)
- extract labels from sequence chunks (`01_extract_labels_from_sequences.py`)
- GT-aware sequence extraction/QA pipelines (`02_extract_sequences_and_sequence_QA_using_GT*.py`)

### `fine-tuning/`
Training/validation experiments for multimodal alignment:

- prototype educational pipeline (`pipeline.py`)
- larger custom training pipelines (`custom_framework_train_*.py`)
- validation/inference scripts and scoring (`custom_framework_validate*.py`, `calculate_score.py`, `plot_loss.py`)

### `batch-processing/`
Bulk processing/evaluation scripts for validation sets and large CSV runs.

### `bedrock-inference/`
Inference path targeting AWS Bedrock-hosted models.

### `slr/`
Systematic literature review helpers (PDF preprocessing + staged screening/extraction).

### `md-files/`
Project notes, diagrams, roadmap artifacts, and weekly updates.

### `z-deprecated-scripts/`
Legacy scripts retained for reference. Prefer active folders above for new work.

---

## End-to-end workflow (recommended)

Below is a practical run order for a fresh experiment.

### 1) Build sequence-level image artifacts from DICOM

- Use `utils/01_process_dicom_sequences_multi_thread.py` to recursively find DICOMs, extract frame PNGs, and emit per-sequence `metadata.csv`.
- Optional: trim black borders in frames.

Example:

```bash
python utils/01_process_dicom_sequences_multi_thread.py \
  --input_dir /path/to/raw_dicom_root \
  --output_dir /path/to/DICOM_Sequence_Processed \
  --workers 16
```

### 2) Create mosaics per sequence

- Run `frame-processing/00_mosaic_creation.py` to sample frames and tile them into `mosaic.png` per sequence directory.

Example:

```bash
python frame-processing/00_mosaic_creation.py \
  --base_path /path/to/DICOM_Sequence_Processed \
  --max_frames 16 \
  --workers 8
```

### 3) Generate vision-side QA predictions

- For local Ollama VLMs, use `frame-processing/02_extract_labels_from_mosaics.py`.
- Alternate model backends exist in sibling scripts (`*_clip.py`, `*_biomedclip.py`, `*_medclip.py`, Bedrock variant).

Example:

```bash
python frame-processing/02_extract_labels_from_mosaics.py \
  --base_path /path/to/DICOM_Sequence_Processed \
  --validation_csv /path/to/gt.csv \
  --model llama3.2-vision:11b \
  --skip_existing
```

### 4) Generate text-side sequence supervision from reports

- Run `text-report-processing/00_extract_sequences_from_reports.py` to partition consolidated reports into sequence-level JSON outputs.
- Then run `01_extract_labels_from_sequences.py` (or GT-aware v2 scripts) to map sequence text to structured question/answer labels.

### 5) Train/validate multimodal models

- Start with `fine-tuning/pipeline.py` for a minimal reproducible conceptual run.
- Move to `custom_framework_train_2.py` + `custom_framework_validate.py` for full experiments.
- Score outputs with `fine-tuning/calculate_score.py` and visualize loss via `plot_loss.py`.

---

## Configuration and question sets

- `configs/questions.py` contains domain QA prompts used across multiple scripts.
- `configs/settings.py` centralizes default validation/data paths used by some validation tooling.

If you are running outside the original `/data/Deep_Angiography/...` environment, update paths via CLI flags (preferred) or local config edits.

---

## Environment setup

### 1) Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Model/runtime dependencies

Depending on which scripts you run, you may also need:

- local Ollama server and pulled vision models (`http://localhost:11434` defaults used in several scripts)
- PyTorch with CUDA for fine-tuning/inference-heavy workloads
- optional AWS credentials/region configuration for Bedrock scripts

---

## Practical notes before you run experiments

- **Path sensitivity:** many scripts embed absolute defaults tied to a specific filesystem layout.
- **Scale:** image extraction, mosaic creation, and VLM calls are expensive; use `--limit`/sampling options first.
- **Output hygiene:** most scripts write CSV/JSON incrementally; keep outputs in dedicated run folders.
- **Research state:** there are duplicate and “copy” scripts; treat filenames with `d_` or `copy` as experimental variants unless reviewed.

---

## Quick-start command shortlist

```bash
# DICOM -> frames + metadata
python utils/01_process_dicom_sequences_multi_thread.py --help

# frames -> sequence mosaics
python frame-processing/00_mosaic_creation.py --help

# mosaic -> VLM binary QA predictions
python frame-processing/02_extract_labels_from_mosaics.py --help

# report -> sequence chunks
python text-report-processing/00_extract_sequences_from_reports.py --help

# sequence chunks -> labels
python text-report-processing/01_extract_labels_from_sequences.py --help

# toy multimodal fine-tuning demo
python fine-tuning/pipeline.py --help
```

---

## Status

This README reflects the current repository structure and active script families as of April 2026. As this is an actively evolving research workspace, prefer checking each script’s `--help` output and header comments before launching long runs.
