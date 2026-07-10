# AngioVision Image Matching (`utils/image-matching/`)

Embedding-based **DSA sequence matching** with K@N cross-validated evaluation.
Each labeled angiography sequence is embedded frame-by-frame with a vision
model, indexed in an in-memory ChromaDB collection, and query sequences are
matched by cosine nearest-neighbour search + majority vote over the retrieved
frames' source sequences. The predicted label is the winning sequence's
`angio_run` category.

Modularized from `utils/metadata_db/eval_kn_retrieval.py`, which it
**supersedes as the maintained code path** (the original stays in place for
provenance, alongside `how-to-run-kn-retrieval.txt` which documents the same
CLI). Flags, defaults, seeds and output formats are identical — existing
commands and results carry over unchanged.

## Quick start

```bash
pip install -r requirements.txt          # + torch/transformers for real models

python run_matching.py                                        # rad-dino, fl mode, 10-fold CV
python run_matching.py --model vit-b16 --n-folds 10
python run_matching.py --model openclip-b32 --frame-mode all --max-frames 20
python run_matching.py --temporal --n-folds 5                 # 1 mean+std vector per sequence
python run_matching.py --k-values 1 3 5 --workers 8
```

Reproducing the legacy `eval_k1*.py` 80/20 experiments:

```bash
python run_matching.py --model rad-dino --frame-mode best --split-mode holdout --scale-down 0.80 --k-values 1
python run_matching.py --model rad-dino --frame-mode fl   --split-mode holdout --scale-down 0.80 --k-values 1
python run_matching.py --model rad-dino --frame-mode all --max-frames 10 --split-mode holdout --scale-down 0.80 --k-values 1
```

Default data paths point at the lab server
(`/data/Deep_Angiography/...`) — override with
`--labeled-csv / --dicom-root / --sqlite-db` elsewhere.

## Pipeline stages

1. **Load labeled CSV** (`data_loading`) — group by `angio_run`; drop
   'other'/duplicate/pathless rows; rows missing the frame annotation the
   chosen `--frame-mode` needs are excluded and written to a sidecar CSV
   next to the input.
2. **DICOM index** (`data_loading`) — file-stem → path map from the
   `utils/metadata_db` SQLite `dicom_files` table (fast path) or a
   filesystem walk of `--dicom-root` (fallback).
3. **Splits** (`splits`) — stratified n-fold CV (default, 10 folds) or a
   single stratified holdout (`--split-mode holdout`, train fraction
   `--scale-down`).
4. **Embedding model** (`embedding`) — `rad-dino` (default) / `vit-b16` /
   `vit-l16` / any HF AutoModel ID via transformers, or
   `openclip-b32`/`openclip-l14` via open-clip-torch. L2-normalised CLS /
   image features.
5. **Precompute** (`vector_store`) — every sequence embedded exactly once,
   before the fold loop (threaded DICOM reading, batched forward passes);
   `--temporal` collapses each sequence to one mean+std-pooled 2×D vector.
6. **Fold loop** (`vector_store` + `evaluation`) — per fold: fresh in-memory
   ChromaDB ← train embeddings; each test sequence queries K neighbours per
   frame; majority vote over retrieved source sequences → prediction; swept
   over all `--k-values`.
7. **Aggregate + report** (`evaluation` + `reporting`) — pooled micro/macro
   accuracy and per-fold mean±std; outputs auto-named
   `kn_results_{model}_{mode}_{split}[_temporal].{md,png,docx,csv}`:
   console table, bar chart + K-sweep line plot, Markdown tables, a Word
   doc of K=1 hit/miss retrieval examples with frame thumbnails (last fold),
   and a machine-readable **CSV** (long format: one row per category × K
   plus MICRO/MACRO rows per K, with the run parameters — model, frame mode,
   temporal, folds, seed — in every row so files from different runs
   concatenate cleanly for pandas/R analysis). Override paths with
   `--out-md / --out-plot / --out-docx / --out-csv`.

## Layout

```
image-matching/
├── run_matching.py        # entry point (thin wrapper over imatch.cli.main)
├── requirements.txt
└── imatch/
    ├── config.py          # constants, model aliases, label codes, output naming
    ├── data_loading.py    # labeled CSV, DICOM index, path resolution   (stdlib-only)
    ├── splits.py          # stratified CV / holdout splitting           (stdlib-only)
    ├── embedding.py       # model loader (lazy torch import) + temporal pooling
    ├── frames.py          # DICOM pixel extraction (best / fl / all modes)
    ├── vector_store.py    # ChromaDB, embedding precompute, fold ingestion
    ├── evaluation.py      # K@N eval, CV aggregation, K=1 example collection
    ├── reporting.py       # console / PNG / Markdown / docx outputs
    └── cli.py             # argparse + orchestration
```

Heavy dependencies are localized: torch/transformers/open_clip load lazily in
`embedding.py`, matplotlib/python-docx degrade to a logged warning in
`reporting.py`, and `data_loading`/`splits`/`evaluation` are stdlib-only —
so the data/split/aggregation logic is importable and testable anywhere.

## Key flags

| Flag | Values | Meaning |
|------|--------|---------|
| `--model` | `rad-dino` (default), `vit-b16`, `vit-l16`, `openclip-b32`, `openclip-l14`, any HF ID | Embedding model |
| `--frame-mode` | `best`, `fl` (default), `all` | Which DICOM frames per sequence |
| `--max-frames` | int, `0`=all | Cap frames when `--frame-mode all` (uniform sampling) |
| `--split-mode` | `cv` (default), `holdout` | Validation scheme |
| `--n-folds` / `--scale-down` | `10` / `0.80` | CV folds / holdout train fraction |
| `--k-values` | `1 3 5 7 9 11 13 15` | K values to sweep |
| `--temporal` | flag | Mean+std temporal pooling (one vector per sequence) |
| `--min-seqs` / `--seed` | `3` / `42` | Category floor / RNG seed |
| `--limit-cats` | int, `0`=all | Smoke-test on first N categories |
