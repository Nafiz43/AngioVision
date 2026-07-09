# AngioVision `fine-tuning/`

Contrastive vision–language fine-tuning for coronary angiography: pairing
DICOM frame sequences with radiology reports, then answering per-sequence
yes/no questions zero-shot via image–text similarity.

This directory contains **three distinct, coexisting tracks** — don't assume
everything here is one pipeline:

| Track | What it is | Entry points | Docs |
|---|---|---|---|
| 1. **Unified temporal framework** (actively maintained) | `angio_ft/` package + thin CLI wrappers; all architecture variants are flags | `train.py`, `validate.py`, `train_*.sh` | this file + `README_pipeline.md` |
| 2. CLIP/SigLIP on mosaics | Separate lineage fine-tuning vanilla HF CLIP on pre-rendered `mosaic.png` grids | `finetune_clip_on_mosaics_*.py` | intentionally kept separate |
| 3. Pedagogical scaffold | Stdlib-only runnable prototype of the architecture diagram; not a real trainer | `pipeline.py` | `README_pipeline_scaffold.md` |

The rest of this README covers track 1. `README_pipeline.md` is the deep
reference (design rationale, ablation matrix, legacy mapping); this file is the
practical guide.

---

## 1. The data contract

**Training** (study-level contrastive): one study = ONE report + MULTIPLE
frame sequences.

- `--meta_csv` — one row per study: `Anon Acc #`, `SOPInstanceUIDs`
  (comma-separated or Python-list string of sequence UIDs).
- `--reports_csv` — `Anon Acc #`, `radrpt` (report text), `Type`
  (`Original` / `Augmented`; one variant is sampled per epoch).
- `--base_frames_dir` — frames on disk at
  `<base>/<AccessionNumber>/<SOPInstanceUID>/frames/*.png`
  (several fallback layouts are probed — see `angio_ft/common.py:find_frame_files_for_sop`).

**Validation** (sequence-level binary QA): one directory per sequence.

- `--val_data_dir` — each `<seq_dir>/` holds `metadata.csv`
  (`Information,Value` rows with `AccessionNumber`, `SOPInstanceUID`; the
  lowercase `accession_number` / `sop_instance_uid` keys written by
  `utils/visual-data-preparation` are accepted as fallbacks) and
  `frames/*.png`. Discovery is **recursive**: sequence dirs may sit at any
  depth under `--val_data_dir`, so both the flat legacy layout and the nested
  `<acc>/<sop>/` extraction layout work.
- Ground-truth CSV (`--validation_csv`, or central `settings.py`):
  `Accession, SOPInstanceUID, Question, Answer` with yes/no answers.
  Each question becomes a yes/no hypothesis pair; the sequence embedding's
  cosine similarity picks the answer.

## 2. Architectures

All four share the same skeleton — vision tower → frame features →
(optional temporal encoding) → pool frames → pool sequences per study →
projection MLP ↔ text tower (BERT-family) CLS → projection MLP — and are
selected with `--arch`:

| `--arch` | Loss | Default vision tower | What's different |
|---|---|---|---|
| `clip` | softmax contrastive | `microsoft/rad-dino` | baseline |
| `siglip` | sigmoid pairwise (learnable bias) | `microsoft/rad-dino` | loss ablation, batch-size robust |
| `siglip2` | sigmoid pairwise | `google/siglip2-base-patch16-224` | SigLIP2-pretrained tower (the pretraining extras don't apply at contrastive fine-tune time — the backbone is the change) |
| `xclip` | softmax contrastive | `microsoft/xclip-base-patch32` | **video tower**: a sequence's frames are uniformly sampled/padded to the tower's clip length (`config.num_frames`, 8 for base) and encoded **jointly with cross-frame attention** — temporality is modelled inside the tower, so use `--temporal_mode none` (the scripts default to it for this arch) |

`--vit_name` accepts any ViT / DINO / SigLIP / SigLIP2 / CLIP / X-CLIP
checkpoint for any arch; `--bert_name` accepts any AutoModel text encoder
(default `UCSD-VA-health/RadBERT-RoBERTa-4m`). Other structural knobs:
`--temporal_mode {none,sinusoidal}` (parameter-free, checkpoints stay mutually
loadable), `--pooling {max,mean,logsumexp}` at frame and sequence level,
partial ViT unfreezing (`--vit_trainable_blocks`), LiT-style locking
(`--freeze_vision [--freeze_vision_proj]`), GradCache-style contrastive
accumulation (`--contrastive_accum`, experimental).

The same `PooledCLIP` class serves training **and** validation, so any
training checkpoint loads cleanly for evaluation (checkpoints embed their
config; `validate.py` applies it automatically).

## 3. Quick start — one command per architecture

```bash
./scripts/train_clip.sh        # CLIP    (softmax, rad-dino tower)
./scripts/train_siglip.sh      # SigLIP  (sigmoid, rad-dino tower)
./scripts/train_siglip2.sh     # SigLIP2 (sigmoid, siglip2 tower)
./scripts/train_xclip.sh       # X-CLIP  (softmax, video tower w/ cross-frame attention)
```

Every script supports five modes:

| command | does |
|---|---|
| `./scripts/train_<arch>.sh` | **train only** (per-epoch val loss + QA metrics CSV still produced) |
| `./scripts/train_<arch>.sh resume` | continue from `last.pt` in the run dir (metrics CSV appends) |
| `./scripts/train_<arch>.sh validate` | checkpoint validation + scoring + loss plot for the run dir |
| `./scripts/train_<arch>.sh eval <ckpt.pt>` | evaluate one specific checkpoint file |
| `./scripts/train_<arch>.sh full` | train, then validate + score + plot in one go |

Everything is env-var overridable (`scripts/train_common.sh` lists all knobs):

```bash
EPOCHS=10 BATCH_SIZE=4 ./scripts/train_clip.sh
META_CSV=/my/meta.csv REPORTS_CSV=/my/reports.csv BASE_FRAMES_DIR=/my/frames ./scripts/train_siglip.sh
VIT_NAME=google/medsiglip-448 RUN_NAME=medsiglip ./scripts/train_siglip2.sh
DEVICE=cpu VALIDATION_CSV=/my/gt.csv ./scripts/train_xclip.sh full
```

**Each wrapper's header contains exact commands for the variants most likely
to produce strong results** (domain text towers, MedSigLIP, larger towers,
LiT locking, pooling/temporal ablations) — open `scripts/train_<arch>.sh` and copy the
one you want.

## 4. Outputs (per run name `$RUN_NAME`, default `<arch>_<temporal>_e<E>_bs<B>`)

```
$OUT_DIR/$RUN_NAME/
├── epoch_<n>.pt                        # per-epoch checkpoints
├── last.pt                             # rolling checkpoint (resume point)
├── <RUN_NAME>_loss.csv                 # step-level training loss log
├── <RUN_NAME>_epoch_metrics.csv        # ★ one row per epoch (see below)
└── epoch_eval/                         # per-epoch prediction CSVs
$OUTPUT_DIR/
├── preds_<RUN_NAME>.csv                # best-checkpoint predictions
└── errors_<RUN_NAME>.csv               # skipped/errored sequences
```

The **epoch metrics CSV** (written incrementally after every epoch, so
progress survives interruption; resume appends) has 43 columns:

- `epoch`, `train_loss`, `val_loss` — validation loss comes from a held-out,
  seed-deterministic study-level split (`--val_fraction`, default 0.1);
- `ACCURACY / PRECISION / RECALL / F1 / TP / TN / FP / FN` for each of
  `ORIGINAL` (the model), `FLIPPED` (answers inverted), and the baselines
  `ALL_YES`, `ALL_NO`, `RANDOM`.

`examples/smoke_test/` holds real example outputs of every file above for all
four architectures (produced on tiny offline models + synthetic data shaped
like the real corpus), plus `leaderboard.csv` — the comparison format to reuse
for real runs.

## 5. Manual CLI (what the scripts wrap)

```bash
# train
python3 train.py --arch xclip \
    --meta_csv .../consolidated_metadata_gt.csv \
    --reports_csv .../cleaned_report_list.csv \
    --base_frames_dir .../DICOM_Sequence_Processed \
    --val_data_dir .../Validation_Data/.../DICOM_Sequence_Processed \
    --epochs 10 --batch_size 2 --val_fraction 0.1 --epoch_qa_eval \
    --run_name my_xclip_run

# resume
python3 train.py ... --run_name my_xclip_run --resume

# validate every checkpoint in a run dir, score, keep the best
python3 validate.py --checkpoint checkpoints/my_xclip_run \
    --data_dir .../DICOM_Sequence_Processed --out_csv preds.csv

# evaluate one checkpoint
python3 validate.py --checkpoint checkpoints/my_xclip_run/epoch_7.pt ...
```

`train.py --help` / `validate.py --help` list every flag. Special case:
`train.py --enable_generation` delegates to the archived GPT-2
report-generation trainer in `legacy/` (never ported into `angio_ft/`).

## 6. Package layout (`angio_ft/`)

| module | contents |
|---|---|
| `cli.py` | argparse builders for both entrypoints |
| `constants.py` | dependency-free enums: `ARCH_CHOICES`, per-arch default towers, pooling/temporal choices |
| `common.py` | frame discovery, pooling ops, sinusoidal PE, image-processor loading + native-resolution detection |
| `data.py` | `StudyDataset` (study → sequences → frame paths), collate, worker seeding |
| `losses.py` | `clip_loss_chunked` (softmax), `siglip_loss` (sigmoid) |
| `models.py` | `PooledCLIP` — one model class for train **and** validate; vision-tower loader (`cls` / `pooled` / `video` feature modes); X-CLIP clip sampler |
| `engine.py` | training loop, per-epoch validation + metrics CSV, checkpoint I/O, run-dir naming by config hash |
| `contrastive_accum.py` | GradCache-style contrastive accumulation (experimental) |
| `qa_eval.py` | binary QA validation: hypothesis pairs → cosine similarity → `calculate_score.py`; in-process per-epoch eval via `predict_and_score` |

Support scripts: `calculate_score.py` (metrics + baselines),
`plot_loss.py` (loss curves), `split_gt.py` (dev/test GT split — see
`README_pipeline.md` §6 selection protocol).

## 7. Testing

```bash
python3 -m pytest tests/test_units.py -q   # CPU-only, no downloads, <5 s
```

Covers pooling numerics, sinusoidal PE, both losses, config-hash sensitivity,
epoch-metrics writer resume-safety, val-split determinism, X-CLIP frame
sampling + end-to-end encode, GradCache gradient equivalence, checkpoint
round-trip. **Run before and after touching anything in `angio_ft/`.**

## 8. Where to look next

- `README_pipeline.md` — design rationale, full ablation matrix, legacy
  script mapping, checkpoint-selection protocol.
- `legacy/README.md` — 1:1 mapping from archived originals to unified flags.
- `examples/smoke_test/README.md` — what each example output file means.
- `experiments-to-run.md`, `log.md`, `commands.txt` — run notes (not code).
