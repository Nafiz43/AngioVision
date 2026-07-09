# AngioVision Fine-Tuning — Unified Pipeline

One code path for the whole **ViT + text-encoder contrastive** family that used
to live in a dozen near-duplicate `custom_framework_*` / `siglip` scripts.
Every structural variant is now a **flag**, and every hyper-parameter is a CLI
argument, so ablations are reproducible and diffable instead of copy-pasted.

```
fine-tuning/
├── train.py                 # unified training entrypoint
├── validate.py              # unified binary-QA validation entrypoint
├── angio_ft/                # the actual implementation (importable package)
│   ├── constants.py         # dependency-free choice constants (arch/pool/temporal)
│   ├── common.py            # frame discovery, pooling, temporal PE, ViT utils
│   ├── data.py              # StudyDataset + collate + worker seeding
│   ├── losses.py            # clip_loss_chunked + siglip_loss
│   ├── models.py            # PooledCLIP (train forward + inference) + summaries
│   ├── engine.py            # train loop, optimizer, checkpoint I/O, post-pipeline
│   ├── qa_eval.py           # QA validation + scoring + multi-checkpoint compare
│   └── cli.py               # argparse definitions for both entrypoints
├── calculate_score.py       # (unchanged) metric computation, called by validate
├── plot_loss.py             # (unchanged) loss-curve plotting
└── legacy/                  # archived originals (nothing deleted)
```

---

## 1. The ablation matrix (structural variants = flags)

| Variant | Command |
| --- | --- |
| **CLIP, no temporal**   | `python3 train.py --arch clip    --temporal_mode none        …` |
| **CLIP, + temporal**    | `python3 train.py --arch clip    --temporal_mode sinusoidal  …` |
| **SigLIP, no temporal** | `python3 train.py --arch siglip  --temporal_mode none        …` |
| **SigLIP, + temporal**  | `python3 train.py --arch siglip  --temporal_mode sinusoidal  …` |
| **SigLIP2 (± temporal)**| `python3 train.py --arch siglip2 --temporal_mode {none,sinusoidal} …` |
| **X-CLIP (video tower)**| `python3 train.py --arch xclip   --temporal_mode none        …` |

- `--arch clip` uses the symmetric softmax contrastive loss (`clip_loss_chunked`).
- `--arch siglip` uses the sigmoid pairwise loss (`siglip_loss`) and adds a single
  learnable `siglip_bias` scalar (init `-10.0`). It is the **only** parameter
  difference between the two objectives; everything else is identical.
- `--arch siglip2` trains with the same sigmoid pairwise loss but defaults the
  **vision tower** to a SigLIP2 pretrained encoder
  (`google/siglip2-base-patch16-224`). SigLIP2's pretraining extras (captioning
  decoder, self-distillation) do not apply at contrastive fine-tune time — the
  transferable difference is the backbone. `--vit_name` may point at any
  SigLIP / SigLIP2 / CLIP / ViT / DINO checkpoint for **any** arch; the loader
  picks the right tower class and feature extraction (CLS token for ViT/DINO,
  attention-pooled head for SigLIP/CLIP) automatically.
- `--arch xclip` trains with the softmax contrastive loss but loads an
  **X-CLIP video vision tower** (default `microsoft/xclip-base-patch32`).
  Instead of encoding frames one-by-one and pooling, each sequence's frames
  are uniformly sampled — or padded by repetition — to the tower's fixed clip
  length (`config.num_frames`, 8 for the base checkpoint) and encoded in a
  **single forward pass with cross-frame attention** (message tokens), so
  temporal structure is modelled inside the tower. Frame pooling then
  collapses the per-frame features exactly as in the other archs.
  Sinusoidal frame PE is redundant here; use `--temporal_mode none`
  (`train_xclip.sh` defaults to it). X-CLIP checkpoints ship a video
  processor that returns 5-D tensors, so the loader substitutes a
  `CLIPImageProcessor` from the same repo — per-frame preprocessing is
  CLIP-style, so numerics match.
- Per-arch default vision towers (used when `--vit_name` is omitted):
  clip/siglip → `microsoft/rad-dino`, siglip2 → `google/siglip2-base-patch16-224`,
  xclip → `microsoft/xclip-base-patch32`.
- Temporal encoding is **deterministic** (sinusoidal, no learnable params), so a
  temporal-on and a temporal-off checkpoint remain mutually loadable.
- Frame-level temporal is **on by default** when `--temporal_mode sinusoidal`.
  Toggle with `--disable_frame_temporal` / `--enable_sequence_temporal`.

Each run is written to an **ablation-aware directory** so variants never
overwrite each other:

```
<out_dir>/<arch>_<tempON|tempOFF>_<epochs>_<batch>_<max_seq>_<max_frames>/
    ├── last.pt                     # rolling checkpoint
    ├── epoch_<n>.pt                # per-epoch checkpoints
    ├── <run_name>_loss.csv         # buffered loss log
    ├── train_cmd.sh                # exact command used
    └── train_config.txt            # full resolved config
```

Checkpoint payload is unchanged: `{step, epoch, model_state, opt_state}`.

---

## 2. Minimal training example

```bash
python3 train.py \
  --arch clip --temporal_mode sinusoidal \
  --meta_csv        /path/to/meta.csv \
  --reports_csv     /path/to/reports.csv \
  --base_frames_dir /path/to/frames_root \
  --vit_name  microsoft/rad-dino \
  --bert_name UCSD-VA-health/RadBERT-RoBERTa-4m \
  --epochs 5 --batch_size 2 --amp
```

Frames are expected at `BASE/<Anon Acc #>/<SOPInstanceUID>/frames/<images>`
(with sensible fallbacks; see `angio_ft/common.py:find_frame_files_for_sop`).

---

## 3. Hyper-parameter ablations (all CLI flags)

| Knob | Flags |
| --- | --- |
| **Pooling** | `--pooling {max,mean,logsumexp}`, `--frame_pooling`, `--sequence_pooling` |
| **Learning rates** | `--vision_backbone_lr`, `--text_lr`, `--head_lr`, `--weight_decay` |
| **Schedule / batch** | `--epochs`, `--batch_size`, `--grad_accum`, `--contrastive_accum`, `--warmup_steps`, `--grad_clip` |
| **Mixed precision** | `--amp`, `--vit_grad_ckpt` |
| **ViT unfreezing** | `--freeze_vision`, `--vit_trainable_blocks N`, `--vit_unfreeze_patch_embed` |
| **Text unfreezing** | `--freeze_text` |
| **Projection heads** | `--freeze_vision_proj`, `--freeze_text_proj` (trainable by default; `--freeze_vision --freeze_vision_proj` = fully locked image tower, LiT-style) |
| **Temporal scales** | `--frame_temporal_scale`, `--sequence_temporal_scale` |
| **Frame/seq limits** | `--max_frames_per_sequence`, `--max_sequences_per_study`, `--min_frames_per_sequence` |
| **Throughput** | `--num_workers`, `--prefetch_factor`, `--io_threads`, `--frame_chunk_size` |
| **CUDA memory** | `--vit_image_size`, `--empty_cache_each_step`, `--cache_clear_interval` |

`--vit_image_size` is auto-detected from the ViT config when omitted (e.g.
`microsoft/rad-dino` → 518), preventing the classic
`Input image size doesn't match model` error. Pass it explicitly to override.

Run `python3 train.py --help` for the complete, self-documenting surface.

### Reproducibility & run management

- `--seed N` (default 42) seeds torch/numpy/random, including DataLoader workers
  and report-variant sampling.
- Run dirs are named `{arch}_{tempTAG}_{epochs}_{bs}_{maxseq}_{maxframes}_h{cfghash}`
  where the 8-char hash covers **every result-affecting flag** (LRs, pooling,
  freeze flags, temporal scales, seed, ...) - runs that differ in any of them can
  no longer collide. `--run_name NAME` overrides the whole name.
- If the run dir already has checkpoints, training **stops with an error**
  instead of silently skipping. Choose `--resume` (continue from `last.pt`:
  model + optimizer + step, LR schedule fast-forwarded) or `--force` (retrain,
  purging stale `epoch_*.pt`/`last.pt` first).

### Contrastive batch size

Plain `--grad_accum` computes an independent loss per micro-batch, so it adds
**no extra in-batch negatives** - the contrastive signal stays starved at small
`--batch_size`. `--contrastive_accum` enables a GradCache-style path where
`batch_size * grad_accum` acts as **one** contrastive batch (all micro-batch
embeddings become mutual negatives) at single-micro-batch activation memory, via
a two-forward trick with pinned RNG state. Default off (legacy numerics).

> **Validation status:** unit-tested for gradient-equivalence to the naive
> full-batch loss (`tests/test_units.py::test_gradcache_matches_full_batch_gradient`),
> but not yet run end-to-end on GPU - smoke-test before large runs.

### Tests

CPU-only, no model downloads:

```bash
python3 -m pytest tests/test_units.py -q
```

Covers pooling, sinusoidal encoding, both losses, config-hash sensitivity,
hypothesis sets, checkpoint round-trip, and GradCache gradient-equivalence.

---

## 4. Per-epoch validation & the epoch-metrics CSV

Two independent per-epoch signals, both optional flags on `train.py`:

- **`--val_fraction F`** (e.g. `0.1`) holds out a seeded, study-level fraction
  of the training set and computes the **contrastive validation loss** on it
  after every epoch (same objective as training, no gradient steps). The split
  is deterministic in `--seed`, so resumed runs keep the same split.
- **`--epoch_qa_eval`** runs the full binary-QA validation
  (`--val_data_dir` sequence dirs against `--validation_csv` GT, empty =
  central `settings.py` `VALIDATION_CSV`) on the **live model** after every
  epoch and logs the complete metric set.

Every epoch appends one row to
`<run_dir>/<run_name>_epoch_metrics.csv` — written **incrementally** (append +
flush per epoch), so progress survives interruption, and `--resume` appends
rather than rewriting:

| column(s) | meaning |
| --- | --- |
| `epoch` | 1-based epoch number |
| `train_loss` | batch-size-weighted mean training loss over the epoch |
| `val_loss` | contrastive loss on the held-out split (blank if `--val_fraction 0`) |
| `ORIGINAL_*` | model metrics: `ACCURACY PRECISION RECALL F1 TP TN FP FN` |
| `FLIPPED_*` | all predictions inverted (sanity baseline) |
| `ALL_YES_*` / `ALL_NO_*` | constant-answer baselines |
| `RANDOM_*` | seeded coin-flip baseline (`--seed`) |

Per-epoch prediction/error CSVs land in `<run_dir>/epoch_eval/`.

---

## 5. One-command per-architecture scripts

`train_clip.sh`, `train_siglip.sh`, `train_siglip2.sh`, `train_xclip.sh`
(thin wrappers over `scripts/train_common.sh`) cover the whole pipeline.
Each wrapper's header also lists **exact commands for recommended variants**
(domain text towers, MedSigLIP, larger towers, LiT locking, pooling/temporal
ablations). **Training and post-hoc
evaluation are decoupled**: `train`/`resume` only train (per-epoch metrics are
still logged via `--epoch_qa_eval`); checkpoint validation, scoring and the
loss plot run via `validate` (or chain both with `full`).

```bash
./scripts/train_clip.sh                    # training only (default mode: train)
./scripts/train_siglip.sh resume           # continue SigLIP from last.pt (training only)
./scripts/train_siglip2.sh validate        # checkpoint validation + scoring + loss plot
./scripts/train_clip.sh eval <ckpt.pt>     # evaluate one specific checkpoint
./scripts/train_clip.sh full               # train, then validate + score + plot in one go
```

Every path/hyper-parameter is an env var (see the header of
`scripts/train_common.sh`), e.g.:

```bash
EPOCHS=10 BATCH_SIZE=4 VAL_FRACTION=0.1 ./scripts/train_siglip.sh
META_CSV=/my/meta.csv REPORTS_CSV=/my/reports.csv BASE_FRAMES_DIR=/my/frames ./scripts/train_clip.sh
VIT_NAME=google/siglip2-large-patch16-256 ./scripts/train_siglip2.sh
```

Output locations for a run named `$RUN_NAME` (default
`{arch}_{temporal}_e{epochs}_bs{batch}`):

```
$OUT_DIR/$RUN_NAME/epoch_<n>.pt                      per-epoch checkpoints
$OUT_DIR/$RUN_NAME/last.pt                           rolling checkpoint (resume point)
$OUT_DIR/$RUN_NAME/${RUN_NAME}_loss.csv              step-level loss log
$OUT_DIR/$RUN_NAME/${RUN_NAME}_epoch_metrics.csv     epoch-wise metrics (see §4)
$OUT_DIR/$RUN_NAME/epoch_eval/                       per-epoch predictions/errors
$OUTPUT_DIR/preds_${RUN_NAME}.csv                    best-checkpoint predictions
$OUTPUT_DIR/errors_${RUN_NAME}.csv                   best-checkpoint error log
```

Per-architecture quick reference (identical surface, different `ARCH`):

| | train (only) | resume | validate + score + plot | evaluate one checkpoint | train + evaluate |
| --- | --- | --- | --- | --- | --- |
| CLIP    | `./scripts/train_clip.sh`    | `./scripts/train_clip.sh resume`    | `./scripts/train_clip.sh validate`    | `./scripts/train_clip.sh eval <pt>` | `./scripts/train_clip.sh full` |
| SigLIP  | `./scripts/train_siglip.sh`  | `./scripts/train_siglip.sh resume`  | `./scripts/train_siglip.sh validate`  | `./scripts/train_siglip.sh eval <pt>` | `./scripts/train_siglip.sh full` |
| SigLIP2 | `./scripts/train_siglip2.sh` | `./scripts/train_siglip2.sh resume` | `./scripts/train_siglip2.sh validate` | `./scripts/train_siglip2.sh eval <pt>` | `./scripts/train_siglip2.sh full` |
| X-CLIP  | `./scripts/train_xclip.sh`   | `./scripts/train_xclip.sh resume`   | `./scripts/train_xclip.sh validate`   | `./scripts/train_xclip.sh eval <pt>` | `./scripts/train_xclip.sh full` |

---

## 6. Validation

`validate.py` evaluates a **single checkpoint file** or a **whole run directory**
(`epoch_*.pt` / `last.pt`), runs `calculate_score.py`, and copies the
best-by-`ORIGINAL_ACCURACY` predictions to `--out_csv`.

> **Auto-matching:** checkpoints produced by this pipeline **embed their training
> config** (`arch`, `temporal_mode`, temporal flags/scales, pooling, `vit_name`,
> `bert_name`, `embed_dim`, `vit_image_size`). `validate.py` reads it and rebuilds
> the exact architecture automatically, so you normally do **not** need to repeat
> the model flags. They are still accepted as a fallback for older checkpoints
> that predate the embedded config.

```bash
python3 validate.py \
  --arch clip --temporal_mode sinusoidal \
  --vit_name microsoft/rad-dino --bert_name UCSD-VA-health/RadBERT-RoBERTa-4m \
  --checkpoint <run_dir> \
  --out_csv    preds.csv \
  --error_csv  errs.csv \
  --device cuda --frame_chunk_size 32 --pooling max
```

`--sequence_repeat_factor` (default `1`) reproduces the legacy base-validator
behaviour when set to `16`.

### Dev/test selection protocol (recommended)

Selecting the best epoch and reporting the final number on the **same** GT CSV
inflates results. Split once, then select on dev and report on test:

```bash
python3 split_gt.py --gt_path gt.csv --dev_frac 0.5 --seed 42
python3 validate.py --checkpoint <run_dir> \
  --selection_csv  gt_dev.csv \
  --validation_csv gt_test.csv \
  --out_csv preds.csv
```

Predictions cover both splits; each checkpoint is scored on the dev split, and
the selected checkpoint is scored **once** on the held-out test split
(`[HELD-OUT TEST METRICS]` block).

### Prompt ensembling

`--prompt_ensemble` averages text embeddings over 3 paraphrases per yes/no
polarity (per question family) instead of a single sentence pair - a
zero-retraining accuracy lever. Default off (numerics identical to legacy).

Strict loading: validation now **fails loudly** if the checkpoint is missing
model keys (architecture mismatch) instead of silently evaluating a
partially-initialised model.

---

## 7. Automatic post-training pipeline

Add `--run_post_pipeline` to `train.py` to chain **validate → score → plot**
right after training. The exact structural flags (`--arch`, `--temporal_mode`,
scales, pooling, `--vit_image_size`) are forwarded to `validate.py`
automatically so the evaluated architecture matches training.

---

## 8. Optional report-generation head

The GPT-2 decoder + hold-out generation machinery is preserved verbatim in the
archived generation trainer. Expose it through the unified entrypoint with:

```bash
python3 train.py --enable_generation --arch clip \
  --meta_csv … --reports_csv … --base_frames_dir … \
  --decoder_model_name gpt2 --gen_loss_weight 1.0 --clip_loss_weight 1.0
```

`train.py` delegates to `legacy/custom_framework_train_with_generation.py`,
forwarding all generation-specific flags (`--decoder_*`, `--gen_*`,
`--generation_*`, `--holdout_*`). Generation is CLIP-only.

---

## 9. Legacy → unified mapping

| Archived script (`legacy/`) | Unified replacement |
| --- | --- |
| `custom_framework_train_temporal.py`       | `train.py --arch clip --temporal_mode sinusoidal` |
| `custom_framework_train_2.py`              | `train.py --arch clip --temporal_mode none` |
| `custom_framework_train_skip_frames.py`    | `train.py --arch clip --temporal_mode none --max_frames_per_sequence N` |
| `siglip.py`                                | `train.py --arch siglip …` |
| `custom_framework_train_with_generation.py`| `train.py --enable_generation …` (delegates) |
| `custom_framework_validate_temporal.py`    | `validate.py --temporal_mode sinusoidal …` |
| `custom_framework_validate.py`             | `validate.py --temporal_mode none --sequence_repeat_factor 16 …` |

**Bug fixed in passing:** the old base validator declared its text encoder as
`self.bert` (`BertModel`), while training saved it as `self.text_model`
(`AutoModel`). Training checkpoints therefore loaded with a *randomly
initialised* text tower during base validation. The unified pipeline uses the
**same `PooledCLIP`** for training and validation, so keys always match.

> Out-of-scope scripts kept in place (different lineage): `pipeline.py`,
> `train_gpt.py`, `d_custom_framework_*`, `finetune_clip_on_mosaics_*`,
> `infer_finetuned_clip_on_validation_mosaics.py`, `generate_dataset_table.py`,
> `custom_framework_sanity_check.py`.
