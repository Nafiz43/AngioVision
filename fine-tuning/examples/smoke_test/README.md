# Smoke-test example outputs (dummy data)

Example result files produced by running the **full** per-architecture
pipelines (`./scripts/train_clip.sh`, `./scripts/train_siglip.sh`, `./scripts/train_siglip2.sh`,
`./scripts/train_xclip.sh`) end-to-end on CPU with tiny offline models and synthetic
data shaped like the real corpus:

- **Training**: 8 studies, each with ONE report and MULTIPLE sequences
  (2–3 SOPInstanceUIDs per study, 3 frames per sequence) — sequences are
  pooled per study and contrasted against the study report.
- **Validation**: 6 sequence directories (`metadata.csv` + `frames/`), each
  with 1–3 per-sequence yes/no questions in the GT CSV.
- Config: `EPOCHS=2 BATCH_SIZE=2 VAL_FRACTION=0.25 DEVICE=cpu`, run 2026-07-08.
  The xclip run additionally shows a **resume**: it was extended from 2 to 3
  epochs via `RUN_NAME=xclip_none_e2_bs2 EPOCHS=3 ./scripts/train_xclip.sh resume`,
  which appended epoch 3 to the same metrics CSV. xclip's run name says
  `none` because that arch defaults to `--temporal_mode none` (temporality is
  modelled inside the X-CLIP tower by cross-frame attention).

Files per architecture (run names `clip_sinusoidal_e2_bs2`,
`siglip_sinusoidal_e2_bs2`, `siglip2_sinusoidal_e2_bs2`, `xclip_none_e2_bs2`):

| file | produced by | contents |
| --- | --- | --- |
| `<run>_epoch_metrics.csv` | `train.py --val_fraction --epoch_qa_eval` | one row per epoch: `epoch, train_loss, val_loss` + `ACCURACY/PRECISION/RECALL/F1/TP/TN/FP/FN` for `ORIGINAL, FLIPPED, ALL_YES, ALL_NO, RANDOM`; appended incrementally after every epoch |
| `preds_<run>.csv` | `validate.py` (best checkpoint) | `AccessionNumber, SOPInstanceUID, Question, Answer` per sequence-question |
| `<run>_loss.csv` | `train.py` | step-level loss log (`epoch, step, batch_size, loss, loss_ema`) |

`leaderboard.csv` ranks the architectures by best-epoch ORIGINAL accuracy with
the baseline accuracies (ALL_YES / ALL_NO / RANDOM / FLIPPED) alongside — the
format to reuse for real-run comparisons.

Metric values are meaningless (random tiny towers, 2 epochs, 12 GT rows) —
these files document the exact **format** each run produces. The baselines
behave as expected on the dummy GT: ALL_YES/ALL_NO = the class prior (0.5),
RANDOM = seeded coin flip, FLIPPED = 1 − ORIGINAL accuracy.
