# Legacy scripts (archived)

These are the original, pre-consolidation training/validation scripts. They are
**kept for reference and reproducibility — nothing here was deleted**. New work
should use the unified pipeline: [`../README_pipeline.md`](../README_pipeline.md)
(`../train.py`, `../validate.py`, package `../angio_ft/`).

| File | Superseded by |
| --- | --- |
| `custom_framework_train_temporal.py`        | `train.py --arch clip --temporal_mode sinusoidal` |
| `custom_framework_train_2.py`               | `train.py --arch clip --temporal_mode none` |
| `custom_framework_train_skip_frames.py`     | `train.py --arch clip --temporal_mode none --max_frames_per_sequence N` |
| `siglip.py`                                 | `train.py --arch siglip …` |
| `custom_framework_train_with_generation.py` | `train.py --enable_generation …` (delegates here) |
| `custom_framework_validate_temporal.py`     | `validate.py --temporal_mode sinusoidal …` |
| `custom_framework_validate.py`              | `validate.py --temporal_mode none --sequence_repeat_factor 16 …` |

## Note on `custom_framework_train_with_generation.py`

The unified `train.py --enable_generation` **delegates** to this file (it still
owns the full, tested GPT-2 decoder + hold-out generation implementation).
Keep it here; `train.py` resolves it from `legacy/` automatically.
