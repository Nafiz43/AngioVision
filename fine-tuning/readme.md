# Fine-Tuning Pipeline (Minimal Prototype)

This folder contains a **small, runnable prototype** of the multimodal pipeline in the figure:

1. A study has multiple visual frame sequences + one textual report.
2. Each sequence goes into its own **trainable vision encoder**.
3. Per-sequence features are merged with a simple **fusion layer** (average).
4. The report goes into a **frozen text encoder**.
5. A CLIP-style **contrastive objective** aligns merged visual features with text features.

The implementation is intentionally lightweight and uses only Python stdlib.

## Files

- `pipeline.py`: End-to-end toy training loop + CLI.

## Run

From repo root:

```bash
python fine-tuning/pipeline.py --epochs 20
```

Example with custom parameters:

```bash
python fine-tuning/pipeline.py \
  --epochs 40 \
  --embedding-dim 32 \
  --num-sequences 3 \
  --sequence-length 4 \
  --lr 0.02
```

## What this does (and does not)

### Included

- Clear mapping to the figure blocks (encoders, features, fusion, frozen text branch, contrastive training).
- Tiny synthetic dataset to demonstrate training flow.
- Simple similarity printouts before/after training.

### Not included (by design)

- No DICOM loader.
- No real CNN/ViT/BERT/CLIP dependencies.
- No distributed training, batching optimizations, or checkpointing.

This is a starting scaffold you can later replace with real model components while keeping the same overall structure.
