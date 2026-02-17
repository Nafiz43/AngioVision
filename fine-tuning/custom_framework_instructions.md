# AngioVision Fine-Tuning (ViT + BERT + Contrastive Loss) --- Parameter Guide

This guide explains the CLI parameters for `train_framework_pooled.py`
(the pooled version: max/mean/logsumexp pooling across frames and
sequences).

The script trains a CLIP-style alignment model: - **Vision:** frames →
ViT → pool across frames (per sequence) → pool across sequences (per
study) → projected embedding - **Text:** report text → BERT → projected
embedding - **Loss:** contrastive (image-text matching within the batch)

------------------------------------------------------------------------

## Required Parameters

### --meta_csv

Path to metadata CSV containing: - `Anon Acc #` - `SOPInstanceUIDs`

Used to determine which sequences belong to each study.

------------------------------------------------------------------------

### --reports_csv

Path to report CSV containing: - `Anon Acc #` - Report text column
(default: `radrpt`)

Used to retrieve report text aligned with each study.

------------------------------------------------------------------------

### --base_frames_dir

Root directory containing frames in the layout:

BASE/\<Anon Acc
#\>/`<SOPInstanceUID>`{=html}/frames/`<image files>`{=html}

------------------------------------------------------------------------

## Column Mapping Parameters

### --report_text_col (default: radrpt)

Column name in reports CSV containing report text.

### --anon_col (default: Anon Acc #)

Column name used as study identifier key.

### --sop_col (default: SOPInstanceUIDs)

Column containing list of SOPInstanceUIDs per study.

Supports: - "uid1,uid2,uid3" - "\['uid1','uid2'\]" - "('uid1','uid2')"

------------------------------------------------------------------------

## Model Parameters

### --vit_name

Vision backbone model (default: google/vit-base-patch16-224-in21k)

### --bert_name

Text encoder model (default: bert-base-uncased)

### --embed_dim (default: 256)

Projection embedding dimension for contrastive learning.

------------------------------------------------------------------------

## Pooling Parameters

Pooling happens at two levels: 1. Frame-level (within sequence) 2.
Sequence-level (within study)

### --pooling (default: max)

Default pooling mode for both levels.

Options: - max - mean - logsumexp

### --frame_pooling

Override pooling for frame-level.

### --sequence_pooling

Override pooling for sequence-level.

Pooling behavior: - max → strongest activation dominates - mean → equal
contribution - logsumexp → smooth max (soft emphasis on strong signals)

------------------------------------------------------------------------

## Training Parameters

### --batch_size (default: 2)

Number of studies per batch.

### --epochs (default: 1)

Number of full dataset passes.

### --lr (default: 1e-4)

Learning rate.

### --weight_decay (default: 0.01)

AdamW weight decay.

### --grad_clip (default: 1.0)

Gradient clipping threshold.

------------------------------------------------------------------------

## Performance Parameters

### --num_workers (default: 4)

DataLoader worker processes.

### --frame_chunk_size (default: 16)

Number of frames processed at once through ViT.

### --cpu

Force CPU training.

------------------------------------------------------------------------

## Dataset Filtering

### --min_frames_per_sequence (default: 1)

Minimum frames required to include a sequence.

### --max_sequences_per_study

Limit number of SOPs per study.

### --keep_missing_reports

Keep rows even if report text missing.

------------------------------------------------------------------------

## Checkpointing

### --out_dir (default: ./checkpoints_pooled)

Directory to save checkpoints.

### --save_every (default: 0)

Save checkpoint every N steps.

------------------------------------------------------------------------

## Common Tuning Advice

If GPU utilization is low: - Increase batch_size - Increase
frame_chunk_size - Increase num_workers

If out-of-memory: - Reduce batch_size - Reduce frame_chunk_size - Limit
max_sequences_per_study

------------------------------------------------------------------------

## Pooling Strategy Guidance

-   Use max if important features appear in few frames.
-   Use mean if signal distributed across many frames.
-   Use logsumexp for balanced soft-max behavior.
