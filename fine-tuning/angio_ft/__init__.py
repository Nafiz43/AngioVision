"""
angio_ft
─────────
Unified AngioVision contrastive fine-tuning package.

One code path, driven by ``--arch`` and ``--temporal_mode``, replaces the
family of duplicated ``custom_framework_train_*`` / ``siglip`` / validate
scripts.  Structural ablations:

    CLIP   without temporal :  --arch clip   --temporal_mode none
    CLIP   with    temporal :  --arch clip   --temporal_mode sinusoidal
    SigLIP without temporal :  --arch siglip --temporal_mode none
    SigLIP with    temporal :  --arch siglip --temporal_mode sinusoidal

Everything else (pooling, LR groups, batch/epoch/grad-accum, ViT unfreezing,
frame/sequence limits, ...) are hyper-parameter ablations exposed as CLI flags.

Public modules:
    common   - shared helpers (frame discovery, pooling, temporal, ViT utils)
    data     - StudyDataset + collate + worker seeding
    models   - PooledCLIP (train forward + inference) + param summary
    losses   - clip_loss_chunked + siglip_loss
    engine   - training loop, optimizer, checkpoint I/O, post-training pipeline
    qa_eval  - binary-QA validation + scoring
    cli      - argument parsers for the train.py / validate.py entrypoints
"""

# NOTE: submodules are intentionally NOT imported here. common/data/losses/
# models/engine/qa_eval pull in torch + transformers; importing them eagerly
# would make even `train.py --help` require the full DL stack. Import the
# submodule you need explicitly, e.g. `from angio_ft.engine import train`.

__all__ = ["common", "constants", "data", "losses", "models", "engine", "qa_eval", "cli"]
__version__ = "1.0.0"
