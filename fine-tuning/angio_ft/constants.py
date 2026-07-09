"""
angio_ft.constants
────────────────────
Dependency-free choice constants shared across the package.

Kept free of heavy imports (no torch / transformers) so the CLI parsers - and
thus ``train.py --help`` / ``validate.py --help`` - work in any environment,
including machines without the deep-learning stack installed.
"""

# Contrastive objective (structural ablation)
#   clip    -> symmetric softmax contrastive loss
#   siglip  -> sigmoid pairwise loss (Zhai et al. 2023), adds learnable bias
#   siglip2 -> same sigmoid pairwise objective; intended to be paired with a
#              SigLIP2 pretrained vision tower (Tschannen et al. 2025). The
#              SigLIP2 pretraining extras (captioning decoder, self-distillation)
#              do not apply to contrastive fine-tuning, so at fine-tune time the
#              objective is the sigmoid loss - what changes is the backbone.
#   xclip   -> softmax contrastive loss with an X-CLIP video vision tower
#              (Ni et al. 2022): the sequence's frames are encoded JOINTLY with
#              cross-frame attention (message tokens) instead of one-by-one, so
#              temporality is modelled inside the tower. Frames are uniformly
#              sampled/padded to the tower's fixed clip length
#              (config.num_frames, e.g. 8). Frame-level sinusoidal PE is
#              redundant here - use --temporal_mode none with this arch.
ARCH_CHOICES = ("clip", "siglip", "siglip2", "xclip")

# Default vision tower per architecture, used when --vit_name is not given.
ARCH_DEFAULT_VIT = {
    "clip": "microsoft/rad-dino",
    "siglip": "microsoft/rad-dino",
    "siglip2": "google/siglip2-base-patch16-224",
    "xclip": "microsoft/xclip-base-patch32",
}


def arch_uses_sigmoid(arch: str) -> bool:
    """True for architectures trained with the SigLIP sigmoid pairwise loss."""
    return arch in ("siglip", "siglip2")

# Pooling modes (frame-level and sequence-level)
POOL_CHOICES = ("max", "mean", "logsumexp")

# Temporal encoding modes (structural ablation)
TEMPORAL_MODE_CHOICES = ("none", "sinusoidal")
