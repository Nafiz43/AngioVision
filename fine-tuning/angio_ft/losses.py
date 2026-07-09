"""
angio_ft.losses
────────────────
The two contrastive objectives selectable via ``--arch``:

  • ``clip``   -> :func:`clip_loss_chunked`  (symmetric softmax cross-entropy)
  • ``siglip`` -> :func:`siglip_loss`        (sigmoid pairwise, Zhai et al. 2023)

Both build the B x B similarity matrix in row chunks to bound peak memory, and
are lifted verbatim from the original ``custom_framework_train_temporal.py`` /
``siglip.py`` implementations so numerics are identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_loss_chunked(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    chunk: int = 4,
) -> torch.Tensor:
    """Symmetric CLIP loss.

    Computes the full BxB similarity matrix once (chunked to save peak memory),
    then extracts both image->text and text->image logits from the same matrix
    instead of doing two separate matmuls.
    """
    B = image_embeds.size(0)
    device = image_embeds.device
    targets = torch.arange(B, device=device)

    # Build full similarity matrix row-by-row to avoid peak BxB allocation
    sim = torch.empty(B, B, device=device, dtype=image_embeds.dtype)
    for i in range(0, B, chunk):
        sim[i: i + chunk] = image_embeds[i: i + chunk] @ text_embeds.t()

    logits = logit_scale * sim                                       # (B, B)
    loss_i2t = F.cross_entropy(logits, targets)                      # rows  -> image->text
    loss_t2i = F.cross_entropy(logits.t(), targets)                  # cols  -> text->image
    return 0.5 * (loss_i2t + loss_t2i)


def siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    siglip_bias: torch.Tensor,
    chunk: int = 4,
) -> torch.Tensor:
    """SigLIP sigmoid pairwise loss (Zhai et al. 2023).

    Each of the B^2 (image_i, text_j) pairs is an independent binary
    classification:
      y_ij = +1  if i == j  (positive pair)
      y_ij = -1  if i != j  (negative pair)

    Loss = -mean_{i,j}  log sigma( y_ij * (t * <img_i, txt_j> + b) )

    where t = logit_scale (learnable temperature),
          b = siglip_bias  (learnable per-pair offset, init=-10.0).
    """
    B = image_embeds.size(0)
    device = image_embeds.device
    dtype = image_embeds.dtype

    # Build BxB cosine similarity matrix (chunked to bound peak memory)
    sim = torch.empty(B, B, device=device, dtype=dtype)
    for i in range(0, B, chunk):
        sim[i: i + chunk] = image_embeds[i: i + chunk] @ text_embeds.t()

    # Scaled logits with learnable bias
    logits = logit_scale * sim + siglip_bias  # (B, B)

    # Binary labels: +1 on diagonal (positives), -1 off-diagonal (negatives)
    labels = 2.0 * torch.eye(B, device=device, dtype=dtype) - 1.0

    # Mean sigmoid cross-entropy over all B^2 pairs
    return -F.logsigmoid(labels * logits).mean()
