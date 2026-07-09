"""
angio_ft.contrastive_accum
──────────────────────────
GradCache-style gradient accumulation for contrastive losses (Gao et al. 2021,
"Scaling Deep Contrastive Learning Batch Size under Memory Limited Setups").

Why this exists
---------------
Plain ``--grad_accum`` computes an *independent* contrastive loss per
micro-batch and sums the gradients. Each micro-batch therefore only ever sees
``batch_size`` in-batch negatives, so accumulation adds NO negatives — the
contrastive signal stays starved at small batch sizes.

This module instead makes ``batch_size × accum_steps`` behave like ONE big
contrastive batch (all micro-batch embeddings are negatives for each other),
while keeping peak activation memory at the level of a single micro-batch.

How it works (three passes, memory-bounded)
-------------------------------------------
1. **Represent** every micro-batch under ``no_grad`` and cache the (detached)
   image/text embeddings. Cheap in memory (embeddings only, no activations).
2. **Loss + rep-grad**: concatenate the cached embeddings, compute the full
   contrastive loss over the big batch, and backward *only as far as the cached
   embeddings* to obtain ``d loss / d embedding`` for every example.
   Learnable scalars used directly by the loss (``logit_scale``,
   ``siglip_bias``) receive their gradients here, since they are leaf params.
3. **Backprop into the backbone**: re-forward each micro-batch *with* grad and
   call ``torch.autograd.backward(embeds, grad_tensors=cached_rep_grads)`` so the
   cached rep-gradients flow into the model parameters, one micro-batch of
   activations at a time.

RNG state is saved after pass 1 and restored before each pass-3 re-forward so
dropout masks match between the two forwards (required for correctness).
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch


class _RngSnapshot:
    """Capture/restore CPU (and CUDA) RNG so two forwards are bit-identical."""

    def __init__(self, device: torch.device):
        self._device = device
        self._cpu = torch.get_rng_state()
        self._cuda = (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        )

    def restore(self) -> None:
        torch.set_rng_state(self._cpu)
        if self._cuda is not None:
            torch.cuda.set_rng_state(self._cuda, self._device)


def gradcache_contrastive_step(
    micro_batches: List[dict],
    forward_fn: Callable[[dict], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    amp_ctx_factory: Callable[[], "torch.autocast"],
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> torch.Tensor:
    """Run one GradCache optimizer-step worth of accumulation.

    Args:
        micro_batches: the ``accum_steps`` batches whose embeddings jointly form
            one big contrastive batch.
        forward_fn: ``batch -> (image_embeds, text_embeds, logit_scale)``. Must
            be deterministic given RNG state (it is — RNG is pinned below).
        loss_fn: ``(image_embeds, text_embeds, logit_scale) -> scalar loss`` over
            the FULL concatenated batch. It may also close over model params such
            as ``siglip_bias``; their grads are populated in pass 2.
        device, amp_ctx_factory, scaler: mixed-precision plumbing. ``scaler`` is
            used to scale the pass-2 loss so the cached rep-grads (and therefore
            the pass-3 param grads) are scaled consistently with the rest of
            training; ``scaler.unscale_``/``step`` are handled by the caller.

    Returns:
        The detached full-batch loss (for logging).

    The caller is responsible for ``opt.zero_grad`` before and
    ``scaler.step``/``opt.step`` + ``scheduler.step`` after.
    """
    if not micro_batches:
        raise ValueError("gradcache_contrastive_step got no micro-batches")

    use_amp = scaler is not None

    # ── Pass 1: represent (no grad), cache detached embeddings + RNG states ──
    img_reps: List[torch.Tensor] = []
    txt_reps: List[torch.Tensor] = []
    rng_states: List[_RngSnapshot] = []
    logit_scale_ref: torch.Tensor | None = None

    for mb in micro_batches:
        rng_states.append(_RngSnapshot(device))
        with torch.no_grad(), amp_ctx_factory():
            img, txt, logit_scale = forward_fn(mb)
        img_reps.append(img.detach())
        txt_reps.append(txt.detach())
        logit_scale_ref = logit_scale  # same leaf param every step

    sizes = [r.size(0) for r in img_reps]

    # ── Pass 2: loss over the full batch, grad only w.r.t. cached reps ───────
    img_all = torch.cat(img_reps, dim=0).detach().requires_grad_(True)
    txt_all = torch.cat(txt_reps, dim=0).detach().requires_grad_(True)

    with amp_ctx_factory():
        loss = loss_fn(img_all, txt_all, logit_scale_ref)

    scaled = scaler.scale(loss) if use_amp else loss
    scaled.backward()  # populates img_all.grad, txt_all.grad, and leaf-param grads

    grad_img = img_all.grad.detach()
    grad_txt = txt_all.grad.detach()

    # ── Pass 3: re-forward each micro-batch WITH grad, inject cached rep-grads ─
    offset = 0
    for mb, n, rng in zip(micro_batches, sizes, rng_states):
        rng.restore()  # reproduce pass-1 dropout masks exactly
        with amp_ctx_factory():
            img, txt, _ = forward_fn(mb)
        gi = grad_img[offset:offset + n]
        gt = grad_txt[offset:offset + n]
        torch.autograd.backward([img, txt], grad_tensors=[gi, gt])
        offset += n

    return loss.detach()
