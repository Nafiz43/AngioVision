"""
angio_ft.models
────────────────
The unified ``PooledCLIP`` model shared by training and validation.

Architecture (identical for both contrastive objectives):
    Vision : frames -> ViT -> CLS token -> (optional temporal) -> pool over frames
             -> pool over sequences (per study) -> vision_proj -> L2-norm
    Text   : report -> AutoModel (BERT/BioBERT/RadBERT) -> CLS -> text_proj -> L2-norm

Ablation switches (all via constructor args, wired from the CLI):
    arch            : "clip"  -> symmetric softmax contrastive loss
                      "siglip" -> sigmoid pairwise loss (adds one learnable
                                  scalar ``siglip_bias``; everything else is
                                  byte-for-byte the same set of parameters)
    temporal_mode   : "none" | "sinusoidal"   (deterministic; NO extra params,
                      so temporal on/off checkpoints stay mutually loadable)
    frame/sequence pooling, ViT partial-unfreezing, temporal scales, ...

Because the SAME class is used for training and validation, a checkpoint saved
during training always loads cleanly for evaluation (fixes the historical
``self.bert`` vs ``self.text_model`` key-mismatch in the old base validator).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, ViTModel

from .constants import ARCH_CHOICES, POOL_CHOICES, TEMPORAL_MODE_CHOICES, arch_uses_sigmoid
from .common import (
    build_sinusoidal_position_encoding,
    pool_stack,
    uniform_subsample,
    vision_model_type,
)


# ─────────────────────────────────────────────────────────────────────────────
# Vision-tower loading (ViT / DINO / SigLIP / SigLIP2 / CLIP checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

# model_type -> (loader class name, feature mode). "cls" takes the raw CLS
# token from last_hidden_state (legacy ViT behaviour, numerics-preserving);
# "pooled" uses pooler_output (SigLIP attention-pooling head / CLIP layernormed
# CLS) falling back to mean-over-patches when the checkpoint has no head.
_SIGLIP_MODEL_TYPES = {"siglip", "siglip_vision_model", "siglip2", "siglip2_vision_model"}
_CLIP_MODEL_TYPES = {"clip", "clip_vision_model"}
_XCLIP_MODEL_TYPES = {"xclip", "xclip_vision_model"}


def load_vision_tower(vit_name: str) -> Tuple[nn.Module, str]:
    """Load a vision encoder and report its feature-extraction mode.

    Plain ViT / DINO checkpoints keep the legacy ``ViTModel`` path (identical
    numerics for all existing runs). SigLIP / SigLIP2 / CLIP checkpoints load
    their native vision towers, which have no CLS token in ViT's sense and use
    a pooling head instead. X-CLIP checkpoints load the video vision tower
    ("video" mode): a sequence's frames are encoded jointly with cross-frame
    attention, and the batch dim must be a multiple of ``config.num_frames``.
    """
    mtype = vision_model_type(vit_name)
    if mtype in _SIGLIP_MODEL_TYPES:
        from transformers import SiglipVisionModel
        return SiglipVisionModel.from_pretrained(vit_name), "pooled"
    if mtype in _XCLIP_MODEL_TYPES:
        from transformers import XCLIPVisionModel
        return XCLIPVisionModel.from_pretrained(vit_name), "video"
    if mtype in _CLIP_MODEL_TYPES:
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained(vit_name), "pooled"
    return ViTModel.from_pretrained(vit_name), "cls"


def sample_video_frame_indices(n_available: int, clip_len: int) -> List[int]:
    """Uniform indices mapping ``n_available`` frames onto a fixed ``clip_len``.

    Handles both directions: subsampling when the sequence is longer than the
    clip, and padding-by-repetition when it is shorter (X-CLIP requires exactly
    ``num_frames`` frames per video). Deterministic; preserves temporal order.
    """
    if n_available <= 0 or clip_len <= 0:
        return []
    if n_available == 1:
        return [0] * clip_len
    return [round(i * (n_available - 1) / (clip_len - 1)) for i in range(clip_len)] \
        if clip_len > 1 else [0]


class PooledCLIP(nn.Module):
    def __init__(
        self,
        vit_name: str,
        text_model_name: str,
        embed_dim: int = 256,
        arch: str = "clip",
        freeze_vision: bool = False,
        freeze_text: bool = False,
        freeze_vision_proj: bool = False,
        freeze_text_proj: bool = False,
        frame_pooling: str = "max",
        sequence_pooling: str = "max",
        vit_trainable_blocks: int = 3,
        vit_unfreeze_patch_embed: bool = False,
        temporal_mode: str = "sinusoidal",
        temporal_on_frames: bool = True,
        temporal_on_sequences: bool = False,
        frame_temporal_scale: float = 1.0,
        sequence_temporal_scale: float = 1.0,
    ):
        super().__init__()
        if arch not in ARCH_CHOICES:
            raise ValueError(f"arch={arch!r} must be one of {ARCH_CHOICES}")
        for choice_name, val, choices in [
            ("frame_pooling", frame_pooling, POOL_CHOICES),
            ("sequence_pooling", sequence_pooling, POOL_CHOICES),
            ("temporal_mode", temporal_mode, TEMPORAL_MODE_CHOICES),
        ]:
            if val not in choices:
                raise ValueError(f"{choice_name}={val!r} must be one of {choices}")

        self.arch = arch
        self.vit, self.vision_feature_mode = load_vision_tower(vit_name)
        # X-CLIP video towers require exactly config.num_frames frames per
        # forward call; sequences are uniformly sampled/padded to this length.
        self.video_num_frames = (
            int(getattr(self.vit.config, "num_frames", 8))
            if self.vision_feature_mode == "video" else 0
        )
        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.vit_hidden = self.vit.config.hidden_size
        self.text_hidden = self.text_model.config.hidden_size

        self.frame_pooling = frame_pooling
        self.sequence_pooling = sequence_pooling
        self.temporal_mode = temporal_mode
        self.temporal_on_frames = bool(temporal_on_frames)
        self.temporal_on_sequences = bool(temporal_on_sequences)
        self.frame_temporal_scale = float(frame_temporal_scale)
        self.sequence_temporal_scale = float(sequence_temporal_scale)

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.vit_hidden),
            nn.GELU(),
            nn.Linear(self.vit_hidden, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden, self.text_hidden),
            nn.GELU(),
            nn.Linear(self.text_hidden, embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        # SigLIP/SigLIP2 add one learnable per-pair bias (init=-10.0 as in the
        # paper). CLIP checkpoints never carry this key; sigmoid-loss
        # checkpoints always do.
        if arch_uses_sigmoid(arch):
            self.siglip_bias = nn.Parameter(torch.tensor(-10.0))
        else:
            self.siglip_bias = None

        if freeze_vision:
            for p in self.vit.parameters():
                p.requires_grad = False
        else:
            self._configure_partial_vit_finetuning(vit_trainable_blocks, vit_unfreeze_patch_embed)

        if freeze_text:
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Projection heads are trainable by default; these flags freeze them
        # independently of the backbones (e.g. --freeze_vision +
        # --freeze_vision_proj = fully locked image tower, LiT-style).
        if freeze_vision_proj:
            for p in self.vision_proj.parameters():
                p.requires_grad = False
        if freeze_text_proj:
            for p in self.text_proj.parameters():
                p.requires_grad = False

    # ── partial ViT fine-tuning ───────────────────────────────────────────

    def _vision_encoder_layers(self):
        """Locate the transformer block list across vision-tower families and
        transformers versions (v4 nests blocks under .encoder; v5 flattens)."""
        candidates = [self.vit, getattr(self.vit, "vision_model", None)]
        for root in candidates:
            if root is None:
                continue
            enc = getattr(root, "encoder", None)
            for holder in (enc, root):
                if holder is None:
                    continue
                for attr in ("layer", "layers"):
                    layers = getattr(holder, attr, None)
                    if layers is not None and len(layers) > 0:
                        return layers
        raise RuntimeError(
            "Cannot locate vision encoder layers (tried .encoder.layer(s) and "
            ".layer(s) on the tower and its .vision_model)."
        )

    def _configure_partial_vit_finetuning(
        self, vit_trainable_blocks: int, vit_unfreeze_patch_embed: bool
    ) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False

        layers = self._vision_encoder_layers()
        n = max(0, min(int(vit_trainable_blocks), len(layers)))
        # NOTE: `layers[-n:]` breaks for n=0 (Python treats -0 as 0, so it would
        # select ALL layers instead of none) - slice from a non-negative start.
        for block in layers[len(layers) - n:]:
            for p in block.parameters():
                p.requires_grad = True

        # Final norm / pooling head: names differ per tower family, and the
        # module may sit on the model itself (transformers v5, flattened) or on
        # a nested .vision_model (v4 SigLIP/CLIP).
        roots = [self.vit, getattr(self.vit, "vision_model", None)]
        for root in roots:
            if root is None:
                continue
            for attr in ("layernorm", "pooler", "post_layernorm", "head"):
                obj = getattr(root, attr, None)
                if obj is not None:
                    for p in obj.parameters():
                        p.requires_grad = True

        if vit_unfreeze_patch_embed:
            for root in roots:
                emb = getattr(root, "embeddings", None) if root is not None else None
                if emb is not None:
                    for p in emb.parameters():
                        p.requires_grad = True
                    break

    # ── running pool (used when ViT is trainable and grad must flow) ──────

    @torch.no_grad()
    def _init_running(self, device: torch.device, d: int, mode: str):
        if mode == "max":
            return torch.full((d,), -1e9, device=device), None
        if mode == "mean":
            return torch.zeros(d, device=device), torch.tensor(0, device=device)
        if mode == "logsumexp":
            return torch.full((d,), float("-inf"), device=device), None
        raise ValueError(mode)

    def _update_running(self, running, aux, chunk: torch.Tensor, mode: str):
        if mode == "max":
            return torch.maximum(running, chunk.max(0).values), aux
        if mode == "mean":
            return running + chunk.sum(0), aux + chunk.size(0)
        if mode == "logsumexp":
            return torch.logaddexp(running, torch.logsumexp(chunk, 0)), aux
        raise ValueError(mode)

    def _finalize_running(self, running, aux, mode: str) -> torch.Tensor:
        if mode == "max":
            return running
        if mode == "mean":
            count = aux.item() if hasattr(aux, "item") else int(aux)
            return running / float(count) if count > 0 else running
        if mode == "logsumexp":
            return running
        raise ValueError(mode)

    # ── utilities ─────────────────────────────────────────────────────────

    def _vit_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.vit.parameters())

    def _text_is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.text_model.parameters())

    def _pool_tensor(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        return pool_stack(x, mode)

    def _frame_features(self, out) -> torch.Tensor:
        """Per-frame feature vector from a vision-tower forward output.

        "cls"    -> raw CLS token (legacy ViT/DINO behaviour, numerics-identical
                    to all existing checkpoints).
        "pooled" -> pooler_output (SigLIP attention-pool head / CLIP layernormed
                    CLS); falls back to mean-over-patches if the checkpoint was
                    exported without a pooling head.
        """
        if self.vision_feature_mode == "pooled":
            pooled = getattr(out, "pooler_output", None)
            if pooled is not None:
                return pooled
            return out.last_hidden_state.mean(dim=1)
        return out.last_hidden_state[:, 0, :]

    def _add_temporal_encoding(
        self, x: torch.Tensor, positions: torch.Tensor, scale: float
    ) -> torch.Tensor:
        if self.temporal_mode == "none" or scale == 0.0:
            return x
        pe = build_sinusoidal_position_encoding(positions, x.size(-1), x.device, x.dtype)
        return x + scale * pe

    # ── video-mode (X-CLIP) sequence encoding ─────────────────────────────

    def _encode_video_clip(
        self,
        loaded: List[Tuple[int, Image.Image]],
        processor,
        device: torch.device,
        vit_image_size: Optional[int],
    ) -> torch.Tensor:
        """Encode one sequence as a fixed-length video clip -> ``(T, D)`` feats.

        ``loaded`` is the sequence's readable frames as (original_index, image)
        in temporal order. Frames are uniformly sampled / padded-by-repetition
        to exactly ``self.video_num_frames`` and encoded in a SINGLE forward
        pass so X-CLIP's cross-frame attention sees the whole clip. Sinusoidal
        temporal PE (if enabled) uses the sampled frames' original indices,
        though it is redundant for this tower - prefer --temporal_mode none.
        """
        sel = sample_video_frame_indices(len(loaded), self.video_num_frames)
        clip = [loaded[j] for j in sel]
        imgs = [t[1] for t in clip]
        positions = [t[0] for t in clip]

        proc_kwargs: Dict[str, Any] = dict(images=imgs, return_tensors="pt")
        if vit_image_size is not None:
            proc_kwargs["size"] = {"height": vit_image_size, "width": vit_image_size}
        inputs = processor(**proc_kwargs)
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        vit_ctx = torch.no_grad() if self._vit_is_frozen() else torch.enable_grad()
        with vit_ctx:
            out = self.vit(pixel_values=pixel_values)  # batch = one clip of T frames
            feats = getattr(out, "pooler_output", None)
            if feats is None:
                feats = out.last_hidden_state[:, 0, :]
            if self.temporal_on_frames and self.temporal_mode != "none":
                pos_t = torch.tensor(positions, device=device, dtype=torch.long)
                feats = self._add_temporal_encoding(feats, pos_t, self.frame_temporal_scale)
        return feats  # (video_num_frames, D)

    # ── frame encoding (parallel image loading) ──────────────────────────

    @staticmethod
    def _load_image(path: Path) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def encode_framepaths_pooled(
        self,
        frame_paths: List[Path],
        processor,
        device: torch.device,
        chunk_size: int = 16,
        pooling: str = "max",
        vit_image_size: Optional[int] = None,
        io_threads: int = 4,
    ) -> torch.Tensor:
        """Encode all frames for one sequence and return a pooled ``(D,)`` vector.

        Images are loaded in parallel threads (io_threads) to hide disk latency.
        """
        loaded: List[Tuple[int, Image.Image]] = []
        with ThreadPoolExecutor(max_workers=io_threads) as ex:
            futures = {ex.submit(self._load_image, p): i for i, p in enumerate(frame_paths)}
            for fut in as_completed(futures):
                g_idx = futures[fut]
                img = fut.result()
                if img is not None:
                    loaded.append((g_idx, img))
        loaded.sort(key=lambda t: t[0])  # preserve frame order

        # X-CLIP video tower: the whole clip goes through in one forward pass
        # (cross-frame attention needs all frames together), then frame pooling
        # collapses the per-frame features exactly as in the per-frame path.
        if self.vision_feature_mode == "video":
            if not loaded:
                return torch.zeros(self.vit_hidden, device=device)
            feats = self._encode_video_clip(loaded, processor, device, vit_image_size)
            return pool_stack(feats, pooling)

        running, aux = self._init_running(device, self.vit_hidden, pooling)
        vit_ctx = torch.no_grad() if self._vit_is_frozen() else torch.enable_grad()

        for i in range(0, len(loaded), chunk_size):
            chunk = loaded[i: i + chunk_size]
            global_idxs = [t[0] for t in chunk]
            chunk_imgs = [t[1] for t in chunk]

            proc_kwargs: Dict[str, Any] = dict(images=chunk_imgs, return_tensors="pt")
            if vit_image_size is not None:
                proc_kwargs["size"] = {"height": vit_image_size, "width": vit_image_size}
            inputs = processor(**proc_kwargs)
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            with vit_ctx:
                out = self.vit(pixel_values=pixel_values)
                frame_emb = self._frame_features(out)  # (chunk, D)

                if self.temporal_on_frames and self.temporal_mode != "none":
                    pos_t = torch.tensor(global_idxs, device=device, dtype=torch.long)
                    frame_emb = self._add_temporal_encoding(
                        frame_emb, pos_t, self.frame_temporal_scale
                    )
                running, aux = self._update_running(running, aux, frame_emb, pooling)

            del pixel_values, out, frame_emb, inputs, chunk_imgs

        return self._finalize_running(running, aux, pooling)

    # ── training forward ──────────────────────────────────────────────────

    def forward(
        self,
        batch_sequences: List[List[List[Path]]],
        texts: List[str],
        processor,
        tokenizer,
        device: torch.device,
        frame_chunk_size: int = 16,
        vit_image_size: Optional[int] = None,
        io_threads: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(batch_sequences)
        assert B == len(texts), "Batch size mismatch between images and texts"

        study_visuals: List[torch.Tensor] = []
        for sequences in batch_sequences:
            seq_feats: List[torch.Tensor] = []
            for frame_paths in sequences:
                if not frame_paths:
                    continue
                seq_feat = self.encode_framepaths_pooled(
                    frame_paths=frame_paths,
                    processor=processor,
                    device=device,
                    chunk_size=frame_chunk_size,
                    pooling=self.frame_pooling,
                    vit_image_size=vit_image_size,
                    io_threads=io_threads,
                )
                seq_feats.append(seq_feat)

            if not seq_feats:
                study_feat = torch.zeros(self.vit_hidden, device=device)
            else:
                seq_stack = torch.stack(seq_feats, dim=0)
                if self.temporal_on_sequences and self.temporal_mode != "none":
                    seq_pos = torch.arange(seq_stack.size(0), device=device, dtype=torch.long)
                    seq_stack = self._add_temporal_encoding(
                        seq_stack, seq_pos, self.sequence_temporal_scale
                    )
                study_feat = pool_stack(seq_stack, self.sequence_pooling)

            study_visuals.append(study_feat)

        study_visuals = torch.stack(study_visuals, dim=0)                          # (B, D_vit)
        image_embeds = F.normalize(self.vision_proj(study_visuals), dim=-1)        # (B, E)

        tok = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        text_ctx = torch.no_grad() if self._text_is_frozen() else torch.enable_grad()
        with text_ctx:
            tout = self.text_model(**tok)
            tcls = tout.last_hidden_state[:, 0, :]                                  # (B, D_txt)

        text_embeds = F.normalize(self.text_proj(tcls), dim=-1)                    # (B, E)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        return image_embeds, text_embeds, logit_scale

    # ── inference helpers (validation) ────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
        tok = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        out = self.text_model(**tok)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.text_proj(cls), dim=-1)

    @torch.no_grad()
    def encode_sequence_from_frames(
        self,
        processor,
        frame_paths: List[Path],
        device: torch.device,
        frame_chunk_size: int,
        max_frames: Optional[int],
        sequence_repeat_factor: int = 1,
        vit_image_size: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Encode one validation sequence directory into a normalised embedding.

        Temporal encoding (frame + sequence) is applied identically to training,
        so temporal checkpoints evaluate faithfully. ``sequence_repeat_factor``
        replicates the pooled sequence before study-level pooling (legacy base
        validator behaviour; default 1 = no repeat).
        """
        frame_paths = uniform_subsample(frame_paths, max_frames)
        if not frame_paths:
            return None, "no_frames"

        # X-CLIP video tower: encode the sequence as one fixed-length clip.
        if self.vision_feature_mode == "video":
            loaded: List[Tuple[int, Image.Image]] = []
            for idx, p in enumerate(frame_paths):
                img = self._load_image(p)
                if img is not None:
                    loaded.append((idx, img))
            if not loaded:
                return None, "no_readable_frames"
            feats = self._encode_video_clip(loaded, processor, device, vit_image_size)
            sequence_feat = self._pool_tensor(feats, self.frame_pooling)
            return self._finish_sequence_embedding(
                sequence_feat, device, sequence_repeat_factor
            ), "ok"

        collected_frame_feats: List[torch.Tensor] = []

        for i in range(0, len(frame_paths), frame_chunk_size):
            chunk = frame_paths[i: i + frame_chunk_size]
            imgs: List[Image.Image] = []
            valid_positions: List[int] = []
            for local_idx, p in enumerate(chunk):
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid_positions.append(i + local_idx)
                except Exception:
                    continue
            if not imgs:
                continue

            proc_kwargs: Dict[str, Any] = dict(images=imgs, return_tensors="pt")
            if vit_image_size is not None:
                proc_kwargs["size"] = {"height": vit_image_size, "width": vit_image_size}
            inputs = processor(**proc_kwargs)
            pixel_values = inputs["pixel_values"].to(device)

            out = self.vit(pixel_values=pixel_values)
            feats = self._frame_features(out)

            if self.temporal_on_frames and self.temporal_mode != "none":
                pos_tensor = torch.tensor(valid_positions, device=device, dtype=torch.long)
                feats = self._add_temporal_encoding(feats, pos_tensor, self.frame_temporal_scale)

            collected_frame_feats.append(feats)
            del inputs, pixel_values, out, imgs

        if not collected_frame_feats:
            return None, "no_readable_frames"

        all_frame_feats = torch.cat(collected_frame_feats, dim=0)
        sequence_feat = self._pool_tensor(all_frame_feats, self.frame_pooling)
        return self._finish_sequence_embedding(
            sequence_feat, device, sequence_repeat_factor
        ), "ok"

    def _finish_sequence_embedding(
        self,
        sequence_feat: torch.Tensor,
        device: torch.device,
        sequence_repeat_factor: int,
    ) -> torch.Tensor:
        """Sequence feature -> study-like pooled, projected, normalised embedding."""
        repeat_n = max(1, int(sequence_repeat_factor))
        seq_stack = sequence_feat.unsqueeze(0).repeat(repeat_n, 1)

        if self.temporal_on_sequences and self.temporal_mode != "none":
            seq_positions = torch.arange(seq_stack.size(0), device=device, dtype=torch.long)
            seq_stack = self._add_temporal_encoding(seq_stack, seq_positions, self.sequence_temporal_scale)

        study_like_feat = self._pool_tensor(seq_stack, self.sequence_pooling)
        return F.normalize(self.vision_proj(study_like_feat.unsqueeze(0)), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter accounting / summary
# ─────────────────────────────────────────────────────────────────────────────

def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def count_all_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def print_trainable_summary(model: PooledCLIP) -> None:
    total = count_all_params(model)
    trainable = count_trainable_params(model)
    print(f"[INFO] Architecture:      {model.arch}")
    print(f"[INFO] Total params:      {total:,}")
    print(f"[INFO] Trainable params:  {trainable:,}")
    print(f"[INFO] Frozen params:     {total - trainable:,}")

    try:
        layers = model._vision_encoder_layers()
        trained_blocks = [
            i for i, b in enumerate(layers)
            if any(p.requires_grad for p in b.parameters())
        ]
        print(f"[INFO] ViT trainable encoder blocks: {trained_blocks} / total={len(layers)}")
    except RuntimeError:
        pass

    def _any_trainable(obj) -> bool:
        return obj is not None and any(p.requires_grad for p in obj.parameters())

    print(f"[INFO] ViT embeddings trainable:      {_any_trainable(getattr(model.vit, 'embeddings', None))}")
    print(f"[INFO] ViT final layernorm trainable: {_any_trainable(getattr(model.vit, 'layernorm', None))}")
    print(f"[INFO] Text encoder trainable:        {any(p.requires_grad for p in model.text_model.parameters())}")
    print(f"[INFO] vision_proj trainable:         {any(p.requires_grad for p in model.vision_proj.parameters())}")
    print(f"[INFO] text_proj trainable:           {any(p.requires_grad for p in model.text_proj.parameters())}")
    print(f"[INFO] temporal_mode:                 {model.temporal_mode}")
    print(f"[INFO] temporal_on_frames:            {model.temporal_on_frames} (scale={model.frame_temporal_scale})")
    print(f"[INFO] temporal_on_sequences:         {model.temporal_on_sequences} (scale={model.sequence_temporal_scale})")
    if arch_uses_sigmoid(model.arch) and model.siglip_bias is not None:
        print(f"[INFO] siglip_bias trainable:         {model.siglip_bias.requires_grad} (value={model.siglip_bias.item():.4f})")
