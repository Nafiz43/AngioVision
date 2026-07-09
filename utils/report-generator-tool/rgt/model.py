"""PooledCLIP model (mirrors the training script exactly) + checkpoint loader."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, ViTModel

from rgt.ui import err, info, section, success, warn

POOL_CHOICES = ("max", "mean", "logsumexp")


def build_sinusoidal_pe(
    pos: torch.Tensor, dim: int, device, dtype
) -> torch.Tensor:
    pos = pos.to(device=device, dtype=torch.float32)
    pe = torch.zeros(pos.size(0), dim, device=device)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div)
    pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype)


def pool_stack(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "max":
        return x.max(0).values
    if mode == "mean":
        return x.mean(0)
    return torch.logsumexp(x, 0)


class PooledCLIP(nn.Module):
    def __init__(
        self,
        vit_name:               str,
        text_model_name:        str,
        embed_dim:              int   = 256,
        frame_pooling:          str   = "max",
        sequence_pooling:       str   = "max",
        temporal_mode:          str   = "sinusoidal",
        temporal_on_frames:     bool  = True,
        temporal_on_sequences:  bool  = False,
        frame_temporal_scale:   float = 0.25,
        seq_temporal_scale:     float = 0.25,
        enable_generation:      bool  = True,
        decoder_model_name:     str   = "gpt2",
        gen_use_study_token:    bool  = True,
    ):
        super().__init__()
        self.vit        = ViTModel.from_pretrained(vit_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.vit_hidden  = self.vit.config.hidden_size
        self.text_hidden = self.text_model.config.hidden_size

        self.frame_pooling    = frame_pooling
        self.sequence_pooling = sequence_pooling
        self.temporal_mode    = temporal_mode
        self.temporal_frames  = temporal_on_frames
        self.temporal_seqs    = temporal_on_sequences
        self.frame_pe_scale   = frame_temporal_scale
        self.seq_pe_scale     = seq_temporal_scale

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vit_hidden, self.vit_hidden), nn.GELU(),
            nn.Linear(self.vit_hidden, embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden, self.text_hidden), nn.GELU(),
            nn.Linear(self.text_hidden, embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self.enable_generation   = enable_generation
        self.gen_use_study_token = gen_use_study_token
        self.decoder_model_name  = decoder_model_name
        self.report_decoder      = None
        self.decoder_hidden      = None
        self.generation_visual_proj = None

        if enable_generation:
            dcfg = AutoConfig.from_pretrained(decoder_model_name)
            setattr(dcfg, "add_cross_attention", True)
            setattr(dcfg, "is_decoder",          True)
            self.report_decoder = AutoModelForCausalLM.from_pretrained(
                decoder_model_name, config=dcfg, ignore_mismatched_sizes=True
            )
            for attr in ("n_embd", "hidden_size", "d_model"):
                if hasattr(self.report_decoder.config, attr):
                    self.decoder_hidden = int(getattr(self.report_decoder.config, attr))
                    break
            if self.decoder_hidden is None:
                raise RuntimeError("Cannot infer decoder hidden size from config.")
            self.generation_visual_proj = nn.Sequential(
                nn.Linear(self.vit_hidden, self.vit_hidden), nn.GELU(),
                nn.Linear(self.vit_hidden, self.decoder_hidden),
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_pe(
        self, x: torch.Tensor, pos: torch.Tensor, scale: float
    ) -> torch.Tensor:
        if self.temporal_mode == "none" or scale == 0.0:
            return x
        return x + scale * build_sinusoidal_pe(pos, x.size(-1), x.device, x.dtype)

    @torch.no_grad()
    def _vit_chunked(self, pv: torch.Tensor, chunk: int) -> torch.Tensor:
        if pv.size(0) == 0:
            return torch.zeros(0, self.vit_hidden, device=pv.device, dtype=pv.dtype)
        parts = []
        for i in range(0, pv.size(0), chunk):
            out = self.vit(pixel_values=pv[i: i + chunk])
            parts.append(out.last_hidden_state[:, 0, :])
        return torch.cat(parts, 0)

    # ── Visual encoder ────────────────────────────────────────────────────────

    def encode_visual_batch(
        self,
        batch_pv:   List[List[torch.Tensor]],
        batch_pos:  List[List[torch.Tensor]],
        device:     torch.device,
        chunk:      int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(batch_pv)
        if B == 0:
            z = torch.zeros
            return (
                z(0, self.vit_hidden, device=device),
                z(0, 1, self.vit_hidden, device=device),
                z(0, 1, device=device, dtype=torch.long),
            )

        per_seq_lens: List[List[int]] = [[] for _ in range(B)]
        flat_pv, flat_pos = [], []
        for s, seqs in enumerate(batch_pv):
            for q, seq_pv in enumerate(seqs):
                T = int(seq_pv.size(0))
                per_seq_lens[s].append(T)
                if T:
                    flat_pv.append(seq_pv)
                    flat_pos.append(batch_pos[s][q])

        big_pv = (
            torch.cat(flat_pv, 0).to(device, non_blocking=True)
            if flat_pv
            else torch.zeros(0, 3, 1, 1, device=device)
        )
        all_emb = self._vit_chunked(big_pv, chunk)

        per_study_seqs: List[torch.Tensor] = []
        cursor = flat_idx = 0
        for s in range(B):
            seq_feats = []
            for T in per_seq_lens[s]:
                if T == 0:
                    continue
                emb = all_emb[cursor: cursor + T]
                cursor += T
                if self.temporal_frames and self.temporal_mode != "none":
                    emb = self._add_pe(
                        emb,
                        flat_pos[flat_idx].to(device, dtype=torch.long),
                        self.frame_pe_scale,
                    )
                seq_feats.append(pool_stack(emb, self.frame_pooling))
                flat_idx += 1
            ss = (
                torch.stack(seq_feats)
                if seq_feats
                else torch.zeros(1, self.vit_hidden, device=device)
            )
            per_study_seqs.append(ss)

        study_vecs = []
        for s in range(B):
            ss = per_study_seqs[s]
            if self.temporal_seqs and self.temporal_mode != "none":
                pos = torch.arange(ss.size(0), device=device, dtype=torch.long)
                ss = self._add_pe(ss, pos, self.seq_pe_scale)
                per_study_seqs[s] = ss
            study_vecs.append(pool_stack(ss, self.sequence_pooling))

        sv = torch.stack(study_vecs)
        max_seq = max(x.size(0) for x in per_study_seqs)
        prepend = 1 if self.gen_use_study_token else 0

        tok = torch.zeros(B, max_seq + prepend, self.vit_hidden,
                          device=device, dtype=sv.dtype)
        mask = torch.zeros(B, max_seq + prepend, device=device, dtype=torch.long)
        for i in range(B):
            off = 0
            if self.gen_use_study_token:
                tok[i, 0] = sv[i]
                mask[i, 0] = 1
                off = 1
            ss = per_study_seqs[i]
            tok[i, off: off + ss.size(0)] = ss
            mask[i, off: off + ss.size(0)] = 1

        return sv, tok, mask

    # ── Prompt-conditioned generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate_from_prompt(
        self,
        visual_tokens:        torch.Tensor,   # (1, S, vit_hidden)
        visual_attn_mask:     torch.Tensor,   # (1, S)
        gen_tokenizer,
        prompt_text:          str,
        device:               torch.device,
        max_new_tokens:       int   = 1024,
        do_sample:            bool  = False,
        top_p:                float = 1.0,
        temperature:          float = 0.5,
        repetition_penalty:   float = 1.5,
        no_repeat_ngram_size: int   = 4,
    ) -> str:
        if not self.enable_generation or self.report_decoder is None:
            raise RuntimeError("Generation head not enabled in this model.")

        # Project visual tokens to decoder hidden size once
        proj = self.generation_visual_proj(visual_tokens)   # (1, S, decoder_hidden)

        enc = gen_tokenizer(prompt_text, return_tensors="pt").to(device)

        # HuggingFace .generate() forwards any extra kwargs it doesn't consume
        # to the model's forward() call, so encoder_hidden_states /
        # encoder_attention_mask reach the cross-attention layers correctly.
        gen_kwargs = dict(
            input_ids              = enc["input_ids"],
            attention_mask         = enc["attention_mask"],
            encoder_hidden_states  = proj,
            encoder_attention_mask = visual_attn_mask,
            max_new_tokens         = max_new_tokens,
            repetition_penalty     = repetition_penalty,
            no_repeat_ngram_size   = no_repeat_ngram_size,
            pad_token_id           = (gen_tokenizer.pad_token_id
                                      or gen_tokenizer.eos_token_id),
            eos_token_id           = gen_tokenizer.eos_token_id,
        )

        if do_sample:
            gen_kwargs["do_sample"]   = True
            gen_kwargs["top_p"]       = top_p
            gen_kwargs["temperature"] = max(temperature, 1e-3)
        else:
            gen_kwargs["do_sample"]   = False

        output_ids = self.report_decoder.generate(**gen_kwargs)

        # Decode only the newly generated tokens (skip the prompt prefix)
        prompt_len = enc["input_ids"].shape[-1]
        new_ids = output_ids[0][prompt_len:]
        return gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def load_model(args, device: torch.device) -> PooledCLIP:
    section("Loading model architecture")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        err(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    info(f"Reading checkpoint → {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    # Prefer model_config saved in the checkpoint (requires patched train
    # script). Fall back to CLI flags if the key is absent (old checkpoints).
    cfg = ckpt.get("model_config")
    if cfg is not None:
        success("Found model_config in checkpoint — reconstructing arch from it")
        model = PooledCLIP(**cfg)
    else:
        warn(
            "Checkpoint has no model_config key (old-format checkpoint). "
            "Reconstructing from CLI flags — make sure they match training exactly."
        )
        model = PooledCLIP(
            vit_name               = args.vit_name,
            text_model_name        = args.bert_name,
            embed_dim              = args.embed_dim,
            frame_pooling          = args.pooling,
            sequence_pooling       = args.pooling,
            temporal_mode          = "sinusoidal",
            temporal_on_frames     = True,
            temporal_on_sequences  = True,
            frame_temporal_scale   = args.frame_temporal_scale,
            seq_temporal_scale     = args.seq_temporal_scale,
            enable_generation      = True,
            decoder_model_name     = args.decoder_model_name,
            gen_use_study_token    = True,
        )

    miss, unex = model.load_state_dict(state, strict=False)
    if miss:
        warn(f"Missing keys  ({len(miss)}): {miss[:4]}{'…' if len(miss) > 4 else ''}")
    if unex:
        warn(f"Unexpected keys ({len(unex)}): {unex[:4]}{'…' if len(unex) > 4 else ''}")

    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("step", "?")
    success(f"Checkpoint loaded  |  epoch={epoch}  step={step}")
    model.eval().to(device)
    return model
