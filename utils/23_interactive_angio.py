#!/usr/bin/env python3
"""
angiovision_interactive.py

Interactive CLI for AngioVision — no external API required.
Everything is powered by your fine-tuned checkpoint.

Workflow:
  1. Load the trained PooledCLIP checkpoint.
  2. List all studies in the holdout CSV; user picks one or more.
  3. Encode the selected sequences → visual tokens (kept in memory).
  4. Generate the initial free-form report via the decoder.
  5. Enter a Q&A loop:
       Each question is wrapped as:
           "Report: <generated report>
            Q: <user question>
            A:"
       and fed to the decoder with the SAME visual cross-attention tokens,
       so every answer is grounded in both the images and the prior report.

Usage:
    python angiovision_interactive.py \\
        --checkpoint /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/gen/500_16_16_32/last.pt \\
        [--decoder_model_name microsoft/biogpt | gpt2] \\
        [--vit_name google/vit-base-patch16-224-in21k] \\
        [--bert_name dmis-lab/biobert-base-cased-v1.1] \\
        [--embed_dim 256] \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTModel,
)

try:
    from transformers import ViTImageProcessor as _ViTProcessor
except Exception:
    _ViTProcessor = None
try:
    from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
except Exception:
    _ViTFeatureExtractor = None


# ─────────────────────────────────────────────────────────────
# Terminal colour helpers
# ─────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def banner(msg: str) -> None:
    try:
        width = min(80, os.get_terminal_size().columns)
    except OSError:
        width = 80
    print(c("─" * width, CYAN))
    print(c(f"  {msg}", BOLD + CYAN))
    print(c("─" * width, CYAN))


def section(msg: str)  -> None: print(f"\n{c('▶', GREEN)} {c(msg, BOLD)}\n")
def info(msg: str)     -> None: print(c(f"  ℹ  {msg}", DIM))
def warn(msg: str)     -> None: print(c(f"  ⚠  {msg}", YELLOW))
def err(msg: str)      -> None: print(c(f"  ✗  {msg}", RED))
def success(msg: str)  -> None: print(c(f"  ✔  {msg}", GREEN))


def prompt(msg: str) -> str:
    return input(c(f"\n  ❯  {msg}: ", BOLD + MAGENTA))


# ─────────────────────────────────────────────────────────────
# Data utilities  (self-contained; mirrors train script)
# ─────────────────────────────────────────────────────────────
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
POOL_CHOICES = ("max", "mean", "logsumexp")


def parse_sop_instance_uids(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    s = str(val).strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[0] == s[-1]:
        s = s[1:-1].strip()
    if not s:
        return []
    if s[0] in "[(":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [t.strip() for t in re.split(r"\s*,\s*", s) if t.strip()]


def _list_images(d: Path) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_frame_files(base: Path, acc: str, sop: str) -> List[Path]:
    sop_dir = base / str(acc).strip() / str(sop).strip()
    for cand in [sop_dir / "frames", sop_dir]:
        imgs = _list_images(cand)
        if imgs:
            return imgs
    if sop_dir.is_dir():
        nested: List[Path] = []
        for ch in sop_dir.iterdir():
            if ch.is_dir():
                nested.extend(_list_images(ch))
        if nested:
            return sorted(nested)
    return []


def get_vit_processor(name: str):
    if _ViTProcessor is not None:
        return _ViTProcessor.from_pretrained(name)
    if _ViTFeatureExtractor is not None:
        return _ViTFeatureExtractor.from_pretrained(name)
    raise ImportError("No ViT image processor available in transformers.")


def preprocess_frames(
    frame_files: List[Path],
    processor,
    vit_image_size: Optional[int],
) -> Tuple[Optional[torch.Tensor], List[int]]:
    imgs, valid = [], []
    for i, p in enumerate(frame_files):
        try:
            imgs.append(Image.open(p).convert("RGB"))
            valid.append(i)
        except Exception:
            continue
    if not imgs:
        return None, []
    kw: Dict[str, Any] = dict(images=imgs, return_tensors="pt")
    if vit_image_size:
        kw["size"] = {"height": vit_image_size, "width": vit_image_size}
    pv = processor(**kw)["pixel_values"]
    for img in imgs:
        try:
            img.close()
        except Exception:
            pass
    return pv, valid


def build_sinusoidal_pe(
    pos: torch.Tensor, dim: int, device, dtype
) -> torch.Tensor:
    pos = pos.to(device=device, dtype=torch.float32)
    pe  = torch.zeros(pos.size(0), dim, device=device)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div)
    pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype)


def pool_stack(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "max":  return x.max(0).values
    if mode == "mean": return x.mean(0)
    return torch.logsumexp(x, 0)


# ─────────────────────────────────────────────────────────────
# Model  (mirrors PooledCLIP from training script exactly)
# ─────────────────────────────────────────────────────────────
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
        self.generation_visual_proj     = None

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
                ss  = self._add_pe(ss, pos, self.seq_pe_scale)
                per_study_seqs[s] = ss
            study_vecs.append(pool_stack(ss, self.sequence_pooling))

        sv = torch.stack(study_vecs)
        max_seq = max(x.size(0) for x in per_study_seqs)
        prepend = 1 if self.gen_use_study_token else 0

        tok  = torch.zeros(B, max_seq + prepend, self.vit_hidden,
                           device=device, dtype=sv.dtype)
        mask = torch.zeros(B, max_seq + prepend, device=device, dtype=torch.long)
        for i in range(B):
            off = 0
            if self.gen_use_study_token:
                tok[i, 0] = sv[i]; mask[i, 0] = 1; off = 1
            ss = per_study_seqs[i]
            tok [i, off: off + ss.size(0)] = ss
            mask[i, off: off + ss.size(0)] = 1

        return sv, tok, mask

    # ── Prompt-conditioned generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate_from_prompt(
        self,
        visual_tokens:       torch.Tensor,   # (1, S, vit_hidden)
        visual_attn_mask:    torch.Tensor,   # (1, S)
        gen_tokenizer,
        prompt_text:         str,
        device:              torch.device,
        max_new_tokens:      int   = 1024,
        do_sample:           bool  = False,
        top_p:               float = 1.0,
        temperature:         float = 0.5,
        repetition_penalty:  float = 1.5,
        no_repeat_ngram_size: int  = 4,
    ) -> str:
        if not self.enable_generation or self.report_decoder is None:
            raise RuntimeError("Generation head not enabled in this model.")

        # Project visual tokens to decoder hidden size once
        proj = self.generation_visual_proj(visual_tokens)   # (1, S, decoder_hidden)

        enc = gen_tokenizer(prompt_text, return_tensors="pt").to(device)

        # Build generation kwargs — HuggingFace .generate() forwards any extra
        # kwargs it doesn't consume directly to the model's forward() call,
        # so encoder_hidden_states / encoder_attention_mask reach the cross-
        # attention layers correctly.
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
            gen_kwargs["do_sample"]    = True
            gen_kwargs["top_p"]        = top_p
            gen_kwargs["temperature"]  = max(temperature, 1e-3)
        else:
            gen_kwargs["do_sample"]    = False

        output_ids = self.report_decoder.generate(**gen_kwargs)

        # Decode only the newly generated tokens (skip the prompt prefix)
        prompt_len = enc["input_ids"].shape[-1]
        new_ids    = output_ids[0][prompt_len:]
        return gen_tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────
def load_model(args, device: torch.device) -> PooledCLIP:
    section("Loading model architecture")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        err(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    info(f"Reading checkpoint → {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)

    # ── Prefer model_config saved in the checkpoint (requires patched train
    #    script).  Fall back to CLI flags if the key is absent (old checkpoints).
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
    step  = ckpt.get("step",  "?")
    success(f"Checkpoint loaded  |  epoch={epoch}  step={step}")
    model.eval().to(device)
    return model


# ─────────────────────────────────────────────────────────────
# Holdout study helpers
# ─────────────────────────────────────────────────────────────
def load_holdout_studies(
    meta_csv:  Path,
    anon_col:  str,
    sop_col:   str,
    base_dir:  Path,
) -> List[Dict]:
    df = pd.read_csv(meta_csv)
    studies = []
    for _, row in df.iterrows():
        acc = str(row.get(anon_col, "")).strip()
        if not acc:
            continue
        sops     = parse_sop_instance_uids(row.get(sop_col, ""))
        seq_info = [
            {"sop": s, "n_frames": len(find_frame_files(base_dir, acc, s))}
            for s in sops
        ]
        studies.append({"acc": acc, "sop_uids": sops, "seq_info": seq_info})
    return studies


def display_study_table(studies: List[Dict]) -> None:
    hdr = f"  {'#':>3}  {'Accession':<22}  {'Seqs':>5}  {'Frames':>8}"
    print(c(hdr, BOLD))
    print(c("  " + "─" * (len(hdr) - 2), DIM))
    for i, s in enumerate(studies):
        frames = sum(si["n_frames"] for si in s["seq_info"])
        row    = f"  {i+1:>3}  {s['acc']:<22}  {len(s['sop_uids']):>5}  {frames:>8}"
        print(c(row, CYAN if i % 2 == 0 else ""))
    print()


def pick_studies(studies: List[Dict]) -> List[Dict]:
    display_study_table(studies)
    while True:
        raw = prompt(
            "Select study number(s)  (e.g. 1  or  1,3  or  all)"
        ).strip().lower()
        if not raw:
            continue
        if raw == "all":
            return studies
        parts = re.split(r"[\s,]+", raw)
        sel, ok = [], True
        for p in parts:
            if not p.isdigit():
                err(f"'{p}' is not a number.")
                ok = False
                break
            idx = int(p) - 1
            if not (0 <= idx < len(studies)):
                err(f"Index {int(p)} out of range (1–{len(studies)}).")
                ok = False
                break
            sel.append(studies[idx])
        if ok and sel:
            return sel


def pick_sequences(study: Dict) -> Dict:
    si = study["seq_info"]
    if len(si) <= 1:
        return study

    print(f"\n  Study {c(study['acc'], BOLD)} has {len(si)} sequences:\n")
    print(c(f"    {'#':>3}  {'SOP (truncated)':<44}  {'Frames':>6}", BOLD))
    print(c("    " + "─" * 56, DIM))
    for i, s in enumerate(si):
        short = s["sop"][:42] + ".." if len(s["sop"]) > 42 else s["sop"]
        # short = s["sop"]
        print(f"    {i+1:>3}  {short:<44}  {s['n_frames']:>6}")

    raw = prompt(
        "Use which sequences? (e.g. 1,2  or  all — default: all)"
    ).strip().lower()
    if not raw or raw == "all":
        return study
    keep = {
        int(p) - 1
        for p in re.split(r"[\s,]+", raw)
        if p.isdigit() and 1 <= int(p) <= len(si)
    }
    if not keep:
        warn("Invalid selection — using all sequences.")
        return study
    return {
        "acc":      study["acc"],
        "sop_uids": [study["sop_uids"][i] for i in sorted(keep)],
        "seq_info": [si[i]               for i in sorted(keep)],
    }


# ─────────────────────────────────────────────────────────────
# Visual encoding  (result is cached per-study for Q&A)
# ─────────────────────────────────────────────────────────────
def encode_study(
    model:     PooledCLIP,
    study:     Dict,
    base_dir:  Path,
    processor,
    device:    torch.device,
    args,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns (visual_tokens, visual_mask) shaped (1,S,D) and (1,S) on device,
    or None if no frames could be loaded.
    """
    seq_pvs, seq_pos = [], []
    for sop in study["sop_uids"]:
        files = find_frame_files(base_dir, study["acc"], sop)
        if args.max_frames_per_sequence and len(files) > args.max_frames_per_sequence:
            idxs  = torch.linspace(
                0, len(files) - 1, args.max_frames_per_sequence
            ).long().tolist()
            files = [files[i] for i in idxs]
        if not files:
            continue
        pv, vpos = preprocess_frames(files, processor, args.vit_image_size)
        if pv is None or pv.size(0) == 0:
            continue
        seq_pvs.append(pv)
        seq_pos.append(torch.tensor(vpos, dtype=torch.long))

    if not seq_pvs:
        return None

    with torch.no_grad():
        _, vtok, vmask = model.encode_visual_batch(
            [seq_pvs], [seq_pos], device, chunk=args.frame_chunk_size
        )
    return vtok, vmask


# ─────────────────────────────────────────────────────────────
# Report generation & Q&A wrappers
# ─────────────────────────────────────────────────────────────
def generate_report(
    model:  PooledCLIP,
    vtok:   torch.Tensor,
    vmask:  torch.Tensor,
    gen_tok,
    device: torch.device,
    args,
) -> str:
    """Generate an initial free-form radiology report from visual tokens."""
    bos = gen_tok.bos_token or gen_tok.eos_token or ""
    return model.generate_from_prompt(
        visual_tokens        = vtok,
        visual_attn_mask     = vmask,
        gen_tokenizer        = gen_tok,
        prompt_text          = bos,
        device               = device,
        max_new_tokens       = args.max_new_tokens,
        do_sample            = args.do_sample,
        top_p                = args.top_p,
        temperature          = args.temperature,
        repetition_penalty   = args.repetition_penalty,
        no_repeat_ngram_size = args.no_repeat_ngram_size,
    )


def answer_question(
    model:    PooledCLIP,
    vtok:     torch.Tensor,
    vmask:    torch.Tensor,
    report:   str,
    question: str,
    gen_tok,
    device:   torch.device,
    args,
) -> str:
    """
    Construct a grounded prompt:
        Report: <generated report>
        Q: <user question>
        A:
    Feed it to the decoder together with the visual cross-attention tokens.
    """
    max_rpt_chars = 800
    trimmed = (report[:max_rpt_chars] + "…") if len(report) > max_rpt_chars else report
    prompt_text = (
        f"Report: {trimmed.strip()}\n"
        f"Q: {question.strip()}\n"
        f"A:"
    )
    answer = model.generate_from_prompt(
        visual_tokens        = vtok,
        visual_attn_mask     = vmask,
        gen_tokenizer        = gen_tok,
        prompt_text          = prompt_text,
        device               = device,
        max_new_tokens       = args.qa_max_new_tokens,
        do_sample            = args.do_sample,
        top_p                = args.top_p,
        temperature          = args.temperature,
        repetition_penalty   = args.repetition_penalty,
        no_repeat_ngram_size = args.no_repeat_ngram_size,
    )
    return answer.strip()


# ─────────────────────────────────────────────────────────────
# Per-study interactive session
# ─────────────────────────────────────────────────────────────
def _print_report_box(title: str, text: str) -> None:
    bar = "═" * (74 - len(title) - 2)
    print(c(f"  ╔═ {title} {bar}", CYAN))
    for line in textwrap.wrap(text.strip() or "(empty)", width=72):
        print(c("  ║  ", CYAN) + line)
    print(c("  ╚" + "═" * 74, CYAN))
    print()


def run_study_session(
    model:     PooledCLIP,
    study:     Dict,
    base_dir:  Path,
    processor,
    gen_tok,
    device:    torch.device,
    args,
) -> None:
    acc = study["acc"]
    section(f"Study: {acc}")

    # ── Encode visual tokens once; cache for the whole session ────────────────
    info("Encoding visual sequences …")
    result = encode_study(model, study, base_dir, processor, device, args)
    if result is None:
        warn(f"No valid frames found for {acc} — skipping.")
        return
    vtok, vmask = result
    n_seqs = sum(1 for s in study["seq_info"] if s["n_frames"] > 0)
    success(
        f"Encoded {n_seqs} sequence(s)  →  "
        f"visual token shape {tuple(vtok.shape)}"
    )

    # ── Generate initial report ───────────────────────────────────────────────
    info("Generating report …")
    report = generate_report(model, vtok, vmask, gen_tok, device, args)
    _print_report_box("Generated Report", report)

    # ── Q&A loop ──────────────────────────────────────────────────────────────
    section("Q&A  —  your question → decoder answers using visual + report context")
    print(c("  Commands:", BOLD))
    print(c("    show   — re-print the full report", DIM))
    print(c("    regen  — regenerate the report from scratch", DIM))
    print(c("    back   — return to study selection", DIM))
    print(c("    quit   — exit the tool", DIM))
    print()

    while True:
        try:
            user_q = prompt("You").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_q:
            continue

        cmd = user_q.lower()

        if cmd in ("back", "b"):
            break

        if cmd in ("quit", "exit", "q"):
            banner("Session ended  ◈  Thank you for using AngioVision")
            sys.exit(0)

        if cmd == "show":
            _print_report_box("Generated Report", report)
            continue

        if cmd == "regen":
            info("Regenerating report …")
            report = generate_report(model, vtok, vmask, gen_tok, device, args)
            _print_report_box("Regenerated Report", report)
            continue

        # Regular question
        info("Generating answer …")
        answer = answer_question(
            model, vtok, vmask, report, user_q, gen_tok, device, args
        )
        print()
        print(c("  Model:", BOLD + GREEN))
        for line in textwrap.wrap(answer or "(no answer generated)", width=74):
            print("  " + line)
        print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def build_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="AngioVision Interactive CLI (fine-tuned model only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Checkpoint ────────────────────────────────────────────────────────────
    ap.add_argument(
        "--checkpoint",
        default=(
            "/data/Deep_Angiography/AngioVision/fine-tuning/"
            "checkpoints/500_16_16_32/last.pt"
        ),
        help="Path to trained .pt checkpoint",
    )
    # ── Architecture (matches training command exactly) ───────────────────────
    ap.add_argument("--vit_name",  default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--bert_name", default="UCSD-VA-health/RadBERT-RoBERTa-4m")
    ap.add_argument("--decoder_model_name", default="gpt2")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pooling",   default="logsumexp", choices=list(POOL_CHOICES))
    # ── Holdout data ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--holdout_meta_csv",
        default=(
            "/data/Deep_Angiography/Validation_Data/"
            "Validation_Data_2026_03_23/consolidated_metadata_ALL_Sequences.csv"
        ),
    )
    ap.add_argument(
        "--holdout_base_frames_dir",
        default=(
            "/data/Deep_Angiography/Validation_Data/"
            "Validation_Data_2026_03_23/DICOM_Sequence_Processed"
        ),
    )
    ap.add_argument("--anon_col", default="Anon Acc #")
    ap.add_argument("--sop_col",  default="SOPInstanceUIDs")
    # ── Inference knobs ───────────────────────────────────────────────────────
    ap.add_argument("--device",                  default=None,
                    help="cuda or cpu (auto-detect if omitted)")
    ap.add_argument("--frame_chunk_size",        type=int,   default=32)
    ap.add_argument("--max_new_tokens",          type=int,   default=256,
                    help="Max tokens for the initial report")
    ap.add_argument("--qa_max_new_tokens",       type=int,   default=128,
                    help="Max tokens for each Q&A answer")
    ap.add_argument("--max_frames_per_sequence", type=int,   default=32)
    ap.add_argument("--vit_image_size",          type=int,   default=None)
    ap.add_argument("--frame_temporal_scale",    type=float, default=0.75)
    ap.add_argument("--seq_temporal_scale",      type=float, default=0.5)
    ap.add_argument("--do_sample",               action="store_true")
    ap.add_argument("--top_p",                   type=float, default=1.0)
    ap.add_argument("--temperature",             type=float, default=1.0)
    ap.add_argument("--repetition_penalty",      type=float, default=1.5,
                    help=">1.0 penalises repeated tokens — fixes EAT/EAT loops")
    ap.add_argument("--no_repeat_ngram_size",    type=int,   default=4,
                    help="Forbid repeating any n-gram of this size")
    return ap


def main() -> None:
    args = build_args().parse_args()
    banner("AngioVision Interactive  ◈  Powered by your fine-tuned model")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args, device)

    # ── Tokenisers & processor ────────────────────────────────────────────────
    section("Loading tokenisers & image processor")
    processor = get_vit_processor(args.vit_name)
    gen_tok   = AutoTokenizer.from_pretrained(args.decoder_model_name)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token or "<|pad|>"
    success("Ready")

    # ── Holdout metadata ──────────────────────────────────────────────────────
    section("Loading holdout studies")
    meta_csv = Path(args.holdout_meta_csv)
    base_dir = Path(args.holdout_base_frames_dir)
    if not meta_csv.exists():
        err(f"Holdout CSV not found: {meta_csv}")
        sys.exit(1)
    if not base_dir.exists():
        warn(f"Frames dir not found: {base_dir}  (frame counts will show 0)")

    studies = load_holdout_studies(meta_csv, args.anon_col, args.sop_col, base_dir)
    if not studies:
        err("No studies found in holdout CSV.")
        sys.exit(1)
    success(f"Found {len(studies)} holdout studies")

    # ── Main interaction loop ─────────────────────────────────────────────────
    while True:
        section("Study Selection")
        selected = pick_studies(studies)

        for raw_study in selected:
            refined = pick_sequences(raw_study)
            run_study_session(
                model, refined, base_dir, processor, gen_tok, device, args
            )

        again = prompt("Analyse more studies? (y / n)").strip().lower()
        if again not in ("y", "yes"):
            break

    banner("Session ended  ◈  Thank you for using AngioVision")


if __name__ == "__main__":
    main()
    
# #!/usr/bin/env python3
# """
# angiovision_interactive.py

# Interactive CLI for AngioVision — no external API required.
# Everything is powered by your fine-tuned checkpoint.

# Workflow:
#   1. Load the trained PooledCLIP checkpoint.
#   2. List all studies in the holdout CSV; user picks one or more.
#   3. Encode the selected sequences → visual tokens (kept in memory).
#   4. Generate the initial free-form report via the decoder.
#   5. Enter a Q&A loop:
#        Each question is wrapped as:
#            "Report: <generated report>
#             Q: <user question>
#             A:"
#        and fed to the decoder with the SAME visual cross-attention tokens,
#        so every answer is grounded in both the images and the prior report.

# Usage:
#     python angiovision_interactive.py \\
#         --checkpoint /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/500_16_16_32/last.pt \\
#         [--decoder_model_name microsoft/biogpt | gpt2] \\
#         [--vit_name google/vit-base-patch16-224-in21k] \\
#         [--bert_name dmis-lab/biobert-base-cased-v1.1] \\
#         [--embed_dim 256] \\
#         [--device cuda]
# """

# from __future__ import annotations

# import argparse
# import ast
# import math
# import os
# import re
# import sys
# import textwrap
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import pandas as pd
# import torch
# import torch.nn as nn
# from PIL import Image
# from transformers import (
#     AutoConfig,
#     AutoModel,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     ViTModel,
# )

# try:
#     from transformers import ViTImageProcessor as _ViTProcessor
# except Exception:
#     _ViTProcessor = None
# try:
#     from transformers import ViTFeatureExtractor as _ViTFeatureExtractor
# except Exception:
#     _ViTFeatureExtractor = None


# # ─────────────────────────────────────────────────────────────
# # Terminal colour helpers
# # ─────────────────────────────────────────────────────────────
# RESET   = "\033[0m"
# BOLD    = "\033[1m"
# CYAN    = "\033[96m"
# GREEN   = "\033[92m"
# YELLOW  = "\033[93m"
# RED     = "\033[91m"
# MAGENTA = "\033[95m"
# DIM     = "\033[2m"


# def c(text: str, colour: str) -> str:
#     return f"{colour}{text}{RESET}"


# def banner(msg: str) -> None:
#     try:
#         width = min(80, os.get_terminal_size().columns)
#     except OSError:
#         width = 80
#     print(c("─" * width, CYAN))
#     print(c(f"  {msg}", BOLD + CYAN))
#     print(c("─" * width, CYAN))


# def section(msg: str)  -> None: print(f"\n{c('▶', GREEN)} {c(msg, BOLD)}\n")
# def info(msg: str)     -> None: print(c(f"  ℹ  {msg}", DIM))
# def warn(msg: str)     -> None: print(c(f"  ⚠  {msg}", YELLOW))
# def err(msg: str)      -> None: print(c(f"  ✗  {msg}", RED))
# def success(msg: str)  -> None: print(c(f"  ✔  {msg}", GREEN))


# def prompt(msg: str) -> str:
#     return input(c(f"\n  ❯  {msg}: ", BOLD + MAGENTA))


# # ─────────────────────────────────────────────────────────────
# # Data utilities  (self-contained; mirrors train script)
# # ─────────────────────────────────────────────────────────────
# IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# POOL_CHOICES = ("max", "mean", "logsumexp")


# def parse_sop_instance_uids(val) -> List[str]:
#     if val is None:
#         return []
#     if isinstance(val, float) and pd.isna(val):
#         return []
#     s = str(val).strip()
#     if len(s) >= 2 and s[0] in ('"', "'") and s[0] == s[-1]:
#         s = s[1:-1].strip()
#     if not s:
#         return []
#     if s[0] in "[(":
#         try:
#             parsed = ast.literal_eval(s)
#             if isinstance(parsed, (list, tuple)):
#                 return [str(x).strip() for x in parsed if str(x).strip()]
#         except Exception:
#             pass
#     return [t.strip() for t in re.split(r"\s*,\s*", s) if t.strip()]


# def _list_images(d: Path) -> List[Path]:
#     if not d.exists() or not d.is_dir():
#         return []
#     return sorted(p for p in d.iterdir()
#                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# def find_frame_files(base: Path, acc: str, sop: str) -> List[Path]:
#     sop_dir = base / str(acc).strip() / str(sop).strip()
#     for cand in [sop_dir / "frames", sop_dir]:
#         imgs = _list_images(cand)
#         if imgs:
#             return imgs
#     if sop_dir.is_dir():
#         nested: List[Path] = []
#         for ch in sop_dir.iterdir():
#             if ch.is_dir():
#                 nested.extend(_list_images(ch))
#         if nested:
#             return sorted(nested)
#     return []


# def get_vit_processor(name: str):
#     if _ViTProcessor is not None:
#         return _ViTProcessor.from_pretrained(name)
#     if _ViTFeatureExtractor is not None:
#         return _ViTFeatureExtractor.from_pretrained(name)
#     raise ImportError("No ViT image processor available in transformers.")


# def preprocess_frames(
#     frame_files: List[Path],
#     processor,
#     vit_image_size: Optional[int],
# ) -> Tuple[Optional[torch.Tensor], List[int]]:
#     imgs, valid = [], []
#     for i, p in enumerate(frame_files):
#         try:
#             imgs.append(Image.open(p).convert("RGB"))
#             valid.append(i)
#         except Exception:
#             continue
#     if not imgs:
#         return None, []
#     kw: Dict[str, Any] = dict(images=imgs, return_tensors="pt")
#     if vit_image_size:
#         kw["size"] = {"height": vit_image_size, "width": vit_image_size}
#     pv = processor(**kw)["pixel_values"]
#     for img in imgs:
#         try:
#             img.close()
#         except Exception:
#             pass
#     return pv, valid


# def build_sinusoidal_pe(
#     pos: torch.Tensor, dim: int, device, dtype
# ) -> torch.Tensor:
#     pos = pos.to(device=device, dtype=torch.float32)
#     pe  = torch.zeros(pos.size(0), dim, device=device)
#     div = torch.exp(
#         torch.arange(0, dim, 2, device=device, dtype=torch.float32)
#         * (-math.log(10000.0) / dim)
#     )
#     pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div)
#     pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div[: pe[:, 1::2].shape[1]])
#     return pe.to(dtype)


# def pool_stack(x: torch.Tensor, mode: str) -> torch.Tensor:
#     if mode == "max":  return x.max(0).values
#     if mode == "mean": return x.mean(0)
#     return torch.logsumexp(x, 0)


# # ─────────────────────────────────────────────────────────────
# # Model  (mirrors PooledCLIP from training script exactly)
# # ─────────────────────────────────────────────────────────────
# class PooledCLIP(nn.Module):
#     def __init__(
#         self,
#         vit_name:               str,
#         text_model_name:        str,
#         embed_dim:              int   = 256,
#         frame_pooling:          str   = "max",
#         sequence_pooling:       str   = "max",
#         temporal_mode:          str   = "sinusoidal",
#         temporal_on_frames:     bool  = True,
#         temporal_on_sequences:  bool  = False,
#         frame_temporal_scale:   float = 0.25,
#         seq_temporal_scale:     float = 0.25,
#         enable_generation:      bool  = True,
#         decoder_model_name:     str   = "gpt2",
#         gen_use_study_token:    bool  = True,
#     ):
#         super().__init__()
#         self.vit        = ViTModel.from_pretrained(vit_name)
#         self.text_model = AutoModel.from_pretrained(text_model_name)
#         self.vit_hidden  = self.vit.config.hidden_size
#         self.text_hidden = self.text_model.config.hidden_size

#         self.frame_pooling    = frame_pooling
#         self.sequence_pooling = sequence_pooling
#         self.temporal_mode    = temporal_mode
#         self.temporal_frames  = temporal_on_frames
#         self.temporal_seqs    = temporal_on_sequences
#         self.frame_pe_scale   = frame_temporal_scale
#         self.seq_pe_scale     = seq_temporal_scale

#         self.vision_proj = nn.Sequential(
#             nn.Linear(self.vit_hidden, self.vit_hidden), nn.GELU(),
#             nn.Linear(self.vit_hidden, embed_dim),
#         )
#         self.text_proj = nn.Sequential(
#             nn.Linear(self.text_hidden, self.text_hidden), nn.GELU(),
#             nn.Linear(self.text_hidden, embed_dim),
#         )
#         self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

#         self.enable_generation   = enable_generation
#         self.gen_use_study_token = gen_use_study_token
#         self.decoder_model_name  = decoder_model_name
#         self.report_decoder      = None
#         self.decoder_hidden      = None
#         self.gen_visual_proj     = None

#         if enable_generation:
#             dcfg = AutoConfig.from_pretrained(decoder_model_name)
#             setattr(dcfg, "add_cross_attention", True)
#             setattr(dcfg, "is_decoder",          True)
#             self.report_decoder = AutoModelForCausalLM.from_pretrained(
#                 decoder_model_name, config=dcfg, ignore_mismatched_sizes=True
#             )
#             for attr in ("n_embd", "hidden_size", "d_model"):
#                 if hasattr(self.report_decoder.config, attr):
#                     self.decoder_hidden = int(getattr(self.report_decoder.config, attr))
#                     break
#             if self.decoder_hidden is None:
#                 raise RuntimeError("Cannot infer decoder hidden size from config.")
#             self.gen_visual_proj = nn.Sequential(
#                 nn.Linear(self.vit_hidden, self.vit_hidden), nn.GELU(),
#                 nn.Linear(self.vit_hidden, self.decoder_hidden),
#             )

#     # ── Internal helpers ──────────────────────────────────────────────────────

#     def _add_pe(
#         self, x: torch.Tensor, pos: torch.Tensor, scale: float
#     ) -> torch.Tensor:
#         if self.temporal_mode == "none" or scale == 0.0:
#             return x
#         return x + scale * build_sinusoidal_pe(pos, x.size(-1), x.device, x.dtype)

#     @torch.no_grad()
#     def _vit_chunked(self, pv: torch.Tensor, chunk: int) -> torch.Tensor:
#         if pv.size(0) == 0:
#             return torch.zeros(0, self.vit_hidden, device=pv.device, dtype=pv.dtype)
#         parts = []
#         for i in range(0, pv.size(0), chunk):
#             out = self.vit(pixel_values=pv[i: i + chunk])
#             parts.append(out.last_hidden_state[:, 0, :])
#         return torch.cat(parts, 0)

#     # ── Visual encoder ────────────────────────────────────────────────────────

#     def encode_visual_batch(
#         self,
#         batch_pv:   List[List[torch.Tensor]],
#         batch_pos:  List[List[torch.Tensor]],
#         device:     torch.device,
#         chunk:      int = 16,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         B = len(batch_pv)
#         if B == 0:
#             z = torch.zeros
#             return (
#                 z(0, self.vit_hidden, device=device),
#                 z(0, 1, self.vit_hidden, device=device),
#                 z(0, 1, device=device, dtype=torch.long),
#             )

#         per_seq_lens: List[List[int]] = [[] for _ in range(B)]
#         flat_pv, flat_pos = [], []
#         for s, seqs in enumerate(batch_pv):
#             for q, seq_pv in enumerate(seqs):
#                 T = int(seq_pv.size(0))
#                 per_seq_lens[s].append(T)
#                 if T:
#                     flat_pv.append(seq_pv)
#                     flat_pos.append(batch_pos[s][q])

#         big_pv = (
#             torch.cat(flat_pv, 0).to(device, non_blocking=True)
#             if flat_pv
#             else torch.zeros(0, 3, 1, 1, device=device)
#         )
#         all_emb = self._vit_chunked(big_pv, chunk)

#         per_study_seqs: List[torch.Tensor] = []
#         cursor = flat_idx = 0
#         for s in range(B):
#             seq_feats = []
#             for T in per_seq_lens[s]:
#                 if T == 0:
#                     continue
#                 emb = all_emb[cursor: cursor + T]
#                 cursor += T
#                 if self.temporal_frames and self.temporal_mode != "none":
#                     emb = self._add_pe(
#                         emb,
#                         flat_pos[flat_idx].to(device, dtype=torch.long),
#                         self.frame_pe_scale,
#                     )
#                 seq_feats.append(pool_stack(emb, self.frame_pooling))
#                 flat_idx += 1
#             ss = (
#                 torch.stack(seq_feats)
#                 if seq_feats
#                 else torch.zeros(1, self.vit_hidden, device=device)
#             )
#             per_study_seqs.append(ss)

#         study_vecs = []
#         for s in range(B):
#             ss = per_study_seqs[s]
#             if self.temporal_seqs and self.temporal_mode != "none":
#                 pos = torch.arange(ss.size(0), device=device, dtype=torch.long)
#                 ss  = self._add_pe(ss, pos, self.seq_pe_scale)
#                 per_study_seqs[s] = ss
#             study_vecs.append(pool_stack(ss, self.sequence_pooling))

#         sv = torch.stack(study_vecs)
#         max_seq = max(x.size(0) for x in per_study_seqs)
#         prepend = 1 if self.gen_use_study_token else 0

#         tok  = torch.zeros(B, max_seq + prepend, self.vit_hidden,
#                            device=device, dtype=sv.dtype)
#         mask = torch.zeros(B, max_seq + prepend, device=device, dtype=torch.long)
#         for i in range(B):
#             off = 0
#             if self.gen_use_study_token:
#                 tok[i, 0] = sv[i]; mask[i, 0] = 1; off = 1
#             ss = per_study_seqs[i]
#             tok [i, off: off + ss.size(0)] = ss
#             mask[i, off: off + ss.size(0)] = 1

#         return sv, tok, mask

#     # ── Prompt-conditioned generation ─────────────────────────────────────────

#     @torch.no_grad()
#     def generate_from_prompt(
#         self,
#         visual_tokens:    torch.Tensor,   # (1, S, vit_hidden)
#         visual_attn_mask: torch.Tensor,   # (1, S)
#         gen_tokenizer,
#         prompt_text:      str,
#         device:           torch.device,
#         max_new_tokens:   int   = 256,
#         do_sample:        bool  = False,
#         top_p:            float = 1.0,
#         temperature:      float = 1.0,
#     ) -> str:
#         if not self.enable_generation or self.report_decoder is None:
#             raise RuntimeError("Generation head not enabled in this model.")

#         # Project visual tokens to decoder hidden size
#         proj = self.gen_visual_proj(visual_tokens)   # (1, S, decoder_hidden)

#         enc  = gen_tokenizer(prompt_text, return_tensors="pt").to(device)
#         ids  = enc["input_ids"]
#         attn = enc["attention_mask"]
#         eos  = gen_tokenizer.eos_token_id

#         for _ in range(max_new_tokens):
#             out = self.report_decoder(
#                 input_ids              = ids,
#                 attention_mask         = attn,
#                 encoder_hidden_states  = proj,
#                 encoder_attention_mask = visual_attn_mask,
#                 return_dict            = True,
#             )
#             logits = out.logits[:, -1, :]
#             if temperature and temperature != 1.0:
#                 logits = logits / temperature

#             if do_sample:
#                 probs = torch.softmax(logits, -1)
#                 if top_p < 1.0:
#                     sp, si = torch.sort(probs, descending=True, dim=-1)
#                     cp = torch.cumsum(sp, -1)
#                     m  = cp > top_p
#                     m[:, 1:] = m[:, :-1].clone()
#                     m[:, 0]  = False
#                     sp = sp.masked_fill(m, 0.0)
#                     sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-12)
#                     nt = si.gather(-1, torch.multinomial(sp, 1))
#                 else:
#                     nt = torch.multinomial(probs, 1)
#             else:
#                 nt = torch.argmax(logits, -1, keepdim=True)

#             ids  = torch.cat([ids,  nt], dim=-1)
#             attn = torch.cat(
#                 [attn, torch.ones(1, 1, device=device, dtype=attn.dtype)], dim=-1
#             )
#             if eos is not None and int(nt.item()) == eos:
#                 break

#         decoded = gen_tokenizer.decode(ids[0], skip_special_tokens=True)
#         # Strip back the prompt prefix so we only return newly generated text
#         if decoded.startswith(prompt_text):
#             decoded = decoded[len(prompt_text):].strip()
#         return decoded


# # ─────────────────────────────────────────────────────────────
# # Checkpoint loader
# # ─────────────────────────────────────────────────────────────
# def load_model(args, device: torch.device) -> PooledCLIP:
#     section("Loading model architecture")
#     model = PooledCLIP(
#         vit_name              = args.vit_name,
#         text_model_name       = args.bert_name,
#         embed_dim             = args.embed_dim,
#         frame_pooling         = args.pooling,
#         sequence_pooling      = args.pooling,
#         temporal_mode         = "sinusoidal",
#         temporal_on_frames    = True,
#         temporal_on_sequences = True,   # --enable_sequence_temporal was set
#         frame_temporal_scale  = args.frame_temporal_scale,
#         seq_temporal_scale    = args.seq_temporal_scale,
#         enable_generation     = True,
#         decoder_model_name    = args.decoder_model_name,
#         gen_use_study_token   = True,
#     )

#     ckpt_path = Path(args.checkpoint)
#     if not ckpt_path.exists():
#         err(f"Checkpoint not found: {ckpt_path}")
#         sys.exit(1)

#     info(f"Loading weights → {ckpt_path}")
#     ckpt  = torch.load(ckpt_path, map_location="cpu")
#     state = ckpt.get("model_state", ckpt)
#     miss, unex = model.load_state_dict(state, strict=False)
#     if miss:
#         warn(f"Missing keys  ({len(miss)}): {miss[:4]}{'…' if len(miss) > 4 else ''}")
#     if unex:
#         warn(f"Unexpected keys ({len(unex)}): {unex[:4]}{'…' if len(unex) > 4 else ''}")

#     epoch = ckpt.get("epoch", "?")
#     step  = ckpt.get("step",  "?")
#     success(f"Checkpoint loaded  |  epoch={epoch}  step={step}")
#     model.eval().to(device)
#     return model


# # ─────────────────────────────────────────────────────────────
# # Holdout study helpers
# # ─────────────────────────────────────────────────────────────
# def load_holdout_studies(
#     meta_csv:  Path,
#     anon_col:  str,
#     sop_col:   str,
#     base_dir:  Path,
# ) -> List[Dict]:
#     df = pd.read_csv(meta_csv)
#     studies = []
#     for _, row in df.iterrows():
#         acc = str(row.get(anon_col, "")).strip()
#         if not acc:
#             continue
#         sops     = parse_sop_instance_uids(row.get(sop_col, ""))
#         seq_info = [
#             {"sop": s, "n_frames": len(find_frame_files(base_dir, acc, s))}
#             for s in sops
#         ]
#         studies.append({"acc": acc, "sop_uids": sops, "seq_info": seq_info})
#     return studies


# def display_study_table(studies: List[Dict]) -> None:
#     hdr = f"  {'#':>3}  {'Accession':<22}  {'Seqs':>5}  {'Frames':>8}"
#     print(c(hdr, BOLD))
#     print(c("  " + "─" * (len(hdr) - 2), DIM))
#     for i, s in enumerate(studies):
#         frames = sum(si["n_frames"] for si in s["seq_info"])
#         row    = f"  {i+1:>3}  {s['acc']:<22}  {len(s['sop_uids']):>5}  {frames:>8}"
#         print(c(row, CYAN if i % 2 == 0 else ""))
#     print()


# def pick_studies(studies: List[Dict]) -> List[Dict]:
#     display_study_table(studies)
#     while True:
#         raw = prompt(
#             "Select study number(s)  (e.g. 1  or  1,3  or  all)"
#         ).strip().lower()
#         if not raw:
#             continue
#         if raw == "all":
#             return studies
#         parts = re.split(r"[\s,]+", raw)
#         sel, ok = [], True
#         for p in parts:
#             if not p.isdigit():
#                 err(f"'{p}' is not a number.")
#                 ok = False
#                 break
#             idx = int(p) - 1
#             if not (0 <= idx < len(studies)):
#                 err(f"Index {int(p)} out of range (1–{len(studies)}).")
#                 ok = False
#                 break
#             sel.append(studies[idx])
#         if ok and sel:
#             return sel


# def pick_sequences(study: Dict) -> Dict:
#     si = study["seq_info"]
#     if len(si) <= 1:
#         return study

#     print(f"\n  Study {c(study['acc'], BOLD)} has {len(si)} sequences:\n")
#     print(c(f"    {'#':>3}  {'SOP (truncated)':<44}  {'Frames':>6}", BOLD))
#     print(c("    " + "─" * 56, DIM))
#     for i, s in enumerate(si):
#         short = s["sop"][:42] + ".." if len(s["sop"]) > 42 else s["sop"]
#         print(f"    {i+1:>3}  {short:<44}  {s['n_frames']:>6}")

#     raw = prompt(
#         "Use which sequences? (e.g. 1,2  or  all — default: all)"
#     ).strip().lower()
#     if not raw or raw == "all":
#         return study
#     keep = {
#         int(p) - 1
#         for p in re.split(r"[\s,]+", raw)
#         if p.isdigit() and 1 <= int(p) <= len(si)
#     }
#     if not keep:
#         warn("Invalid selection — using all sequences.")
#         return study
#     return {
#         "acc":      study["acc"],
#         "sop_uids": [study["sop_uids"][i] for i in sorted(keep)],
#         "seq_info": [si[i]               for i in sorted(keep)],
#     }


# # ─────────────────────────────────────────────────────────────
# # Visual encoding  (result is cached per-study for Q&A)
# # ─────────────────────────────────────────────────────────────
# def encode_study(
#     model:     PooledCLIP,
#     study:     Dict,
#     base_dir:  Path,
#     processor,
#     device:    torch.device,
#     args,
# ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
#     """
#     Returns (visual_tokens, visual_mask) shaped (1,S,D) and (1,S) on device,
#     or None if no frames could be loaded.
#     """
#     seq_pvs, seq_pos = [], []
#     for sop in study["sop_uids"]:
#         files = find_frame_files(base_dir, study["acc"], sop)
#         if args.max_frames_per_sequence and len(files) > args.max_frames_per_sequence:
#             idxs  = torch.linspace(
#                 0, len(files) - 1, args.max_frames_per_sequence
#             ).long().tolist()
#             files = [files[i] for i in idxs]
#         if not files:
#             continue
#         pv, vpos = preprocess_frames(files, processor, args.vit_image_size)
#         if pv is None or pv.size(0) == 0:
#             continue
#         seq_pvs.append(pv)
#         seq_pos.append(torch.tensor(vpos, dtype=torch.long))

#     if not seq_pvs:
#         return None

#     with torch.no_grad():
#         _, vtok, vmask = model.encode_visual_batch(
#             [seq_pvs], [seq_pos], device, chunk=args.frame_chunk_size
#         )
#     return vtok, vmask


# # ─────────────────────────────────────────────────────────────
# # Report generation & Q&A wrappers
# # ─────────────────────────────────────────────────────────────
# def generate_report(
#     model:  PooledCLIP,
#     vtok:   torch.Tensor,
#     vmask:  torch.Tensor,
#     gen_tok,
#     device: torch.device,
#     args,
# ) -> str:
#     """Generate an initial free-form radiology report from visual tokens."""
#     bos = gen_tok.bos_token or gen_tok.eos_token or ""
#     return model.generate_from_prompt(
#         visual_tokens    = vtok,
#         visual_attn_mask = vmask,
#         gen_tokenizer    = gen_tok,
#         prompt_text      = bos,
#         device           = device,
#         max_new_tokens   = args.max_new_tokens,
#         do_sample        = args.do_sample,
#         top_p            = args.top_p,
#         temperature      = args.temperature,
#     )


# def answer_question(
#     model:    PooledCLIP,
#     vtok:     torch.Tensor,
#     vmask:    torch.Tensor,
#     report:   str,
#     question: str,
#     gen_tok,
#     device:   torch.device,
#     args,
# ) -> str:
#     """
#     Construct a grounded prompt:
#         Report: <generated report>
#         Q: <user question>
#         A:
#     Feed it to the decoder together with the visual cross-attention tokens.
#     """
#     max_rpt_chars = 800
#     trimmed = (report[:max_rpt_chars] + "…") if len(report) > max_rpt_chars else report
#     prompt_text = (
#         f"Report: {trimmed.strip()}\n"
#         f"Q: {question.strip()}\n"
#         f"A:"
#     )
#     answer = model.generate_from_prompt(
#         visual_tokens    = vtok,
#         visual_attn_mask = vmask,
#         gen_tokenizer    = gen_tok,
#         prompt_text      = prompt_text,
#         device           = device,
#         max_new_tokens   = args.qa_max_new_tokens,
#         do_sample        = args.do_sample,
#         top_p            = args.top_p,
#         temperature      = args.temperature,
#     )
#     return answer.strip()


# # ─────────────────────────────────────────────────────────────
# # Per-study interactive session
# # ─────────────────────────────────────────────────────────────
# def _print_report_box(title: str, text: str) -> None:
#     bar = "═" * (74 - len(title) - 2)
#     print(c(f"  ╔═ {title} {bar}", CYAN))
#     for line in textwrap.wrap(text.strip() or "(empty)", width=72):
#         print(c("  ║  ", CYAN) + line)
#     print(c("  ╚" + "═" * 74, CYAN))
#     print()


# def run_study_session(
#     model:     PooledCLIP,
#     study:     Dict,
#     base_dir:  Path,
#     processor,
#     gen_tok,
#     device:    torch.device,
#     args,
# ) -> None:
#     acc = study["acc"]
#     section(f"Study: {acc}")

#     # ── Encode visual tokens once; cache for the whole session ────────────────
#     info("Encoding visual sequences …")
#     result = encode_study(model, study, base_dir, processor, device, args)
#     if result is None:
#         warn(f"No valid frames found for {acc} — skipping.")
#         return
#     vtok, vmask = result
#     n_seqs = sum(1 for s in study["seq_info"] if s["n_frames"] > 0)
#     success(
#         f"Encoded {n_seqs} sequence(s)  →  "
#         f"visual token shape {tuple(vtok.shape)}"
#     )

#     # ── Generate initial report ───────────────────────────────────────────────
#     info("Generating report …")
#     report = generate_report(model, vtok, vmask, gen_tok, device, args)
#     _print_report_box("Generated Report", report)

#     # ── Q&A loop ──────────────────────────────────────────────────────────────
#     section("Q&A  —  your question → decoder answers using visual + report context")
#     print(c("  Commands:", BOLD))
#     print(c("    show   — re-print the full report", DIM))
#     print(c("    regen  — regenerate the report from scratch", DIM))
#     print(c("    back   — return to study selection", DIM))
#     print(c("    quit   — exit the tool", DIM))
#     print()

#     while True:
#         try:
#             user_q = prompt("You").strip()
#         except (EOFError, KeyboardInterrupt):
#             print()
#             break

#         if not user_q:
#             continue

#         cmd = user_q.lower()

#         if cmd in ("back", "b"):
#             break

#         if cmd in ("quit", "exit", "q"):
#             banner("Session ended  ◈  Thank you for using AngioVision")
#             sys.exit(0)

#         if cmd == "show":
#             _print_report_box("Generated Report", report)
#             continue

#         if cmd == "regen":
#             info("Regenerating report …")
#             report = generate_report(model, vtok, vmask, gen_tok, device, args)
#             _print_report_box("Regenerated Report", report)
#             continue

#         # Regular question
#         info("Generating answer …")
#         answer = answer_question(
#             model, vtok, vmask, report, user_q, gen_tok, device, args
#         )
#         print()
#         print(c("  Model:", BOLD + GREEN))
#         for line in textwrap.wrap(answer or "(no answer generated)", width=74):
#             print("  " + line)
#         print()


# # ─────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────
# def build_args() -> argparse.ArgumentParser:
#     ap = argparse.ArgumentParser(
#         description="AngioVision Interactive CLI (fine-tuned model only)",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     # ── Checkpoint ────────────────────────────────────────────────────────────
#     ap.add_argument(
#         "--checkpoint",
#         default=(
#             "/data/Deep_Angiography/AngioVision/fine-tuning/"
#             "checkpoints/500_16_16_32/last.pt"
#         ),
#         help="Path to trained .pt checkpoint",
#     )
#     # ── Architecture (matches training command exactly) ───────────────────────
#     ap.add_argument("--vit_name",  default="google/vit-base-patch16-224-in21k")
#     ap.add_argument("--bert_name", default="UCSD-VA-health/RadBERT-RoBERTa-4m")
#     ap.add_argument("--decoder_model_name", default="gpt2")
#     ap.add_argument("--embed_dim", type=int, default=256)
#     ap.add_argument("--pooling",   default="logsumexp", choices=list(POOL_CHOICES))
#     # ── Holdout data ──────────────────────────────────────────────────────────
#     ap.add_argument(
#         "--holdout_meta_csv",
#         default=(
#             "/data/Deep_Angiography/Validation_Data/"
#             "Validation_Data_2026_03_23/consolidated_metadata_ALL_Sequences.csv"
#         ),
#     )
#     ap.add_argument(
#         "--holdout_base_frames_dir",
#         default=(
#             "/data/Deep_Angiography/Validation_Data/"
#             "Validation_Data_2026_03_23/DICOM_Sequence_Processed"
#         ),
#     )
#     ap.add_argument("--anon_col", default="Anon Acc #")
#     ap.add_argument("--sop_col",  default="SOPInstanceUIDs")
#     # ── Inference knobs ───────────────────────────────────────────────────────
#     ap.add_argument("--device",                  default=None,
#                     help="cuda or cpu (auto-detect if omitted)")
#     ap.add_argument("--frame_chunk_size",        type=int,   default=32)
#     ap.add_argument("--max_new_tokens",          type=int,   default=256,
#                     help="Max tokens for the initial report")
#     ap.add_argument("--qa_max_new_tokens",       type=int,   default=128,
#                     help="Max tokens for each Q&A answer")
#     ap.add_argument("--max_frames_per_sequence", type=int,   default=32)
#     ap.add_argument("--vit_image_size",          type=int,   default=None)
#     ap.add_argument("--frame_temporal_scale",    type=float, default=0.75)
#     ap.add_argument("--seq_temporal_scale",      type=float, default=0.5)
#     ap.add_argument("--do_sample",               action="store_true")
#     ap.add_argument("--top_p",                   type=float, default=1.0)
#     ap.add_argument("--temperature",             type=float, default=1.0)
#     return ap


# def main() -> None:
#     args = build_args().parse_args()
#     banner("AngioVision Interactive  ◈  Powered by your fine-tuned model")

#     device = torch.device(
#         args.device if args.device
#         else ("cuda" if torch.cuda.is_available() else "cpu")
#     )
#     info(f"Device: {device}")

#     # ── Load model ────────────────────────────────────────────────────────────
#     model = load_model(args, device)

#     # ── Tokenisers & processor ────────────────────────────────────────────────
#     section("Loading tokenisers & image processor")
#     processor = get_vit_processor(args.vit_name)
#     gen_tok   = AutoTokenizer.from_pretrained(args.decoder_model_name)
#     if gen_tok.pad_token is None:
#         gen_tok.pad_token = gen_tok.eos_token or "<|pad|>"
#     success("Ready")

#     # ── Holdout metadata ──────────────────────────────────────────────────────
#     section("Loading holdout studies")
#     meta_csv = Path(args.holdout_meta_csv)
#     base_dir = Path(args.holdout_base_frames_dir)
#     if not meta_csv.exists():
#         err(f"Holdout CSV not found: {meta_csv}")
#         sys.exit(1)
#     if not base_dir.exists():
#         warn(f"Frames dir not found: {base_dir}  (frame counts will show 0)")

#     studies = load_holdout_studies(meta_csv, args.anon_col, args.sop_col, base_dir)
#     if not studies:
#         err("No studies found in holdout CSV.")
#         sys.exit(1)
#     success(f"Found {len(studies)} holdout studies")

#     # ── Main interaction loop ─────────────────────────────────────────────────
#     while True:
#         section("Study Selection")
#         selected = pick_studies(studies)

#         for raw_study in selected:
#             refined = pick_sequences(raw_study)
#             run_study_session(
#                 model, refined, base_dir, processor, gen_tok, device, args
#             )

#         again = prompt("Analyse more studies? (y / n)").strip().lower()
#         if again not in ("y", "yes"):
#             break

#     banner("Session ended  ◈  Thank you for using AngioVision")


# if __name__ == "__main__":
#     main()