"""Inference wrappers: study encoding (cached per session), report
generation, and grounded Q&A."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from rgt.data import find_frame_files, preprocess_frames
from rgt.model import PooledCLIP


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
            idxs = torch.linspace(
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
