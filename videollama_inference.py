#!/usr/bin/env python3
"""Thin CLI wrapper to run Video-LLaMA inference from an external repository."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional


def _resolve_repo_path(repo_arg: Optional[str]) -> Path:
    """Resolve the path to the external Video-LLaMA repository."""
    candidate = repo_arg or os.environ.get("VIDEO_LLAMA_HOME")
    if not candidate:
        raise SystemExit(
            "A path to the Video-LLaMA repository is required. "
            "Pass --videollama-repo or set the VIDEO_LLAMA_HOME environment variable."
        )

    repo_path = Path(candidate).expanduser().resolve()
    if not repo_path.exists():
        raise SystemExit(f"Video-LLaMA repository not found at: {repo_path}")

    if not (repo_path / "video_llama").exists():
        raise SystemExit(
            "The provided path does not look like the Video-LLaMA repository. "
            "Expected to find a 'video_llama' package under the repo root."
        )
    return repo_path


def _prepare_pythonpath(repo_path: Path) -> None:
    """Ensure the Video-LLaMA sources are importable via sys.path and PYTHONPATH."""
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    existing = os.environ.get("PYTHONPATH")
    paths: List[str] = [] if not existing else existing.split(os.pathsep)
    if repo_str not in paths:
        paths.insert(0, repo_str)
        os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def _resolve_cfg_path(cfg_path: str, repo_path: Path) -> Path:
    path = Path(cfg_path)
    if not path.is_absolute():
        path = repo_path / path
    if not path.exists():
        raise SystemExit(f"Video-LLaMA config file not found: {path}")
    return path


def _comma_separated(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return []
    parsed: List[str] = []
    for value in values:
        if value:
            parsed.extend(part for part in value.split(",") if part)
    return parsed


def _initialise_videollama(args: argparse.Namespace):
    """Instantiate the Video-LLaMA chat interface from the external repo."""
    import random

    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    import decord
    from video_llama.common.config import Config
    from video_llama.common.dist_utils import get_rank
    from video_llama.common.registry import registry
    from video_llama.conversation.conversation_video import (
        Chat,
        conv_llava_llama_2,
        default_conversation,
    )

    decord.bridge.set_bridge("torch")

    cfg = Config(
        SimpleNamespace(
            cfg_path=str(args.cfg_path),
            options=args.cfg_options,
        )
    )

    seed = getattr(cfg.run_cfg, "seed", 42) + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    model_config = cfg.model_cfg
    if hasattr(model_config, "device_8bit"):
        model_config.device_8bit = args.gpu_id

    model_cls = registry.get_model_class(model_config.arch)
    if model_cls is None:
        raise SystemExit(f"Unknown Video-LLaMA model architecture: {model_config.arch}")

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[videollama] CUDA not available - running on CPU. This will be slow.")

    model = model_cls.from_config(model_config).to(device)
    model.eval()

    datasets_cfg = cfg.datasets_cfg
    if not hasattr(datasets_cfg, "webvid"):
        raise SystemExit(
            "The loaded config does not define a 'webvid' dataset block, which is required "
            "to instantiate the vision processor."
        )
    vis_processor_cfg = datasets_cfg.webvid.vis_processor.train
    processor_cls = registry.get_processor_class(vis_processor_cfg.name)
    if processor_cls is None:
        raise SystemExit(
            f"Video-LLaMA vision processor '{vis_processor_cfg.name}' is not registered."
        )
    vis_processor = processor_cls.from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device=device)
    if args.conversation_mode == "vicuna":
        conv = default_conversation.copy()
    else:
        conv = conv_llava_llama_2.copy()
    return chat, conv


def run_inference(args: argparse.Namespace) -> dict:
    repo_path = _resolve_repo_path(args.videollama_repo)
    _prepare_pythonpath(repo_path)
    args.cfg_path = _resolve_cfg_path(args.cfg_path, repo_path)
    args.cfg_options = _comma_separated(args.cfg_options)

    chat, conversation = _initialise_videollama(args)

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")

    img_list: List[object] = []
    receipt = chat.upload_video_without_audio(str(video_path), conversation, img_list)
    if args.verbose:
        print(f"[videollama] Upload status: {receipt}")

    if args.system_prompt:
        conversation.system = args.system_prompt

    if args.context:
        chat.ask(args.context, conversation)
    chat.ask(args.question, conversation)
    answer_text, _ = chat.answer(
        conv=conversation,
        img_list=img_list,
        num_beams=args.num_beams,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        max_length=args.max_length,
    )

    result = {
        "question": args.question,
        "answer": answer_text,
        "video": str(video_path),
        "conversation": conversation.to_gradio_chatbot(),
    }
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        if args.verbose:
            print(f"[videollama] Wrote output to {output_path}")

    print(answer_text)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Video-LLaMA video question answering pass by pointing to an external "
            "clone of the official repository."
        )
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("question", help="Question to ask about the video.")
    parser.add_argument(
        "--videollama-repo",
        help=(
            "Filesystem path to the Video-LLaMA Git repository. Defaults to the value of "
            "the VIDEO_LLAMA_HOME environment variable if unset."
        ),
    )
    parser.add_argument(
        "--cfg-path",
        default="eval_configs/video_llama_eval_only_vl.yaml",
        help="Relative path to the Video-LLaMA YAML config inside the repo.",
    )
    parser.add_argument(
        "--cfg-option",
        dest="cfg_options",
        action="append",
        default=[],
        help="Override configuration keys (repeatable). Accepts comma separated KEY=VALUE pairs.",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=["vicuna", "llama-2"],
        default="vicuna",
        help="Conversation template to use when formatting prompts.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index to run on (if available).")
    parser.add_argument(
        "--system-prompt",
        help="Optional system prompt to inject before asking the question.",
    )
    parser.add_argument(
        "--context",
        help="Optional additional context to append before the main question.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate for the answer.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2000,
        help="Overall token cap for the conversation buffer.",
    )
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width during decoding.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty applied to repeated tokens during generation.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty applied during beam search decoding.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to store the generated answer as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit additional logging about repository resolution and outputs.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
