"""Command-line interface for querying Qwen2.5-VL with angiography images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .inference import (
    DEFAULT_MODEL_ID,
    GenerationConfig,
    QwenVisualLanguageClient,
)


DEFAULT_GENERATION = GenerationConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the Qwen2.5-VL vision-language model questions about angiography "
            "images."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Path(s) to image files (PNG, JPG, etc.) to include in the prompt.",
    )
    parser.add_argument(
        "-q",
        "--question",
        required=True,
        help="Question to ask about the supplied images.",
    )
    parser.add_argument(
        "-s",
        "--system-prompt",
        default="You are a concise interventional radiology assistant.",
        help="System prompt to steer the assistant's behaviour.",
    )
    parser.add_argument(
        "-c",
        "--context",
        nargs="*",
        help="Optional extra text snippets that provide context for the question.",
    )
    parser.add_argument(
        "-m",
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier to load.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_GENERATION.max_new_tokens,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_GENERATION.temperature,
        help="Sampling temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty applied to repeated tokens during generation.",
    )
    parser.add_argument(
        "-o",
        "--output-json",
        type=Path,
        help="If provided, save the question and answer as JSON to this path.",
    )
    return parser


def run_cli(args: Optional[List[str]] = None) -> str:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    generation_config = GenerationConfig(
        max_new_tokens=parsed.max_new_tokens,
        temperature=parsed.temperature,
        top_p=parsed.top_p,
        top_k=parsed.top_k,
        repetition_penalty=parsed.repetition_penalty,
    )

    client = QwenVisualLanguageClient(
        model_id=parsed.model_id,
        generation_config=generation_config,
    )

    answer = client.ask(
        question=parsed.question,
        image_inputs=[str(path) for path in parsed.images],
        system_prompt=parsed.system_prompt,
        context=parsed.context,
    )

    if parsed.output_json:
        parsed.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": parsed.model_id,
            "question": parsed.question,
            "images": [str(path) for path in parsed.images],
            "system_prompt": parsed.system_prompt,
            "context": parsed.context or [],
            "answer": answer,
        }
        parsed.output_json.write_text(json.dumps(payload, indent=2))

    return answer


def main() -> None:
    answer = run_cli()
    print(answer)


if __name__ == "__main__":
    main()
