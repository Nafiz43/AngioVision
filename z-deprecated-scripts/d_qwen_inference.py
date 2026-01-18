"""Example script for querying Qwen2.5-VL about angiography images."""
from __future__ import annotations

from pathlib import Path

from angiovision import GenerationConfig, ask_qwen_about_images


def main() -> None:
    # Update the paths to point at your angiography frames.
    example_images = [
        Path("sample_data/angiogram_frame.png"),
    ]

    question = (
        "What arteries are being catheterized in this image? "
        "Answer with specific vessel names (e.g., RCA, LAD, LCx, femoral, radial) "
        "and provide a short rationale."
    )

    config = GenerationConfig(max_new_tokens=256, temperature=0.0)

    answer = ask_qwen_about_images(
        question=question,
        image_inputs=[str(path) for path in example_images],
        generation_config=config,
    )
    print(answer)


if __name__ == "__main__":
    main()
