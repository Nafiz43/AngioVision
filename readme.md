# AngioVision Qwen2.5-VL Inference Toolkit

AngioVision provides a lightweight Python toolkit for running the
[`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
vision-language model on coronary angiography frames. It packages a
re-usable inference client, a command line interface, and example scripts for
posing questions such as *"What arteries are being catheterized in this image?"*
while providing one or more angiography frames.

## Project layout

```
.
├── readme.md                # Project overview and usage instructions
├── requirements.txt         # Python dependencies
├── qwen_inference.py        # Minimal script-based entry point
├── videollama_inference.py  # Bridge script for external Video-LLaMA repo
└── src
    └── angiovision
        ├── __init__.py
        ├── cli.py           # CLI for asking questions from the terminal
        ├── inference.py     # Core Qwen2.5-VL client implementation
        └── qwen_vl_utils.py # Lightweight helper mirroring Qwen reference code
```

## Installation

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > ⚠️ The Qwen2.5-VL model is large (~14 GB in bf16). Ensure you have a
   > compatible GPU (or sufficient CPU RAM) and the correct version of PyTorch
   > installed for your hardware.

## Obtaining angiography frames

Place your DICOM-derived angiography frames or PNG/JPG exports in an accessible
folder. Update the example paths in `qwen_inference.py` or pass them directly to
the CLI when running the tool.

## Usage

### Python API

```python
from angiovision import GenerationConfig, ask_qwen_about_images

answer = ask_qwen_about_images(
    question="What arteries are being catheterized in this image?",
    image_inputs=["/path/to/angiogram_frame.png"],
    generation_config=GenerationConfig(max_new_tokens=256, temperature=0.0),
)

print(answer)
```

For more control over caching and device placement, instantiate the
`QwenVisualLanguageClient` directly and reuse it across multiple questions.

### Command line interface

The CLI is powered by `src/angiovision/cli.py`. Run it after installation with:

```bash
python -m angiovision.cli \
  /path/to/frame1.png /path/to/frame2.png \
  --question "What arteries are being catheterized in this image?" \
  --output-json outputs/answer.json
```

You can customise decoding behaviour using flags such as `--max-new-tokens`,
`--temperature`, `--top-p`, and `--top-k`. Setting `--temperature 0` will switch
inference to greedy decoding, yielding deterministic answers.

### Example script

The top-level [`qwen_inference.py`](./qwen_inference.py) script offers a minimal
Python entry point that can be adapted for quick experiments. Update the example
image paths in the script and run:

```bash
python qwen_inference.py
```

## Video-LLaMA video question answering (separate repo)

Some angiography studies are captured as videos rather than still frames. The
[`videollama_inference.py`](./videollama_inference.py) helper lets you route such
clips through the official [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
repository while keeping their implementation isolated from this project.

1. **Clone Video-LLaMA somewhere outside this repo.** For example:
   ```bash
   git clone https://github.com/DAMO-NLP-SG/Video-LLaMA.git ../Video-LLaMA
   ```

2. **Install its dependencies and download checkpoints** as described in the
   Video-LLaMA README (this typically requires CUDA, `decord`, and the released
   video-language weights).

3. **Run the bridge script**, pointing it at your clone and angiography video:
   ```bash
   python videollama_inference.py \
     /path/to/angiography_clip.mp4 \
     "What arteries are being catheterized in this study?" \
     --videollama-repo ../Video-LLaMA \
     --system-prompt "You are a concise interventional cardiology assistant." \
     --output outputs/videollama_answer.json
   ```

   If you prefer, set the `VIDEO_LLAMA_HOME` environment variable to avoid
   passing `--videollama-repo` on every call.

The script forwards configuration options to the upstream YAML files (via
`--cfg-path` and repeatable `--cfg-option KEY=VALUE` overrides) and surfaces
common decoding controls such as `--max-new-tokens`, `--temperature`, and
`--num-beams`. Output is printed to stdout and can optionally be saved as a JSON
record via `--output`.

## Development notes

- All package code lives under `src/angiovision`. Adjust `PYTHONPATH` or install
  the package in editable mode (`pip install -e .`) if you plan to extend it.
- The helper `process_vision_info` mirrors the official Qwen utility to avoid
  pulling in extra dependencies from external repositories.
- When running on GPU, the client defaults to `bfloat16` for improved
  performance. On CPU or MPS devices the dtype automatically downgrades to
  `float32`/`float16` as appropriate.

## Troubleshooting

- **CUDA out of memory:** Reduce `--max-new-tokens`, close other GPU processes,
  or offload the model to CPU by setting the `CUDA_VISIBLE_DEVICES` environment
  variable to an empty string.
- **Slow responses on CPU:** Qwen2.5-VL is optimised for GPU inference. Consider
  using a smaller checkpoint or trimming the input resolution of your images.

## License

This repository ships example glue code only. Refer to the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
for license details covering the base model weights.
