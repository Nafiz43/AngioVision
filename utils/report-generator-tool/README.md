# report-generator-tool

Interactive CLI for AngioVision report generation and Q&A, powered entirely
by your fine-tuned PooledCLIP checkpoint — no external API. This is the
modularized version of the former `utils/23_interactive_angio.py` (identical
behavior and CLI flags).

## Workflow

1. Load the trained PooledCLIP checkpoint (arch reconstructed from the
   checkpoint's `model_config` when present, else from CLI flags).
2. List holdout studies from the metadata CSV; pick studies + sequences.
3. Encode the selected sequences once → visual tokens cached per session.
4. Generate the initial free-form report via the decoder.
5. Q&A loop: each question is wrapped as `Report: … / Q: … / A:` and fed to
   the decoder with the SAME visual cross-attention tokens, so answers are
   grounded in both the images and the prior report.
   In-session commands: `show`, `regen`, `back`, `quit`.

## Layout

| Module | Contents |
|--------|----------|
| `run_tool.py` | Entry point: CLI args, model/tokenizer/data startup, main loop |
| `rgt/ui.py` | Terminal colors, banner/section/info/warn/err, prompts, report box |
| `rgt/data.py` | SOP-UID parsing, frame discovery, ViT preprocessing, holdout loading |
| `rgt/model.py` | `PooledCLIP` (mirrors the training script), sinusoidal PE, pooling, checkpoint loader |
| `rgt/inference.py` | `encode_study`, `generate_report`, `answer_question` |
| `rgt/session.py` | Study/sequence pickers, per-study interactive session |

## Usage

```bash
python run_tool.py \
    --checkpoint /data/Deep_Angiography/AngioVision/fine-tuning/checkpoints/gen/500_16_16_32/last.pt \
    [--decoder_model_name microsoft/biogpt | gpt2] \
    [--vit_name google/vit-base-patch16-224-in21k] \
    [--embed_dim 256] \
    [--device cuda]
```

Run `python run_tool.py --help` for all flags (holdout CSV/frames dir,
sampling knobs, frame limits, temporal scales, …). Defaults point at the
lab-server checkpoint and validation dataset, same as the original script.

## Dependencies

`torch transformers pandas pillow`
