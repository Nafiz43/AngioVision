"""Step 02 — zero-shot (NOT fine-tuned) CLIP baseline on the validation mosaics.

Answers the same validation (mosaic, question) pairs as step 01, via
image-text similarity: each question is scored against a YES prompt and a NO
prompt, and the higher-probability side wins. Same predictions schema, same
resume behavior — the statistics step consumes the output like any other
baseline.

Default checkpoint is openai/clip-vit-base-patch32 — deliberately the SAME
default the mosaic fine-tuning track starts from
(fine-tuning/finetune_clip_on_mosaics_*.py), so "naive CLIP vs fine-tuned
CLIP" isolates the effect of fine-tuning alone.

Requires: torch + transformers (imported lazily so steps 01/03 run without them).
"""

from __future__ import annotations

import os

from . import qa_runner


def load_clip(model_name: str, device: str = ""):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"loading {model_name} on {dev}")
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, dev


def make_answer_fn(model, processor, dev):
    import torch
    from PIL import Image

    # Mosaics repeat across questions — cache the last decoded image.
    cache = {"path": None, "image": None}

    def answer_fn(mosaic_path: str, question: str):
        if cache["path"] != mosaic_path:
            cache["path"] = mosaic_path
            cache["image"] = Image.open(mosaic_path).convert("RGB")

        # Same "{question} {option}" prompt format as the historical CLIP
        # extractor (frame-processing/02_extract_labels_from_mosaics_clip.py).
        prompts = [f"{question} Yes.", f"{question} No."]
        inputs = processor(text=prompts, images=cache["image"],
                           return_tensors="pt", padding=True,
                           truncation=True).to(dev)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.squeeze(0)
            probs = torch.softmax(logits, dim=0).tolist()

        pred = "YES" if probs[0] >= probs[1] else "NO"
        return pred, f"P(yes)={probs[0]:.4f} P(no)={probs[1]:.4f}"

    return answer_fn


def run(cfg, run_dir) -> dict:
    val_df, seq_index = qa_runner.load_inputs(cfg)
    metrics_csv = os.path.join(cfg.baselines_dir, "baseline_metrics.csv")

    models = [m.strip() for m in cfg.clip_models.split(",") if m.strip()]
    if not models:
        raise ValueError("cfg.clip_models is empty — nothing to benchmark")

    summaries = []
    for model_name in models:
        model, processor, dev = load_clip(model_name, cfg.clip_device)
        answer_fn = make_answer_fn(model, processor, dev)
        summaries.append(qa_runner.run_qa_benchmark(
            f"zeroshot_{model_name}", answer_fn, val_df, seq_index, cfg,
            metrics_csv))
        del model  # free GPU memory before the next checkpoint

    return {"models": summaries, "metrics_csv": metrics_csv}
