"""Step 05 — zero-shot (NOT fine-tuned) SigLIP-family baselines on the validation mosaics.

Same (mosaic, question) task and predictions schema as step 02, but for the
SigLIP tower family — SigLIP, SigLIP2, and MedSigLIP — which the fine-tuning
track uses. Each question is scored against a YES and a NO prompt via
image-text similarity; the higher-similarity side wins.

Why a separate step from s02 (CLIP): SigLIP uses SiglipModel/AutoProcessor and
requires text padding="max_length" (its trained sequence length), and reports
per-pair sigmoid probabilities rather than a softmax over classes. The YES/NO
argmax is identical either way (monotonic in the logit gap), so verdicts are
directly comparable to the CLIP baseline.

These predictions land in baselines_dir like every other baseline, so the
statistics step (04) auto-discovers them with no extra wiring.

Requires: torch + transformers (imported lazily so steps 01/03 run without them).
"""

from __future__ import annotations

import os

from . import qa_runner


def load_siglip(model_name: str, device: str = ""):
    import torch
    from transformers import AutoModel, AutoProcessor

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"loading {model_name} on {dev}")
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    processor = AutoProcessor.from_pretrained(model_name)
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

        prompts = [f"{question} Yes.", f"{question} No."]
        # SigLIP was trained with padding to a fixed length; use max_length.
        inputs = processor(text=prompts, images=cache["image"],
                           return_tensors="pt", padding="max_length",
                           truncation=True).to(dev)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.squeeze(0)
            # SigLIP scores each pair independently (sigmoid), reported for
            # transparency; the YES/NO decision is the higher logit either way.
            probs = torch.sigmoid(logits).tolist()

        pred = "YES" if logits[0] >= logits[1] else "NO"
        return pred, f"sig(yes)={probs[0]:.4f} sig(no)={probs[1]:.4f}"

    return answer_fn


def run(cfg, run_dir) -> dict:
    val_df, seq_index = qa_runner.load_inputs(cfg)
    metrics_csv = os.path.join(cfg.baselines_dir, "baseline_metrics.csv")

    models = [m.strip() for m in cfg.siglip_models.split(",") if m.strip()]
    if not models:
        raise ValueError("cfg.siglip_models is empty — nothing to benchmark")

    summaries = []
    for model_name in models:
        model, processor, dev = load_siglip(model_name, cfg.clip_device)
        answer_fn = make_answer_fn(model, processor, dev)
        summaries.append(qa_runner.run_qa_benchmark(
            f"zeroshot_{model_name}", answer_fn, val_df, seq_index, cfg,
            metrics_csv))
        del model  # free GPU memory before the next checkpoint

    return {"models": summaries, "metrics_csv": metrics_csv}
