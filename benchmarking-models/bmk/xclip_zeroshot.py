"""Zero-shot (NOT fine-tuned) X-CLIP baseline on the validation mosaics.

X-CLIP (microsoft/xclip-base-patch32) is a VIDEO-text model, so unlike the
image towers it has no single-image path. To keep it on the same (mosaic,
question) task as every other baseline we feed the mosaic replicated to the
model's trained clip length (8 frames) as a degenerate "video". This is a
forced but consistent zero-shot probe — noted in the README — so its verdicts
sit in the same predictions schema and get discovered by step 04/06.

Same YES/NO-prompt-similarity scoring as s05: higher logit side wins.
Requires torch + transformers (imported lazily).
"""

from __future__ import annotations

import os

from . import qa_runner

NUM_FRAMES = 8  # xclip-base-patch32 clip length


def load_xclip(model_name: str, device: str = ""):
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
    import numpy as np
    import torch
    from PIL import Image

    # This transformers build mis-routes XCLIPProcessor(videos=...) (pixel_values
    # comes back None); call the image processor + tokenizer separately instead.
    cache = {"path": None, "pixel_values": None}

    def answer_fn(mosaic_path: str, question: str):
        if cache["path"] != mosaic_path:
            cache["path"] = mosaic_path
            img = np.array(Image.open(mosaic_path).convert("RGB"))
            frames = [img] * NUM_FRAMES  # degenerate video: one video, N frames
            cache["pixel_values"] = processor.image_processor(
                images=[frames], return_tensors="pt")["pixel_values"].to(dev)

        prompts = [f"{question} Yes.", f"{question} No."]
        text = processor.tokenizer(prompts, return_tensors="pt",
                                   padding=True, truncation=True).to(dev)
        with torch.no_grad():
            logits = model(pixel_values=cache["pixel_values"],
                           **text).logits_per_video.squeeze(0)
        pred = "YES" if logits[0] >= logits[1] else "NO"
        return pred, f"logit(yes)={logits[0]:.3f} logit(no)={logits[1]:.3f}"

    return answer_fn


def run(cfg, run_dir) -> dict:
    val_df, seq_index = qa_runner.load_inputs(cfg)
    metrics_csv = os.path.join(cfg.baselines_dir, "baseline_metrics.csv")
    model_name = getattr(cfg, "xclip_model", "microsoft/xclip-base-patch32")

    model, processor, dev = load_xclip(model_name, cfg.clip_device)
    answer_fn = make_answer_fn(model, processor, dev)
    summary = qa_runner.run_qa_benchmark(
        f"zeroshot_{model_name}", answer_fn, val_df, seq_index, cfg, metrics_csv)
    return {"model": summary, "metrics_csv": metrics_csv}
