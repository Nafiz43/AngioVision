"""Step 01 — run local Ollama VLM baselines on the validation mosaics.

For each configured Ollama model tag: ask every (mosaic, question) pair from
the validation CSV, normalize the answer to YES/NO, append predictions
row-by-row (resumable via skip_existing), and append one metrics row per
completed model run. The loop itself lives in qa_runner (shared with step 02).

Ported from frame-processing/02_extract_labels_from_mosaics.py (the primary
Ollama VLM extractor — /api/generate with an `images` payload, temperature 0).

Outputs, under cfg.baselines_dir (stable across runs):
    <model>_predictions.csv     one row per (SOP, question)
    baseline_metrics.csv        one row per model run
    <model>_errors.csv          unmatched sequences + failed calls
"""

from __future__ import annotations

import os

import requests

from . import common, qa_runner


def build_prompt(question: str) -> str:
    return (
        "You are analyzing a medical angiography mosaic image.\n"
        "Answer the question using only visible image evidence.\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Respond with exactly one word: YES or NO\n"
        "- Do not explain your answer\n"
        "- Do not add punctuation or extra words\n"
        "- If the finding is not clearly visible, answer NO\n"
        "- If the image is ambiguous, uncertain, low-quality, incomplete, "
        "or cannot confirm the finding, answer NO\n"
        "- Only answer YES when the finding is clearly supported by the image\n"
        "- Output must be exactly YES or NO\n"
    )


def call_ollama_vlm(model: str, image_path: str, question: str,
                    ollama_url: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": build_prompt(question),
        "images": [common.encode_image_base64(image_path)],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def run(cfg, run_dir) -> dict:
    val_df, seq_index = qa_runner.load_inputs(cfg)
    metrics_csv = os.path.join(cfg.baselines_dir, "baseline_metrics.csv")

    models = [m.strip() for m in cfg.vlm_models.split(",") if m.strip()]
    if not models:
        raise ValueError("cfg.vlm_models is empty — nothing to benchmark")

    summaries = []
    for model in models:
        def answer_fn(mosaic_path, question, _model=model):
            raw = call_ollama_vlm(_model, mosaic_path, question,
                                  cfg.ollama_url, cfg.ollama_timeout_s)
            return common.normalize_llm_answer(raw), raw

        summaries.append(qa_runner.run_qa_benchmark(
            model, answer_fn, val_df, seq_index, cfg, metrics_csv))

    return {"models": summaries, "metrics_csv": metrics_csv}
