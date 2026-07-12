"""Step 03 — AWS Bedrock VLM baselines on the validation mosaics.

Same validation (mosaic, question) QA as steps 01/02, answered by Bedrock-hosted
Claude models via the Converse API. Same predictions schema and resume
behavior — the statistics step consumes the output like any other baseline.

Ported from bedrock-inference/02_extract_labels_from_mosaics_bedrock.py, keeping
its Converse call (png image bytes + strict prompt, maxTokens 16, temperature 0)
but generalized to a configurable comma-separated list of model ids.

Auth: boto3's default AWS credential chain (env / ~/.aws / instance role) — no
credentials live in this repo. boto3 is imported lazily so the other steps run
without it.
"""

from __future__ import annotations

import os

from . import common, qa_runner

# Reuse the exact prompt the Ollama step uses, so the only variable across
# backends is the model — not the wording.
from .s01_vlm_baselines import build_prompt


def make_answer_fn(client, model_id: str, max_tokens: int):
    def answer_fn(mosaic_path: str, question: str):
        with open(mosaic_path, "rb") as f:
            image_bytes = f.read()
        response = client.converse(
            modelId=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                    {"text": build_prompt(question)},
                ],
            }],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0, "topP": 1.0},
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        raw = " ".join(item["text"] for item in content if "text" in item).strip()
        return common.normalize_llm_answer(raw), (raw or "")

    return answer_fn


def run(cfg, run_dir) -> dict:
    import boto3

    val_df, seq_index = qa_runner.load_inputs(cfg)
    metrics_csv = os.path.join(cfg.baselines_dir, "baseline_metrics.csv")

    models = [m.strip() for m in cfg.bedrock_models.split(",") if m.strip()]
    if not models:
        raise ValueError("cfg.bedrock_models is empty — nothing to benchmark")

    client = boto3.client("bedrock-runtime", region_name=cfg.bedrock_region)

    summaries = []
    for model_id in models:
        answer_fn = make_answer_fn(client, model_id, cfg.bedrock_max_tokens)
        summaries.append(qa_runner.run_qa_benchmark(
            f"bedrock_{model_id}", answer_fn, val_df, seq_index, cfg, metrics_csv))

    return {"models": summaries, "metrics_csv": metrics_csv}
