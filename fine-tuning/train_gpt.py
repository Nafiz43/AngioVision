# test_report_lm.py
"""
Tests the fine-tuned GPT-2 report language model at three levels:
  1. Perplexity on held-out reports
  2. Generation quality (BLEU, ROUGE, sample outputs)
  3. Vocabulary coverage (does it know your domain terms?)

Usage:
    python3 train_gpt.py \
        --model_dir /data/Deep_Angiography/AngioVision/fine-tuning/gpt2 \
        --reports_csv /data/Deep_Angiography/Reports/Report_List_v01_01_augmented.csv \
        --text_col radrpt \
        --holdout_frac 0.1
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge_score not installed. Run: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False
    print("[WARN] nltk not installed. Run: pip install nltk")


# -------------------------------------------------------
# Level 1 — Perplexity
# -------------------------------------------------------
def compute_perplexity(model, tokenizer, texts, device, max_length=512, batch_size=8):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
            batch_texts = [
                f"<|report|> {t.strip()} <|endoftext|>"
                for t in texts[i: i + batch_size]
            ]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )
            # out.loss is mean over non-masked tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += out.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_nll = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = np.exp(avg_nll)
    return perplexity


# -------------------------------------------------------
# Level 2 — Generation
# -------------------------------------------------------
def generate_from_prompt(model, tokenizer, prompt, device, max_new_tokens=200, num_beams=4):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip the prompt from the output
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return generated[len(prompt_decoded):].strip()


def compute_rouge(reference, hypothesis):
    if not HAS_ROUGE:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def compute_bleu(reference, hypothesis):
    if not HAS_BLEU:
        return None
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method1
    return round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie), 4)


# -------------------------------------------------------
# Level 3 — Vocabulary coverage
# -------------------------------------------------------
ANGIO_TERMS = [
    "stenosis", "occlusion", "coronary", "angiography", "LAD", "RCA", "LCX",
    "stent", "catheter", "radial", "femoral", "contrast", "ejection fraction",
    "lesion", "collateral", "thrombosis", "dissection", "bifurcation", "TIMI",
    "percutaneous", "intervention", "balloon", "fluoroscopy",
]

def check_vocab_coverage(tokenizer):
    known, unknown = [], []
    for term in ANGIO_TERMS:
        ids = tokenizer.encode(term, add_special_tokens=False)
        # If the term encodes to a single token it's natively known
        if len(ids) == 1:
            known.append(term)
        else:
            unknown.append((term, len(ids), tokenizer.convert_ids_to_tokens(ids)))

    print(f"\n[Level 3] Vocabulary Coverage")
    print(f"  Single-token terms ({len(known)}/{len(ANGIO_TERMS)}): {known}")
    print(f"  Multi-token terms  ({len(unknown)}/{len(ANGIO_TERMS)}):")
    for term, n, tokens in unknown:
        print(f"    '{term}' → {n} tokens {tokens}")


# -------------------------------------------------------
# Main test runner
# -------------------------------------------------------
def test(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Model dir  : {args.model_dir}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    # Load reports and split off holdout
    df = pd.read_csv(args.reports_csv)
    texts = (
        df[args.text_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )
    random.seed(42)
    random.shuffle(texts)
    n_holdout = max(10, int(len(texts) * args.holdout_frac))
    holdout_texts = texts[:n_holdout]
    print(f"[INFO] Holdout reports : {len(holdout_texts)} / {len(texts)}")

    # --------------------------------------------------
    # Level 1 — Perplexity
    # --------------------------------------------------
    print("\n" + "="*60)
    print("[Level 1] Perplexity on Held-Out Reports")
    print("="*60)
    ppl = compute_perplexity(model, tokenizer, holdout_texts, device, batch_size=args.batch_size)
    print(f"\n  Perplexity : {ppl:.2f}")
    print("  Interpretation:")
    print("    < 20   → Excellent — model knows this domain well")
    print("    20–50  → Good — solid domain adaptation")
    print("    50–100 → Moderate — partial adaptation")
    print("    > 100  → Poor — model barely adapted")

    # --------------------------------------------------
    # Level 2 — Generation quality
    # --------------------------------------------------
    print("\n" + "="*60)
    print("[Level 2] Generation Quality")
    print("="*60)

    test_prompts = [
        "<|report|> TECHNIQUE:",
        "<|report|> FINDINGS:",
        "<|report|> IMPRESSION:",
        "<|report|> TECHNIQUE: Coronary angiography was performed via right radial access. FINDINGS:",
    ]

    rouge_scores = []
    bleu_scores = []

    for i, prompt in enumerate(test_prompts):
        generated = generate_from_prompt(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"\n  Prompt {i+1} : {prompt}")
        print(f"  Output    : {generated[:300]}")

    # ROUGE/BLEU against real holdout reports using FINDINGS: prompt
    print(f"\n  Computing ROUGE/BLEU on {min(50, len(holdout_texts))} holdout reports...")
    sample_refs = holdout_texts[:50]
    for ref in tqdm(sample_refs, desc="ROUGE/BLEU"):
        # Use first ~30 words as prompt, rest as reference
        words = ref.split()
        if len(words) < 20:
            continue
        prompt = "<|report|> " + " ".join(words[:15])
        reference = " ".join(words[15:])
        hypothesis = generate_from_prompt(
            model, tokenizer, prompt, device, max_new_tokens=100, num_beams=2
        )
        r = compute_rouge(reference, hypothesis)
        b = compute_bleu(reference, hypothesis)
        if r:
            rouge_scores.append(r)
        if b is not None:
            bleu_scores.append(b)

    if rouge_scores:
        avg_r1 = np.mean([s["rouge1"] for s in rouge_scores])
        avg_r2 = np.mean([s["rouge2"] for s in rouge_scores])
        avg_rL = np.mean([s["rougeL"] for s in rouge_scores])
        print(f"\n  Avg ROUGE-1 : {avg_r1:.4f}")
        print(f"  Avg ROUGE-2 : {avg_r2:.4f}")
        print(f"  Avg ROUGE-L : {avg_rL:.4f}")
        print("  Interpretation:")
        print("    ROUGE-1 > 0.4  → Good lexical overlap with real reports")
        print("    ROUGE-2 > 0.2  → Good bigram overlap (fluency + terminology)")

    if bleu_scores:
        avg_bleu = np.mean(bleu_scores)
        print(f"\n  Avg BLEU    : {avg_bleu:.4f}")
        print("  Interpretation:")
        print("    BLEU > 0.15  → Reasonable for open-ended medical text generation")

    # --------------------------------------------------
    # Level 3 — Vocabulary coverage
    # --------------------------------------------------
    print("\n" + "="*60)
    check_vocab_coverage(tokenizer)

    print("\n" + "="*60)
    print("[DONE] Testing complete.")
    print("="*60)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="./gpt2-angio-reports")
    ap.add_argument(
        "--reports_csv",
        type=str,
        default="/data/Deep_Angiography/Reports/Report_List_v01_01_augmented.csv",
    )
    ap.add_argument("--text_col", type=str, default="radrpt")
    ap.add_argument("--holdout_frac", type=float, default=0.1,
                    help="Fraction of reports to hold out for evaluation.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--cpu", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    test(args)