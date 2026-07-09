"""
Step 01 — Report augmentation via Ollama (from 17_report_augmentor.py).

For each unique accession, keeps the original (cleaned) report and generates
cfg.n_augmentations conservative rephrasings, tagged in a 'Type' column
(Original / Augmented 1..N).

Resume-safe: output rows are appended incrementally after each report, and
accessions whose full row set already exists in the output CSV are skipped
on re-run. That is why the output lives at the stable path
data_dir/augmented_reports.csv rather than inside a per-run directory.

If Ollama is unavailable or a generation repeatedly fails validation, the
variant falls back to the original text (logged in data_dir/augment.log).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from tdp.common import normalize_text

_VARIATION_STYLES = {
    1: "Make only very small sentence-level wording changes.",
    2: "Slightly reorganize sentence structure while preserving the same medical meaning.",
    3: "Use light paraphrasing, but keep clinical terms unchanged whenever possible.",
    4: "Use minor structural variation and wording changes while preserving all findings.",
}


def _append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def build_prompt(report_text: str, variant_idx: int) -> str:
    variation = _VARIATION_STYLES.get(
        variant_idx, "Make slight structural changes only.")
    return f"""
You are helping augment radiology/interventional procedure reports.

Your task:
Create ONE conservative rephrased version of the report below.

Critical rules:
1. Preserve the exact clinical meaning.
2. Preserve key clinical terms exactly whenever possible.
3. Do NOT change anatomy, pathology, procedure names, vessels, devices, measurements, dates, findings, or clinical values.
4. Do NOT add any new information.
5. Do NOT remove important findings.
6. Keep the output professional and report-like.
7. Do NOT output bullets unless the original uses bullets.
8. Do NOT explain anything.
9. Do NOT add quotation marks.
10. Return ONLY the rewritten report text. Do not start with or end with any extra text please.

Variation requirement:
{variation}

Original report:
\"\"\"
{report_text}
\"\"\"
""".strip()


def run_ollama(prompt: str, model: str, timeout: int) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("Could not find 'ollama' in PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ollama timed out after {timeout} seconds.")

    if result.returncode != 0:
        message = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(f"Ollama failed: {message}")
    if not (result.stdout or "").strip():
        raise RuntimeError("Ollama returned empty output.")
    return result.stdout


def normalize_generated_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(
        r"^(rephrased report|rewritten report|augmented report|report)\s*:\s*",
        "", text, flags=re.IGNORECASE,
    ).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def validate_generated_output(original: str, generated: str) -> Tuple[bool, str]:
    if not generated.strip():
        return False, "Generated output is empty."

    bad_starts = ["thinking", "here is", "here's", "certainly", "i can",
                  "i'm sorry", "sorry", "note:", "explanation:"]
    lower = generated.lower().strip()
    for prefix in bad_starts:
        if lower.startswith(prefix):
            return False, f"Starts with disallowed prefix: {prefix!r}"

    orig_words = max(1, len(original.split()))
    gen_words = len(generated.split())
    if gen_words < max(5, int(orig_words * 0.35)):
        return False, (f"Suspiciously short: original={orig_words} words, "
                       f"generated={gen_words} words.")
    return True, "OK"


def generate_augmentation(
    report_text: str, accession: str, variant_idx: int,
    cfg, log_file: Path,
) -> str:
    """One augmented variant; falls back to the original text on repeated failure."""
    prompt = build_prompt(report_text, variant_idx)

    for attempt in range(1, cfg.max_retries + 1):
        try:
            raw = run_ollama(prompt, cfg.model, cfg.ollama_timeout)
            generated = normalize_generated_text(raw)
            is_valid, reason = validate_generated_output(report_text, generated)
            if not is_valid:
                _append_log(log_file,
                            f"INVALID | accession={accession} | variant={variant_idx} "
                            f"| attempt={attempt} | reason={reason}\n"
                            f"----- OUTPUT -----\n{generated}\n----- END -----")
                raise RuntimeError(reason)
            return generated
        except Exception as e:
            exc = "".join(traceback.format_exception_only(type(e), e)).strip()
            _append_log(log_file,
                        f"FAILURE | accession={accession} | variant={variant_idx} "
                        f"| attempt={attempt} | {exc}")
            print(f"[WARN] accession={accession} variant={variant_idx} "
                  f"attempt {attempt}/{cfg.max_retries} failed: {e}",
                  file=sys.stderr)
            time.sleep(cfg.retry_sleep)

    _append_log(log_file,
                f"FALLBACK | accession={accession} | variant={variant_idx} "
                f"| all {cfg.max_retries} attempts failed — using original text")
    return report_text


def expected_types(n_augmentations: int) -> Set[str]:
    return {"Original"} | {f"Augmented {i}" for i in range(1, n_augmentations + 1)}


def get_completed_accessions(
    output_csv: Path, acc_col: str, n_augmentations: int, log_file: Path,
) -> Set[str]:
    """An accession counts as complete only if all expected Type rows exist."""
    if not output_csv.exists():
        return set()
    try:
        out_df = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    except Exception as e:
        _append_log(log_file, f"WARNING | Could not read output CSV for resume: {e}")
        return set()
    if not {acc_col, "Type"}.issubset(out_df.columns):
        _append_log(log_file, "WARNING | Output CSV missing resume columns.")
        return set()

    needed = expected_types(n_augmentations)
    grouped = out_df.groupby(out_df[acc_col].str.strip())["Type"].apply(
        lambda s: set(s.str.strip()))
    completed = {acc for acc, types in grouped.items() if needed.issubset(types)}
    _append_log(log_file, f"RESUME_SCAN | {len(completed)} accessions already complete.")
    return completed


def run(cfg, run_dir: Path, data_dir: Path) -> Dict:
    cleaned_csv = data_dir / "cleaned_reports.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"{cleaned_csv} not found — run step 00 (cleaning) first.")

    output_csv = data_dir / "augmented_reports.csv"
    log_file = data_dir / "augment.log"
    acc_col = cfg.accession_column

    df = pd.read_csv(cleaned_csv, dtype=str, keep_default_na=False)
    if acc_col not in df.columns:
        raise KeyError(f"Column '{acc_col}' not in {cleaned_csv}")

    # Prefer the cleaned text produced by step 00; fall back to the raw column.
    text_col = next((c for c in df.columns if c.startswith("cleaned_")), None)
    if text_col is None:
        text_col = cfg.report_column
    if text_col not in df.columns:
        raise KeyError(f"No report-text column found in {cleaned_csv}")

    df[acc_col] = df[acc_col].str.strip()
    df[text_col] = df[text_col].map(normalize_text)
    df = df[(df[acc_col] != "") & (df[text_col] != "")]
    df = df.drop_duplicates(subset=[acc_col], keep="first").reset_index(drop=True)

    completed = get_completed_accessions(output_csv, acc_col,
                                         cfg.n_augmentations, log_file)
    remaining = df[~df[acc_col].isin(completed)].reset_index(drop=True)

    _append_log(log_file,
                f"RUN_START | total={len(df)} | already_complete={len(completed)} "
                f"| remaining={len(remaining)} | model={cfg.model}")
    print(f"[01] Reports: {len(df)} total, {len(completed)} already complete, "
          f"{len(remaining)} to process (text column: '{text_col}')")

    write_header = not output_csv.exists()
    fallbacks = 0

    for _, row in tqdm(remaining.iterrows(), total=len(remaining),
                       desc="[01] Augmenting", unit="report"):
        accession = row[acc_col]
        original = row[text_col]
        base = row.to_dict()

        rows_out: List[Dict] = [{**base, "Type": "Original"}]
        for idx in range(1, cfg.n_augmentations + 1):
            augmented = generate_augmentation(original, accession, idx, cfg, log_file)
            if augmented == original:
                fallbacks += 1
            rows_out.append({**base, text_col: augmented, "Type": f"Augmented {idx}"})

        pd.DataFrame(rows_out).to_csv(output_csv, mode="a", index=False,
                                      header=write_header)
        write_header = False
        _append_log(log_file, f"COMPLETED | accession={accession} "
                              f"| rows_written={len(rows_out)}")

    _append_log(log_file, f"RUN_END | processed_this_run={len(remaining)}")

    summary = {
        "unique_reports": len(df),
        "already_complete": len(completed),
        "processed_this_run": len(remaining),
        "augmentations_per_report": cfg.n_augmentations,
        "fallbacks_to_original": fallbacks,
        "output_csv": str(output_csv),
        "text_column": text_col,
    }
    print(f"[01] {summary}")
    return summary
