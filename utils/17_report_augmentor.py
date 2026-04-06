#!/usr/bin/env python3
"""
augment_reports_ollama_resume.py

Standalone script for report-level augmentation for AngioVision reports.

Input CSV:
    /data/Deep_Angiography/Reports/Report_List_v01_01.csv

Expected columns:
    - Orig Acc #
    - Anon Acc #
    - radrpt

Behavior:
    - Uses "Anon Acc #" as the unique accession identifier
    - Uses "radrpt" as the report text
    - For each unique accession/report:
        * keeps the original
        * generates 4 rephrased versions
    - Adds a new column: Type
        * Original
        * Augmented 1
        * Augmented 2
        * Augmented 3
        * Augmented 4
    - Saves output CSV in the same directory as the input
    - Writes incrementally after each report
    - Supports resume: if output CSV already exists, skips accessions already completed
    - Writes detailed failure logs with raw/normalized model outputs

Model:
    thewindmom/llama3-med42-8b

Ollama usage:
    Uses the Ollama CLI:
        ollama run thewindmom/llama3-med42-8b

Example:
    python3 augment_reports_ollama_resume.py

Optional:
    python3 augment_reports_ollama_resume.py \
        --input_csv /data/Deep_Angiography/Reports/Report_List_v01_01.csv \
        --output_csv /data/Deep_Angiography/Reports/Report_List_v01_01_augmented.csv \
        --log_file /data/Deep_Angiography/Reports/Report_List_v01_01_augmented.log

Requirements:
    pip install pandas tqdm

Make sure the model exists locally:
    ollama pull thewindmom/llama3-med42-8b
"""

import argparse
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


DEFAULT_INPUT_CSV = "/data/Deep_Angiography/Reports/Report_List_v01_01.csv"
DEFAULT_OUTPUT_CSV = "/data/Deep_Angiography/Reports/Report_List_v01_01_augmented.csv"
DEFAULT_LOG_FILE = "/data/Deep_Angiography/Reports/Report_List_v01_01_augmented.log"
DEFAULT_MODEL = "thewindmom/llama3-med42-8b"

ACCESSION_COL = "Anon Acc #"
REPORT_COL = "radrpt"
EXPECTED_TYPES = {
    "Original",
    "Augmented 1",
    "Augmented 2",
    "Augmented 3",
    "Augmented 4",
}


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving report content."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{now_ts()}] {message}\n")


def format_block(title: str, content: str) -> str:
    return (
        f"\n----- {title} START -----\n"
        f"{content}\n"
        f"----- {title} END -----\n"
    )


def build_prompt(report_text: str, variant_idx: int) -> str:
    """
    Conservative augmentation prompt.
    The goal is slight structural rephrasing while preserving key clinical terms.
    """
    variation_styles = {
        1: "Make only very small sentence-level wording changes.",
        2: "Slightly reorganize sentence structure while preserving the same medical meaning.",
        3: "Use light paraphrasing, but keep clinical terms unchanged whenever possible.",
        4: "Use minor structural variation and wording changes while preserving all findings."
    }
    variation = variation_styles.get(variant_idx, "Make slight structural changes only.")

    prompt = f"""
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

    return prompt


def run_ollama(prompt: str, model: str, timeout: int = 300) -> str:
    """Run Ollama model via CLI and return raw stdout."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("Could not find 'ollama' in PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Ollama timed out after {timeout} seconds.")

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        message = stderr if stderr else stdout
        raise RuntimeError(f"Ollama failed: {message}")

    output = result.stdout
    if output is None or not str(output).strip():
        raise RuntimeError("Ollama returned empty output.")

    return output


def normalize_generated_text(text: str) -> str:
    """Remove wrapper artifacts sometimes added by LLMs."""
    text = clean_text(text)

    text = re.sub(
        r"^(rephrased report|rewritten report|augmented report|report)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    if len(text) >= 2:
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()

    return text


def validate_generated_output(original: str, generated: str) -> Tuple[bool, str]:
    """
    Return (is_valid, reason).
    This makes the exact rejection reason visible in logs.
    """
    if not generated.strip():
        return False, "Generated output is empty."

    bad_starts = [
        "thinking", 
        "here is",
        "here's",
        "certainly",
        "i can",
        "i'm sorry",
        "sorry",
        "note:",
        "explanation:",
    ]
    lower = generated.lower().strip()
    for prefix in bad_starts:
        if lower.startswith(prefix):
            return False, f"Generated output starts with disallowed prefix: {prefix!r}"

    orig_words = max(1, len(original.split()))
    gen_words = len(generated.split())

    if gen_words < max(5, int(orig_words * 0.35)):
        return (
            False,
            f"Generated output is suspiciously short. "
            f"Original words={orig_words}, generated words={gen_words}."
        )

    return True, "OK"


def log_generation_failure(
    log_file: Path,
    accession: str,
    variant_idx: int,
    attempt: int,
    reason: str,
    prompt: str,
    raw_output: Optional[str] = None,
    normalized_output: Optional[str] = None,
    exception_text: Optional[str] = None,
) -> None:
    parts = [
        f"FAILURE | accession={accession} | augmentation=Augmented {variant_idx} | attempt={attempt}",
        f"reason={reason}",
    ]

    if exception_text:
        parts.append(format_block("EXCEPTION", exception_text))
    parts.append(format_block("PROMPT", prompt))

    if raw_output is not None:
        parts.append(format_block("RAW_MODEL_OUTPUT", raw_output))
    if normalized_output is not None:
        parts.append(format_block("NORMALIZED_OUTPUT", normalized_output))

    append_log(log_file, "\n".join(parts))


def generate_augmentation(
    report_text: str,
    accession: str,
    variant_idx: int,
    model: str,
    log_file: Path,
    max_retries: int = 3,
    retry_sleep: float = 1.5,
) -> str:
    """
    Generate one augmented report; fallback to original if repeated failure.
    Logs exact raw/normalized LLM outputs for failed attempts.
    """
    prompt = build_prompt(report_text, variant_idx)
    last_error = None

    for attempt in range(1, max_retries + 1):
        raw_output = None
        normalized_output = None

        try:
            raw_output = run_ollama(prompt, model=model)
            normalized_output = normalize_generated_text(raw_output)

            is_valid, reason = validate_generated_output(report_text, normalized_output)
            if not is_valid:
                log_generation_failure(
                    log_file=log_file,
                    accession=accession,
                    variant_idx=variant_idx,
                    attempt=attempt,
                    reason=reason,
                    prompt=prompt,
                    raw_output=raw_output,
                    normalized_output=normalized_output,
                    exception_text=None,
                )
                raise RuntimeError(reason)

            return normalized_output

        except Exception as e:
            last_error = e
            exc_text = "".join(traceback.format_exception_only(type(e), e)).strip()

            if raw_output is None and normalized_output is None:
                log_generation_failure(
                    log_file=log_file,
                    accession=accession,
                    variant_idx=variant_idx,
                    attempt=attempt,
                    reason="Exception during generation before valid output could be processed.",
                    prompt=prompt,
                    raw_output=None,
                    normalized_output=None,
                    exception_text=exc_text,
                )

            print(
                f"[WARN] accession={accession} | Augmented {variant_idx}, "
                f"attempt {attempt}/{max_retries} failed: {e}",
                file=sys.stderr,
            )
            time.sleep(retry_sleep)

    fallback_reason = f"All {max_retries} attempts failed. Falling back to original report."
    append_log(
        log_file,
        f"FALLBACK | accession={accession} | augmentation=Augmented {variant_idx} | reason={fallback_reason}"
    )
    return report_text


def load_and_prepare_reports(input_csv: Path) -> pd.DataFrame:
    """Load CSV and prepare unique report rows."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = [ACCESSION_COL, REPORT_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[ACCESSION_COL] = df[ACCESSION_COL].astype(str).str.strip()
    df[REPORT_COL] = df[REPORT_COL].astype(str).map(clean_text)

    df = df[
        (df[ACCESSION_COL] != "") &
        (df[ACCESSION_COL].str.lower() != "nan") &
        (df[REPORT_COL] != "") &
        (df[REPORT_COL].str.lower() != "nan")
    ].copy()

    df = df.drop_duplicates(subset=[ACCESSION_COL], keep="first").reset_index(drop=True)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Move Type column right after report column if possible."""
    cols = list(df.columns)
    if "Type" not in cols:
        return df

    cols.remove("Type")
    if REPORT_COL in cols:
        insert_idx = cols.index(REPORT_COL) + 1
        cols = cols[:insert_idx] + ["Type"] + cols[insert_idx:]
    else:
        cols = ["Type"] + cols
    return df[cols]


def get_completed_accessions(output_csv: Path, log_file: Path) -> Set[str]:
    """
    Resume logic:
    consider an accession complete only if all 5 expected Type rows exist in output CSV.
    """
    completed: Set[str] = set()

    if not output_csv.exists():
        return completed

    try:
        out_df = pd.read_csv(output_csv)
    except Exception as e:
        append_log(log_file, f"WARNING | Could not read existing output CSV for resume: {e}")
        return completed

    needed_cols = {ACCESSION_COL, "Type"}
    if not needed_cols.issubset(set(out_df.columns)):
        append_log(
            log_file,
            "WARNING | Existing output CSV missing required columns for resume logic. "
            "Will not trust resume state."
        )
        return completed

    out_df[ACCESSION_COL] = out_df[ACCESSION_COL].astype(str).str.strip()
    out_df["Type"] = out_df["Type"].astype(str).str.strip()

    grouped = out_df.groupby(ACCESSION_COL)["Type"].apply(set)

    for accession, type_set in grouped.items():
        if EXPECTED_TYPES.issubset(type_set):
            completed.add(accession)

    append_log(log_file, f"RESUME_SCAN | Found {len(completed)} completed accessions in existing output.")
    return completed


def write_rows(output_csv: Path, rows_to_write: List[Dict], write_header: bool) -> None:
    temp_df = pd.DataFrame(rows_to_write)
    temp_df = reorder_columns(temp_df)
    temp_df.to_csv(
        output_csv,
        mode="a",
        index=False,
        header=write_header,
    )


def augment_and_save_incremental(
    df: pd.DataFrame,
    output_csv: Path,
    model: str,
    log_file: Path,
) -> None:
    """
    Process each report and append its 5 rows to output CSV.
    Resume-safe: skips accessions already fully completed in existing output file.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    completed_accessions = get_completed_accessions(output_csv, log_file)
    total_reports = len(df)

    remaining_mask = ~df[ACCESSION_COL].astype(str).isin(completed_accessions)
    remaining_df = df[remaining_mask].reset_index(drop=True)

    append_log(
        log_file,
        f"RUN_START | total_reports={total_reports} | completed_already={len(completed_accessions)} "
        f"| remaining={len(remaining_df)} | output_csv={output_csv}"
    )

    print(f"[INFO] Already completed accessions: {len(completed_accessions)}")
    print(f"[INFO] Remaining reports to process: {len(remaining_df)}")

    write_header = not output_csv.exists()

    processed_this_run = 0
    for _, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Augmenting reports"):
        accession = str(row[ACCESSION_COL]).strip()
        base_row = row.to_dict()
        original_report = clean_text(row[REPORT_COL])

        rows_to_write: List[Dict] = []

        # Original
        original_row = dict(base_row)
        original_row["Type"] = "Original"
        original_row[REPORT_COL] = original_report
        rows_to_write.append(original_row)

        # Augmented 1..4
        for aug_idx in range(1, 5):
            augmented_text = generate_augmentation(
                report_text=original_report,
                accession=accession,
                variant_idx=aug_idx,
                model=model,
                log_file=log_file,
            )

            aug_row = dict(base_row)
            aug_row["Type"] = f"Augmented {aug_idx}"
            aug_row[REPORT_COL] = augmented_text
            rows_to_write.append(aug_row)

        write_rows(output_csv, rows_to_write, write_header=write_header)
        write_header = False
        processed_this_run += 1

        append_log(
            log_file,
            f"COMPLETED | accession={accession} | rows_written=5 | processed_this_run={processed_this_run}"
        )

        if processed_this_run % 10 == 0:
            print(
                f"[INFO] Processed {processed_this_run}/{len(remaining_df)} reports this run "
                f"({processed_this_run * 5} rows written this run)."
            )

    append_log(
        log_file,
        f"RUN_END | processed_this_run={processed_this_run} | remaining_after_run=0"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Augment AngioVision reports using Ollama with resume support.")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    parser.add_argument("--log_file", default=DEFAULT_LOG_FILE, help="Detailed log file path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    log_file = Path(args.log_file)

    print(f"[INFO] Input CSV: {input_csv}")
    print(f"[INFO] Output CSV: {output_csv}")
    print(f"[INFO] Log file: {log_file}")
    print(f"[INFO] Model: {args.model}")

    append_log(log_file, f"SCRIPT_START | input_csv={input_csv} | output_csv={output_csv} | model={args.model}")

    df = load_and_prepare_reports(input_csv)
    print(f"[INFO] Unique reports found: {len(df)}")

    if len(df) == 0:
        raise ValueError("No valid reports found in input CSV.")

    augment_and_save_incremental(
        df=df,
        output_csv=output_csv,
        model=args.model,
        log_file=log_file,
    )

    print("[INFO] Done.")
    print(f"[INFO] Saved augmented file to: {output_csv}")
    print(f"[INFO] Detailed log file: {log_file}")


if __name__ == "__main__":
    main()