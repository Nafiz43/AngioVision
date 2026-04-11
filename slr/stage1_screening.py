"""
Stage 1 — Title & Abstract Screening
AngioVision Systematic Review (PRISMA)

Backend: ollama Python library  (no separate `ollama serve` required)
         Install:  pip install ollama
         Model:    ollama pull qwen2.5:72b

Usage:
    python3 stage1_screening.py
    python3 stage1_screening.py --input records.csv --output stage1_results.csv

Input CSV columns (all retained in output):
    record_id, source, title, authors, year, journal_venue, doi, url,
    abstract, screen_decision, screen_reason, notes

Added columns (LLM output, merged into input):
    llm_decision, llm_reason, llm_triggered_criteria,
    llm_flag_for_human_review, llm_raw_response
"""

# ── Auto-install dependencies ─────────────────────────────────────────────────
import subprocess
import sys

def _ensure(package, import_name=None):
    name = import_name or package
    try:
        __import__(name)
    except ImportError:
        print(f"[bootstrap] Installing '{package}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

_ensure("ollama")
_ensure("pandas")
_ensure("tqdm")
_ensure("tabulate")

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import argparse
import logging
import pandas as pd
import ollama
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
MODEL       = "qwen2.5:72b"
# MODEL     = "llama3.2:1b"

TEMPERATURE  = 0
MAX_TOKENS   = 512
RETRY_LIMIT  = 3
RETRY_DELAY  = 5
REPAIR_LIMIT = 2

INPUT_COLUMNS = [
    "record_id", "source", "title", "authors", "year",
    "journal_venue", "doi", "url", "abstract",
    "screen_decision", "screen_reason", "notes",
]

LLM_COLUMNS = [
    "llm_decision",
    "llm_reason",
    "llm_triggered_criteria",
    "llm_flag_for_human_review",
    "llm_raw_response",
]

# Criteria labels for readable summary
CRITERIA_LABELS = {
    "I1": "I1 — DSA/fluoroscopy sequence modality",
    "I2": "I2 — Addresses a computational processing task",
    "I3": "I3 — Method described with sufficient detail",
    "I4": "I4 — Quantitative evaluation reported",
    "E1": "E1 — Wrong modality (CT/MRI/US/X-ray only)",
    "E2": "E2 — Single static frame (no temporal component)",
    "E3": "E3 — No original computational method (review/commentary)",
    "E4": "E4 — Abstract-only or poster (no full methods)",
    "E5": "E5 — Non-English language",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("stage1.log")],
)
log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a systematic review assistant. Your task is to determine whether a research
article is RELEVANT or IRRELEVANT for inclusion in a systematic review on deep learning
and machine learning methods applied to angiographic image sequences.

REVIEW SCOPE:
The review covers computational methods (deep learning, machine learning, traditional
image processing) applied to DSA (Digital Subtraction Angiography) or fluoroscopic
image SEQUENCES — i.e., temporally ordered multi-frame data. Tasks of interest include:
image quality enhancement, denoising, artefact reduction, motion correction,
registration, vessel segmentation, detection, classification, localization, and
outcome prediction from sequences.

INCLUSION CRITERIA (ALL must be met):
I1. Imaging modality is DSA or fluoroscopy (sequences/video, not single frames only)
I2. Addresses at least one computational processing task listed above
I3. Method described with sufficient technical detail
I4. Quantitative evaluation reported

EXCLUSION CRITERIA (ANY one triggers exclusion):
E1. Modality is CT, MRI, ultrasound, or X-ray only — no DSA/fluoroscopy
E2. Single static frame only — no temporal/sequence component
E3. Clinical narrative, commentary, or review with no original computational method
E4. Abstract-only or conference poster without full methods
E5. Non-English language

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{
  "decision": "INCLUDE" | "EXCLUDE" | "UNCERTAIN",
  "reason": "<one sentence justification>",
  "triggered_criteria": ["I1", "I2", ...] or ["E1"],
  "flag_for_human_review": true | false
}

If the abstract is ambiguous about the imaging modality or temporal structure,
set decision to "UNCERTAIN" and flag_for_human_review to true.
Do not assume; base your decision only on what is explicitly stated."""

REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or empty response
that was supposed to be a JSON object with this exact schema:

{
  "decision": "INCLUDE" | "EXCLUDE" | "UNCERTAIN",
  "reason": "<one sentence justification>",
  "triggered_criteria": ["I1", "I2", ...] or ["E1"],
  "flag_for_human_review": true | false
}

Return ONLY the corrected JSON object. No preamble, no markdown fences, no explanation."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _repair_json(broken_raw, title, abstract):
    user_content = (
        f"The following response was supposed to be valid JSON but is broken or empty.\n\n"
        f"BROKEN RESPONSE:\n{broken_raw if broken_raw else '(empty)'}\n\n"
        f"ORIGINAL PAPER:\nTitle: {title}\nAbstract: {abstract}\n\n"
        f"Please return the correct JSON object for this paper."
    )
    options = ollama.Options(temperature=0, num_predict=MAX_TOKENS)
    for attempt in range(1, REPAIR_LIMIT + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                format="json",
                options=options,
            )
            raw = response.message.content.strip()
            parsed = _extract_json(raw)
            if parsed:
                log.info(f"  JSON repair succeeded on attempt {attempt}.")
                return parsed, raw
            log.warning(f"  Repair attempt {attempt}/{REPAIR_LIMIT}: still not valid JSON.")
        except Exception as e:
            log.warning(f"  Repair attempt {attempt}/{REPAIR_LIMIT}: error — {e}")
        time.sleep(RETRY_DELAY)
    return None, ""


def _call_model(title, abstract):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Title: {title}\n\nAbstract: {abstract}"},
    ]
    options = ollama.Options(temperature=TEMPERATURE, num_predict=MAX_TOKENS)
    last_raw = ""

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                format="json",
                options=options,
            )
            raw = response.message.content.strip()
            if not raw:
                log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: empty response from model.")
                last_raw = raw
                time.sleep(RETRY_DELAY)
                continue
            parsed = _extract_json(raw)
            if parsed:
                return _build_result(parsed, raw)
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: JSON parse error | raw={raw[:120]}")
            last_raw = raw
            time.sleep(RETRY_DELAY)
        except ollama.ResponseError as e:
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: ollama ResponseError — {e}")
            last_raw = ""
            time.sleep(RETRY_DELAY)
        except Exception as e:
            log.warning(f"  Attempt {attempt}/{RETRY_LIMIT}: Unexpected error — {e}")
            last_raw = ""
            time.sleep(RETRY_DELAY)

    log.warning(f"  All {RETRY_LIMIT} attempts failed. Attempting JSON repair call...")
    parsed, repair_raw = _repair_json(last_raw, title, abstract)
    if parsed:
        return _build_result(parsed, f"[REPAIRED] {repair_raw}")

    log.error("  JSON repair failed. Marking as ERROR.")
    return _error_record(f"JSON repair failed after {RETRY_LIMIT} retries", last_raw)


def _build_result(parsed, raw):
    return {
        "llm_decision":              parsed.get("decision"),
        "llm_reason":                parsed.get("reason"),
        "llm_triggered_criteria":    json.dumps(parsed.get("triggered_criteria", [])),
        "llm_flag_for_human_review": parsed.get("flag_for_human_review"),
        "llm_raw_response":          raw,
    }


def _error_record(reason, raw):
    return {
        "llm_decision":              "ERROR",
        "llm_reason":                reason,
        "llm_triggered_criteria":    "[]",
        "llm_flag_for_human_review": True,
        "llm_raw_response":          raw,
    }


# ── Analysis ──────────────────────────────────────────────────────────────────

def _print_summary(output_path):
    df = pd.read_csv(output_path)

    def parse_criteria(val):
        try:
            result = json.loads(val) if pd.notna(val) else []
            return result if isinstance(result, list) else []
        except Exception:
            return []

    df["_criteria_list"] = df["llm_triggered_criteria"].apply(parse_criteria)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # ── Build criteria counts ─────────────────────────────────────
    criteria_include   = defaultdict(int)
    criteria_exclude   = defaultdict(int)
    criteria_uncertain = defaultdict(int)

    for _, row in df.iterrows():
        decision = str(row.get("llm_decision", ""))
        for c in row["_criteria_list"]:
            if decision == "INCLUDE":
                criteria_include[c] += 1
            elif decision == "EXCLUDE":
                criteria_exclude[c] += 1
            elif decision == "UNCERTAIN":
                criteria_uncertain[c] += 1

    def _tbl(rows, headers):
        """Print a tabulate table with a blank line before and after."""
        print()
        print(tabulate(rows, headers=headers, tablefmt="simple_outline"))
        print()

    # ── 1. Decision counts ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  STAGE 1 SUMMARY")
    print("=" * 50)

    decision_rows = []
    for decision, count in df["llm_decision"].value_counts().items():
        pct = 100 * count / len(df)
        decision_rows.append([decision, count, f"{pct:.1f}%"])
    flags = int(df["llm_flag_for_human_review"].sum())
    decision_rows.append(["Flagged for review", flags, f"{100*flags/len(df):.1f}%"])
    _tbl(decision_rows, ["Decision", "Count", "% of total"])

    # ── 2. Inclusion criteria table ───────────────────────────────
    print("─" * 50)
    print("  INCLUSION CRITERIA BREAKDOWN")
    print("─" * 50)

    inc_rows = []
    for c in sorted(CRITERIA_LABELS.keys()):
        if not c.startswith("I"):
            continue
        inc_rows.append([
            c,
            CRITERIA_LABELS[c].split("—", 1)[1].strip() if "—" in CRITERIA_LABELS[c] else CRITERIA_LABELS[c],
            criteria_include[c],
            criteria_uncertain[c],
        ])
    if inc_rows:
        _tbl(inc_rows, ["ID", "Description", "INCLUDE", "UNCERTAIN"])
    else:
        print("  (none triggered)\n")

    # ── 3. Exclusion criteria table ───────────────────────────────
    print("─" * 50)
    print("  EXCLUSION CRITERIA BREAKDOWN")
    print("─" * 50)

    exc_rows = []
    for c in sorted(CRITERIA_LABELS.keys()):
        if not c.startswith("E"):
            continue
        exc_rows.append([
            c,
            CRITERIA_LABELS[c].split("—", 1)[1].strip() if "—" in CRITERIA_LABELS[c] else CRITERIA_LABELS[c],
            criteria_exclude[c],
            criteria_uncertain[c],
        ])
    if exc_rows:
        _tbl(exc_rows, ["ID", "Description", "EXCLUDE", "UNCERTAIN"])
    else:
        print("  (none triggered)\n")

    # ── 4. Year range table ───────────────────────────────────────
    print("─" * 50)
    print("  YEAR RANGE BY DECISION")
    print("─" * 50)

    year_valid = df["year"].dropna()
    if year_valid.empty:
        print("  No year data available.\n")
    else:
        year_rows = [["ALL", int(year_valid.min()), int(year_valid.max()),
                      int(year_valid.median()), len(year_valid)]]
        for decision in ["INCLUDE", "EXCLUDE", "UNCERTAIN", "ERROR"]:
            subset = df[df["llm_decision"] == decision]["year"].dropna()
            if not subset.empty:
                year_rows.append([decision, int(subset.min()), int(subset.max()),
                                  int(subset.median()), len(subset)])
        _tbl(year_rows, ["Decision", "Min year", "Max year", "Median", "n"])

    # ── 5. Year distribution table ────────────────────────────────
    if not year_valid.empty:
        print("─" * 50)
        print("  YEAR DISTRIBUTION (5-year buckets)")
        print("─" * 50)

        bins   = list(range(int(year_valid.min() // 5 * 5), int(year_valid.max()) + 6, 5))
        labels = [f"{b}–{b+4}" for b in bins[:-1]]
        df["_year_bucket"] = pd.cut(df["year"], bins=bins, labels=labels, right=False)
        bucket_counts = (
            df.groupby(["_year_bucket", "llm_decision"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        # Ensure consistent column order
        col_order = [c for c in ["INCLUDE", "EXCLUDE", "UNCERTAIN", "ERROR"]
                     if c in bucket_counts.columns]
        bucket_counts = bucket_counts[col_order]

        dist_rows = []
        for bucket in bucket_counts.index:
            row_data = [str(bucket)] + [bucket_counts.loc[bucket, c] for c in col_order]
            dist_rows.append(row_data)
        _tbl(dist_rows, ["Period"] + col_order)

    print("=" * 50)

    # ── 6. Save criteria summary CSV ─────────────────────────────
    criteria_rows = []
    for c in sorted(CRITERIA_LABELS.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 99)):
        criteria_rows.append({
            "criterion":   c,
            "description": CRITERIA_LABELS[c],
            "type":        "inclusion" if c.startswith("I") else "exclusion",
            "n_include":   criteria_include[c],
            "n_exclude":   criteria_exclude[c],
            "n_uncertain": criteria_uncertain[c],
        })
    criteria_df = pd.DataFrame(criteria_rows)
    criteria_path = output_path.replace(".csv", "_criteria_summary.csv")
    criteria_df.to_csv(criteria_path, index=False)
    print(f"  Criteria summary CSV saved to: {criteria_path}\n")


# ── Main screening loop ───────────────────────────────────────────────────────

def screen_records(input_path, output_path, resume=True):
    df = pd.read_csv(input_path)

    required = {"record_id", "title", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    for col in INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    done_ids = set()
    if resume and os.path.exists(output_path):
        done_df = pd.read_csv(output_path)
        done_ids = set(done_df["record_id"].astype(str))
        log.info(f"Resuming: {len(done_ids)} records already processed.")
    else:
        done_df = pd.DataFrame(columns=INPUT_COLUMNS + LLM_COLUMNS)

    pending  = df[~df["record_id"].astype(str).isin(done_ids)]
    new_rows = []

    with tqdm(total=len(pending), desc="Screening", unit="record",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for _, row in pending.iterrows():
            rec_id   = str(row["record_id"])
            title    = str(row.get("title",    "")).strip()
            abstract = str(row.get("abstract", "")).strip()

            if not title and not abstract:
                log.warning(f"record_id={rec_id}: empty title & abstract — skipping.")
                pbar.update(1)
                continue

            pbar.set_postfix_str(title[:60], refresh=True)

            llm_out  = _call_model(title, abstract)
            decision = llm_out.get("llm_decision", "?")

            pbar.set_postfix_str(f"{title[:45]}… → {decision}", refresh=True)

            out_row = {col: row.get(col, pd.NA) for col in INPUT_COLUMNS}
            out_row.update(llm_out)
            new_rows.append(out_row)

            batch_df = pd.DataFrame(new_rows, columns=INPUT_COLUMNS + LLM_COLUMNS)
            combined = pd.concat([done_df, batch_df], ignore_index=True)
            combined.to_csv(output_path, index=False)

            pbar.update(1)

    log.info(f"\nDone. Results written to: {output_path}")
    _print_summary(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="Stage 1 — Title & Abstract Screening (ollama / qwen2.5:72b)"
    )
    parser.add_argument("--input",     default="slr_stage1_screening.csv", help="Path to input CSV (default: slr_stage1_screening.csv)")
    parser.add_argument("--output",    default="stage1_results.csv",       help="Path to output CSV (default: stage1_results.csv)")
    parser.add_argument("--model",     default=MODEL,                      help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--no-resume", action="store_true",                help="Start fresh, ignore existing output")
    parser.add_argument("--summary-only", action="store_true",             help="Skip screening, just re-print summary from existing output")
    args = parser.parse_args()

    MODEL = args.model

    if args.summary_only:
        if not os.path.exists(args.output):
            print(f"Error: output file '{args.output}' not found.")
            sys.exit(1)
        _print_summary(args.output)
    else:
        screen_records(args.input, args.output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
