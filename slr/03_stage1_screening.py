"""
Stage 1 — Title & Abstract Screening
AngioVision Systematic Review (PRISMA)

Backend: ollama Python library  (no separate `ollama serve` required)
         Install:  pip install ollama
         Model:    ollama pull qwen2.5:72b

Two-call architecture per paper:
  Call 1 — Inclusion gate  (I1–I4, ALL must be met)
  Call 2 — Exclusion gate  (E1–E5, ANY one triggers exclusion)
           Skipped when Call 1 already rejects the paper.

Final decision logic:
  inclusion_pass=False                          → EXCLUDE  (failed_at=inclusion)
  inclusion_pass=True  & exclusion_pass=False   → EXCLUDE  (failed_at=exclusion)
  inclusion_pass=True  & exclusion_pass=True    → INCLUDE
  Either call is UNCERTAIN                      → UNCERTAIN

Usage:
    python3 stage1_screening.py
    python3 stage1_screening.py --input records.csv --output stage1_results.csv

Input CSV columns (all retained in output):
    record_id, source, title, authors, year, journal_venue, doi, url,
    abstract, screen_decision, screen_reason, notes

Added columns (LLM output, merged into input):
    llm_decision, llm_failed_at,
    llm_inclusion_pass, llm_inclusion_criteria, llm_inclusion_reason,
    llm_exclusion_pass, llm_exclusion_criteria, llm_exclusion_reason,
    llm_flag_for_human_review, llm_raw_inclusion, llm_raw_exclusion
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
MODEL        = "llama3.1:latest"
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
    "llm_failed_at",                # "inclusion" | "exclusion" | "none" | "error"
    "llm_inclusion_pass",           # True / False / UNCERTAIN
    "llm_inclusion_criteria",       # JSON list of triggered I-codes
    "llm_inclusion_reason",
    "llm_exclusion_pass",           # True / False / UNCERTAIN / SKIPPED
    "llm_exclusion_criteria",       # JSON list of triggered E-codes
    "llm_exclusion_reason",
    "llm_flag_for_human_review",
    "llm_raw_inclusion",
    "llm_raw_exclusion",
]

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


# ── System prompts ────────────────────────────────────────────────────────────

INCLUSION_SYSTEM = """You are a systematic review assistant performing the INCLUSION gate.
Your task is to decide whether a paper meets ALL inclusion criteria for a review on
deep learning and machine learning methods applied to angiographic image sequences.

REVIEW SCOPE:
Computational methods (deep learning, machine learning, traditional image processing)
applied to DSA (Digital Subtraction Angiography) or fluoroscopic image SEQUENCES —
i.e., temporally ordered multi-frame data. Tasks of interest: image quality enhancement,
denoising, artefact reduction, motion correction, registration, vessel segmentation,
detection, classification, localization, and outcome prediction from sequences.

INCLUSION CRITERIA (ALL four must be satisfied to pass):
I1. Imaging modality is DSA or fluoroscopy (sequences/video, not single frames only)
I2. Addresses at least one computational processing task listed above
I3. Method described with sufficient technical detail
I4. Quantitative evaluation reported

Decision rules:
- "pass": ALL four criteria are clearly met.
- "fail": At least one criterion is clearly NOT met. List the failing criterion/criteria.
- "uncertain": Ambiguous — cannot determine from title/abstract alone.

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{
  "pass": true | false,
  "uncertain": true | false,
  "triggered_criteria": ["I1", "I2", "I3", "I4"],
  "failed_criteria": ["I2"],
  "reason": "<one sentence>",
  "flag_for_human_review": true | false
}

Rules:
- "triggered_criteria": inclusion criteria that ARE met.
- "failed_criteria": inclusion criteria that are NOT met (empty list if all pass).
- If "uncertain" is true, set "pass" to false and flag_for_human_review to true.
- Base your decision only on what is explicitly stated in the title and abstract."""

EXCLUSION_SYSTEM = """You are a systematic review assistant performing the EXCLUSION gate.
The paper has already passed the inclusion gate. Your task is to check whether any
exclusion criterion applies and the paper should be rejected.

EXCLUSION CRITERIA (ANY one is sufficient to exclude):
E1. Modality is CT, MRI, ultrasound, or plain X-ray only — no DSA/fluoroscopy content
E2. Single static frame only — no temporal/sequence component
E3. Clinical narrative, commentary, or review with no original computational method
E4. Abstract-only or conference poster without full methods section
E5. Non-English language

Decision rules:
- "pass": NO exclusion criteria triggered — paper survives the exclusion gate.
- "fail": At least one exclusion criterion is triggered. List the triggering criterion/criteria.
- "uncertain": Cannot determine from title/abstract alone.

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{
  "pass": true | false,
  "uncertain": true | false,
  "triggered_criteria": ["E1"],
  "reason": "<one sentence>",
  "flag_for_human_review": true | false
}

Rules:
- "triggered_criteria": exclusion criteria that ARE triggered (causes exclusion).
- If "uncertain" is true, set "pass" to false and flag_for_human_review to true.
- Base your decision only on what is explicitly stated in the title and abstract."""

REPAIR_SYSTEM = """You are a JSON repair assistant. You will be given a broken or empty response
that was supposed to be a JSON object. Return ONLY the corrected JSON object.
No preamble, no markdown fences, no explanation."""


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


def _repair_json(broken_raw, system_prompt, title, abstract):
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
                    {"role": "system", "content": REPAIR_SYSTEM + "\n\n" + system_prompt},
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


def _call_gate(gate_name, system_prompt, title, abstract):
    """
    Generic LLM call for a single gate (inclusion or exclusion).
    Returns (parsed_dict, raw_str) or (None, last_raw) on total failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Title: {title}\n\nAbstract: {abstract}"},
    ]
    options  = ollama.Options(temperature=TEMPERATURE, num_predict=MAX_TOKENS)
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
                log.warning(f"  [{gate_name}] Attempt {attempt}/{RETRY_LIMIT}: empty response.")
                last_raw = raw
                time.sleep(RETRY_DELAY)
                continue
            parsed = _extract_json(raw)
            if parsed:
                return parsed, raw
            log.warning(f"  [{gate_name}] Attempt {attempt}/{RETRY_LIMIT}: JSON parse error | raw={raw[:120]}")
            last_raw = raw
            time.sleep(RETRY_DELAY)
        except ollama.ResponseError as e:
            log.warning(f"  [{gate_name}] Attempt {attempt}/{RETRY_LIMIT}: ResponseError — {e}")
            last_raw = ""
            time.sleep(RETRY_DELAY)
        except Exception as e:
            log.warning(f"  [{gate_name}] Attempt {attempt}/{RETRY_LIMIT}: Unexpected error — {e}")
            last_raw = ""
            time.sleep(RETRY_DELAY)

    log.warning(f"  [{gate_name}] All {RETRY_LIMIT} attempts failed. Attempting JSON repair...")
    parsed, repair_raw = _repair_json(last_raw, system_prompt, title, abstract)
    if parsed:
        return parsed, f"[REPAIRED] {repair_raw}"

    log.error(f"  [{gate_name}] JSON repair failed.")
    return None, last_raw


# ── Two-call screening ────────────────────────────────────────────────────────

def _screen_paper(title, abstract):
    """
    Run inclusion gate then (conditionally) exclusion gate.

    Returns a dict with all LLM_COLUMNS fields populated.
    """
    # ── CALL 1: Inclusion gate ────────────────────────────────────
    log.info(f"  [inclusion] calling model...")
    inc_parsed, inc_raw = _call_gate("inclusion", INCLUSION_SYSTEM, title, abstract)

    if inc_parsed is None:
        # Hard error on inclusion call
        return {
            "llm_decision":              "ERROR",
            "llm_failed_at":             "inclusion",
            "llm_inclusion_pass":        "ERROR",
            "llm_inclusion_criteria":    "[]",
            "llm_inclusion_reason":      "LLM call failed after all retries",
            "llm_exclusion_pass":        "SKIPPED",
            "llm_exclusion_criteria":    "[]",
            "llm_exclusion_reason":      "",
            "llm_flag_for_human_review": True,
            "llm_raw_inclusion":         inc_raw,
            "llm_raw_exclusion":         "",
        }

    inc_pass      = bool(inc_parsed.get("pass", False))
    inc_uncertain = bool(inc_parsed.get("uncertain", False))
    inc_triggered = inc_parsed.get("triggered_criteria", [])
    inc_failed    = inc_parsed.get("failed_criteria", [])
    inc_reason    = inc_parsed.get("reason", "")
    inc_flag      = bool(inc_parsed.get("flag_for_human_review", False))

    # Represent what criteria info to store: met criteria + failed criteria
    inc_criteria_store = json.dumps({
        "met":    inc_triggered,
        "failed": inc_failed,
    })

    # Early exit if inclusion fails
    if not inc_pass or inc_uncertain:
        decision  = "UNCERTAIN" if inc_uncertain else "EXCLUDE"
        failed_at = "inclusion" if not inc_uncertain else "inclusion (uncertain)"
        log.info(f"  [inclusion] → {decision} | failed: {inc_failed}")
        return {
            "llm_decision":              decision,
            "llm_failed_at":             failed_at,
            "llm_inclusion_pass":        "UNCERTAIN" if inc_uncertain else False,
            "llm_inclusion_criteria":    inc_criteria_store,
            "llm_inclusion_reason":      inc_reason,
            "llm_exclusion_pass":        "SKIPPED",
            "llm_exclusion_criteria":    "[]",
            "llm_exclusion_reason":      "Skipped — paper did not pass inclusion gate",
            "llm_flag_for_human_review": inc_flag,
            "llm_raw_inclusion":         inc_raw,
            "llm_raw_exclusion":         "",
        }

    log.info(f"  [inclusion] → PASS | met: {inc_triggered}")

    # ── CALL 2: Exclusion gate ────────────────────────────────────
    log.info(f"  [exclusion] calling model...")
    exc_parsed, exc_raw = _call_gate("exclusion", EXCLUSION_SYSTEM, title, abstract)

    if exc_parsed is None:
        return {
            "llm_decision":              "ERROR",
            "llm_failed_at":             "exclusion",
            "llm_inclusion_pass":        True,
            "llm_inclusion_criteria":    inc_criteria_store,
            "llm_inclusion_reason":      inc_reason,
            "llm_exclusion_pass":        "ERROR",
            "llm_exclusion_criteria":    "[]",
            "llm_exclusion_reason":      "LLM call failed after all retries",
            "llm_flag_for_human_review": True,
            "llm_raw_inclusion":         inc_raw,
            "llm_raw_exclusion":         exc_raw,
        }

    exc_pass      = bool(exc_parsed.get("pass", False))
    exc_uncertain = bool(exc_parsed.get("uncertain", False))
    exc_triggered = exc_parsed.get("triggered_criteria", [])
    exc_reason    = exc_parsed.get("reason", "")
    exc_flag      = bool(exc_parsed.get("flag_for_human_review", False))

    exc_criteria_store = json.dumps(exc_triggered)

    if exc_uncertain:
        log.info(f"  [exclusion] → UNCERTAIN | triggered: {exc_triggered}")
        return {
            "llm_decision":              "UNCERTAIN",
            "llm_failed_at":             "exclusion (uncertain)",
            "llm_inclusion_pass":        True,
            "llm_inclusion_criteria":    inc_criteria_store,
            "llm_inclusion_reason":      inc_reason,
            "llm_exclusion_pass":        "UNCERTAIN",
            "llm_exclusion_criteria":    exc_criteria_store,
            "llm_exclusion_reason":      exc_reason,
            "llm_flag_for_human_review": True,
            "llm_raw_inclusion":         inc_raw,
            "llm_raw_exclusion":         exc_raw,
        }

    if not exc_pass:
        log.info(f"  [exclusion] → EXCLUDE | triggered: {exc_triggered}")
        return {
            "llm_decision":              "EXCLUDE",
            "llm_failed_at":             "exclusion",
            "llm_inclusion_pass":        True,
            "llm_inclusion_criteria":    inc_criteria_store,
            "llm_inclusion_reason":      inc_reason,
            "llm_exclusion_pass":        False,
            "llm_exclusion_criteria":    exc_criteria_store,
            "llm_exclusion_reason":      exc_reason,
            "llm_flag_for_human_review": exc_flag,
            "llm_raw_inclusion":         inc_raw,
            "llm_raw_exclusion":         exc_raw,
        }

    # Both gates passed → INCLUDE
    log.info(f"  [exclusion] → PASS | no exclusion criteria triggered")
    return {
        "llm_decision":              "INCLUDE",
        "llm_failed_at":             "none",
        "llm_inclusion_pass":        True,
        "llm_inclusion_criteria":    inc_criteria_store,
        "llm_inclusion_reason":      inc_reason,
        "llm_exclusion_pass":        True,
        "llm_exclusion_criteria":    exc_criteria_store,
        "llm_exclusion_reason":      exc_reason,
        "llm_flag_for_human_review": inc_flag or exc_flag,
        "llm_raw_inclusion":         inc_raw,
        "llm_raw_exclusion":         exc_raw,
    }


# ── Included titles export ────────────────────────────────────────────────────

def _save_included_titles(df, output_path):
    results_dir = os.path.dirname(output_path) or "."
    txt_path    = os.path.join(results_dir, "stage1_included.txt")
    included    = df[df["llm_decision"] == "INCLUDE"].copy()

    if included.empty:
        log.info("  No INCLUDE records found — skipping stage1_included.txt.")
        return

    def _is_flagged(val):
        if isinstance(val, bool):   return val
        if isinstance(val, str):    return val.strip().lower() in ("true", "1", "yes")
        try:                        return bool(val)
        except Exception:           return False

    included["_flagged"] = included["llm_flag_for_human_review"].apply(_is_flagged)
    lines = []
    for _, row in included.iterrows():
        title  = str(row.get("title", "")).strip() or "(no title)"
        suffix = "  [Need-to-Check]" if row["_flagged"] else ""
        lines.append(f"{title}{suffix}")

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    flagged_count = included["_flagged"].sum()
    msg = (f"  Included titles saved to: {txt_path}  "
           f"({len(lines)} total, {flagged_count} flagged [Need-to-Check])")
    log.info(msg)
    print(msg + "\n")


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(output_path):
    df = pd.read_csv(output_path)

    def _tbl(rows, headers):
        print()
        print(tabulate(rows, headers=headers, tablefmt="simple_outline"))
        print()

    def parse_json_col(val):
        try:
            result = json.loads(val) if pd.notna(val) else []
            return result if isinstance(result, list) else []
        except Exception:
            return []

    # Parse inclusion criteria (stored as {"met": [...], "failed": [...]})
    def parse_inc_criteria(val):
        try:
            obj = json.loads(val) if pd.notna(val) else {}
            if isinstance(obj, dict):
                return obj.get("met", []), obj.get("failed", [])
            return [], []
        except Exception:
            return [], []

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    print("\n" + "=" * 60)
    print("  STAGE 1 SUMMARY  (two-gate architecture)")
    print("=" * 60)

    # ── 1. Overall decision counts ────────────────────────────────
    decision_rows = []
    for decision, count in df["llm_decision"].value_counts().items():
        pct = 100 * count / len(df)
        decision_rows.append([decision, count, f"{pct:.1f}%"])
    flags = int(df["llm_flag_for_human_review"].astype(str).str.lower().isin(["true", "1"]).sum())
    decision_rows.append(["Flagged for review", flags, f"{100*flags/len(df):.1f}%"])
    _tbl(decision_rows, ["Decision", "Count", "% of total"])

    # ── 2. Failure stage breakdown ────────────────────────────────
    print("─" * 60)
    print("  WHERE DID EXCLUSIONS FAIL?")
    print("─" * 60)
    excluded = df[df["llm_decision"] == "EXCLUDE"]
    if not excluded.empty:
        fail_rows = []
        for stage, count in excluded["llm_failed_at"].value_counts().items():
            pct = 100 * count / len(excluded)
            fail_rows.append([stage, count, f"{pct:.1f}% of excluded"])
        _tbl(fail_rows, ["Failed at", "Count", "% of excluded"])
    else:
        print("  No EXCLUDE records.\n")

    # ── 3. Inclusion gate — criteria breakdown ────────────────────
    print("─" * 60)
    print("  INCLUSION GATE — CRITERIA BREAKDOWN")
    print("─" * 60)

    inc_met_counts    = defaultdict(int)
    inc_failed_counts = defaultdict(int)

    for _, row in df.iterrows():
        met, failed = parse_inc_criteria(row.get("llm_inclusion_criteria", "{}"))
        for c in met:
            inc_met_counts[c] += 1
        for c in failed:
            inc_failed_counts[c] += 1

    inc_rows = []
    for c in ["I1", "I2", "I3", "I4"]:
        label = CRITERIA_LABELS.get(c, c).split("—", 1)[-1].strip()
        inc_rows.append([c, label, inc_met_counts[c], inc_failed_counts[c]])
    _tbl(inc_rows, ["ID", "Description", "# Met", "# Failed"])

    # ── 4. Exclusion gate — criteria breakdown ────────────────────
    print("─" * 60)
    print("  EXCLUSION GATE — TRIGGERED CRITERIA BREAKDOWN")
    print("─" * 60)

    exc_triggered_counts = defaultdict(int)
    exc_gate_df = df[df["llm_exclusion_pass"].astype(str).str.upper() != "SKIPPED"]

    for _, row in exc_gate_df.iterrows():
        for c in parse_json_col(row.get("llm_exclusion_criteria", "[]")):
            exc_triggered_counts[c] += 1

    exc_rows = []
    for c in ["E1", "E2", "E3", "E4", "E5"]:
        label = CRITERIA_LABELS.get(c, c).split("—", 1)[-1].strip()
        exc_rows.append([c, label, exc_triggered_counts[c]])
    _tbl(exc_rows, ["ID", "Description", "# Triggered"])

    # ── 5. Year range by decision ─────────────────────────────────
    print("─" * 60)
    print("  YEAR RANGE BY DECISION")
    print("─" * 60)
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

    print("=" * 60)

    # ── 6. Criteria summary CSV ───────────────────────────────────
    criteria_rows = []
    for c in ["I1", "I2", "I3", "I4"]:
        criteria_rows.append({
            "criterion":   c,
            "description": CRITERIA_LABELS[c],
            "type":        "inclusion",
            "n_met":       inc_met_counts[c],
            "n_failed":    inc_failed_counts[c],
            "n_triggered": "",
        })
    for c in ["E1", "E2", "E3", "E4", "E5"]:
        criteria_rows.append({
            "criterion":   c,
            "description": CRITERIA_LABELS[c],
            "type":        "exclusion",
            "n_met":       "",
            "n_failed":    "",
            "n_triggered": exc_triggered_counts[c],
        })
    criteria_path = output_path.replace(".csv", "_criteria_summary.csv")
    pd.DataFrame(criteria_rows).to_csv(criteria_path, index=False)
    print(f"  Criteria summary CSV saved to: {criteria_path}\n")

    _save_included_titles(df, output_path)


# ── Main screening loop ───────────────────────────────────────────────────────

def screen_records(input_path, output_path, resume=True):
    df = pd.read_csv(input_path)

    required = {"record_id", "title", "abstract"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    for col in INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    done_ids = set()
    if resume and os.path.exists(output_path):
        done_df  = pd.read_csv(output_path)
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

            pbar.set_postfix_str(title[:55], refresh=True)

            llm_out  = _screen_paper(title, abstract)
            decision = llm_out.get("llm_decision", "?")
            failed_at = llm_out.get("llm_failed_at", "?")

            status_str = decision
            if decision == "EXCLUDE":
                status_str = f"EXCLUDE [{failed_at}]"
            pbar.set_postfix_str(f"{title[:40]}… → {status_str}", refresh=True)

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
        description="Stage 1 — Title & Abstract Screening (two-gate: inclusion then exclusion)"
    )
    parser.add_argument("--input",        default="results/slr_stage1_screening.csv",
                        help="Path to input CSV")
    parser.add_argument("--output",       default="results/stage1_results.csv",
                        help="Path to output CSV")
    parser.add_argument("--model",        default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Start fresh, ignore existing output")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip screening, just re-print summary from existing output")
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