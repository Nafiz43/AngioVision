"""
Stage 1 — Title & Abstract Screening
AngioVision Systematic Review (PRISMA)

Backend: ollama Python library  (no separate `ollama serve` required)
         Install:  pip install ollama
         Model:    ollama pull qwen3:14b

Two-call architecture per paper:
  Call 1 — Inclusion gate  (I1–I4, ALL must be met)
  Call 2 — Exclusion gate  (E1–E5, ANY one triggers exclusion)
           Skipped when Call 1 already rejects the paper.

Parallel execution:
  The script splits pending records across N worker processes (default 2).
  Each worker writes to its own shard CSV.  The main process merges shards
  into the final output CSV once all workers finish.  Resume works per-shard
  so a crashed run picks up exactly where each worker left off.

Usage:
    python3 03_stage1_screening.py
    python3 03_stage1_screening.py --workers 2
    python3 03_stage1_screening.py --input records.csv --output stage1_results.csv
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
import multiprocessing
import pandas as pd
import ollama
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
MODEL            = "qwen3:14b"
TEMPERATURE      = 0
MAX_TOKENS       = 512
RETRY_LIMIT      = 3
RETRY_DELAY      = 2
REPAIR_LIMIT     = 2
KEEP_ALIVE       = "120m"
ABSTRACT_MAX_LEN = 2500
DEFAULT_WORKERS  = 2

INPUT_COLUMNS = [
    "record_id", "source", "title", "authors", "year",
    "journal_venue", "doi", "url", "abstract",
    "screen_decision", "screen_reason", "notes",
]

LLM_COLUMNS = [
    "llm_decision",
    "llm_failed_at",
    "llm_inclusion_pass",
    "llm_inclusion_criteria",
    "llm_inclusion_reason",
    "llm_exclusion_pass",
    "llm_exclusion_criteria",
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


# ── Per-worker logging (each worker writes to its own log file) ───────────────

def _get_logger(worker_id=None):
    name   = f"stage1.worker{worker_id}" if worker_id is not None else "stage1.main"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_file = f"stage1_worker{worker_id}.log" if worker_id is not None else "stage1.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


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


def _repair_json(broken_raw, system_prompt, title, abstract, log):
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
                keep_alive=KEEP_ALIVE,
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


def _call_gate(gate_name, system_prompt, title, abstract, log):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"/no_think\n\nTitle: {title}\n\nAbstract: {abstract}"},
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
                keep_alive=KEEP_ALIVE,
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
    parsed, repair_raw = _repair_json(last_raw, system_prompt, title, abstract, log)
    if parsed:
        return parsed, f"[REPAIRED] {repair_raw}"

    log.error(f"  [{gate_name}] JSON repair failed.")
    return None, last_raw


# ── Model preload ─────────────────────────────────────────────────────────────

def _preload_model(log):
    log.info(f"Preloading model '{MODEL}' into VRAM (keep_alive={KEEP_ALIVE})...")
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": '/no_think\n\nRespond with: {"ok": true}'},
            ],
            format="json",
            options=ollama.Options(temperature=0, num_predict=32),
            keep_alive=KEEP_ALIVE,
        )
        raw = response.message.content.strip()
        log.info(f"Model preloaded. Warm-up response: {raw[:60]}")
    except Exception as e:
        log.warning(f"Preload failed (non-fatal): {e}")


# ── Two-call screening ────────────────────────────────────────────────────────

def _screen_paper(title, abstract, log):
    abstract = abstract[:ABSTRACT_MAX_LEN]

    # ── CALL 1: Inclusion gate ────────────────────────────────────
    log.info(f"  [inclusion] calling model...")
    inc_parsed, inc_raw = _call_gate("inclusion", INCLUSION_SYSTEM, title, abstract, log)

    if inc_parsed is None:
        return {
            "llm_decision":              "ERROR",
            "llm_failed_at":             "inclusion",
            "llm_inclusion_pass":        "ERROR",
            "llm_inclusion_criteria":    "{}",
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

    inc_criteria_store = json.dumps({"met": inc_triggered, "failed": inc_failed})

    # Early exit — exclusion gate skipped entirely
    if not inc_pass or inc_uncertain:
        decision  = "UNCERTAIN" if inc_uncertain else "EXCLUDE"
        failed_at = "inclusion (uncertain)" if inc_uncertain else "inclusion"
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
    exc_parsed, exc_raw = _call_gate("exclusion", EXCLUSION_SYSTEM, title, abstract, log)

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


# ── Worker entry point ────────────────────────────────────────────────────────

def _worker(worker_id, records, shard_path, progress_queue):
    """
    Runs in a separate process.  Screens its slice of records and writes
    results to shard_path.  Sends (worker_id, record_id, decision) tuples
    to progress_queue so the main process can update a unified progress bar.
    """
    log = _get_logger(worker_id)

    # Load any already-processed records for this shard (resume support)
    done_ids = set()
    if os.path.exists(shard_path):
        done_df  = pd.read_csv(shard_path)
        done_ids = set(done_df["record_id"].astype(str))
        log.info(f"Worker {worker_id}: resuming — {len(done_ids)} records already in shard.")
    else:
        done_df = pd.DataFrame(columns=INPUT_COLUMNS + LLM_COLUMNS)

    pending  = [r for r in records if str(r["record_id"]) not in done_ids]
    new_rows = []

    _preload_model(log)

    for row in pending:
        rec_id   = str(row["record_id"])
        title    = str(row.get("title",    "") or "").strip()
        abstract = str(row.get("abstract", "") or "").strip()

        if not title and not abstract:
            log.warning(f"Worker {worker_id}: record_id={rec_id} empty — skipping.")
            progress_queue.put((worker_id, rec_id, "SKIP"))
            continue

        llm_out  = _screen_paper(title, abstract, log)
        decision = llm_out.get("llm_decision", "?")

        out_row = {col: row.get(col, None) for col in INPUT_COLUMNS}
        out_row.update(llm_out)
        new_rows.append(out_row)

        # Write shard after every record (crash-safe)
        batch_df = pd.DataFrame(new_rows, columns=INPUT_COLUMNS + LLM_COLUMNS)
        combined = pd.concat([done_df, batch_df], ignore_index=True)
        combined.to_csv(shard_path, index=False)

        progress_queue.put((worker_id, rec_id, decision))

    log.info(f"Worker {worker_id}: finished.")
    progress_queue.put((worker_id, None, "DONE"))


# ── Merge shards ──────────────────────────────────────────────────────────────

def _merge_shards(shard_paths, output_path, input_path):
    """
    Merge all shard CSVs into the final output, preserving the original
    record order from the input CSV.
    """
    log = _get_logger()
    log.info("Merging shard files...")

    parts = [pd.read_csv(p) for p in shard_paths if os.path.exists(p)]
    if not parts:
        log.error("No shard files found to merge.")
        return

    merged = pd.concat(parts, ignore_index=True)

    # Restore original input order
    input_df = pd.read_csv(input_path)
    order    = input_df["record_id"].astype(str).tolist()
    merged["record_id"] = merged["record_id"].astype(str)
    merged = merged.set_index("record_id").reindex(order).reset_index()

    merged.to_csv(output_path, index=False)
    log.info(f"Merged {len(merged)} records → {output_path}")


# ── Included titles export ────────────────────────────────────────────────────

def _save_included_titles(df, output_path):
    results_dir = os.path.dirname(output_path) or "."
    txt_path    = os.path.join(results_dir, "stage1_included.txt")
    included    = df[df["llm_decision"] == "INCLUDE"].copy()

    if included.empty:
        return

    def _is_flagged(val):
        if isinstance(val, bool): return val
        if isinstance(val, str):  return val.strip().lower() in ("true", "1", "yes")
        try:                      return bool(val)
        except Exception:         return False

    included["_flagged"] = included["llm_flag_for_human_review"].apply(_is_flagged)
    lines = []
    for _, row in included.iterrows():
        title  = str(row.get("title", "")).strip() or "(no title)"
        suffix = "  [Need-to-Check]" if row["_flagged"] else ""
        lines.append(f"{title}{suffix}")

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    flagged_count = included["_flagged"].sum()
    print(f"  Included titles saved to: {txt_path}  "
          f"({len(lines)} total, {flagged_count} flagged [Need-to-Check])\n")


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

    decision_rows = []
    for decision, count in df["llm_decision"].value_counts().items():
        pct = 100 * count / len(df)
        decision_rows.append([decision, count, f"{pct:.1f}%"])
    flags = int(df["llm_flag_for_human_review"].astype(str).str.lower().isin(["true", "1"]).sum())
    decision_rows.append(["Flagged for review", flags, f"{100*flags/len(df):.1f}%"])
    _tbl(decision_rows, ["Decision", "Count", "% of total"])

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

    print("─" * 60)
    print("  INCLUSION GATE — CRITERIA BREAKDOWN")
    print("─" * 60)

    inc_met_counts    = defaultdict(int)
    inc_failed_counts = defaultdict(int)
    for _, row in df.iterrows():
        met, failed = parse_inc_criteria(row.get("llm_inclusion_criteria", "{}"))
        for c in met:    inc_met_counts[c]    += 1
        for c in failed: inc_failed_counts[c] += 1

    inc_rows = []
    for c in ["I1", "I2", "I3", "I4"]:
        label = CRITERIA_LABELS.get(c, c).split("—", 1)[-1].strip()
        inc_rows.append([c, label, inc_met_counts[c], inc_failed_counts[c]])
    _tbl(inc_rows, ["ID", "Description", "# Met", "# Failed"])

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

    criteria_rows = []
    for c in ["I1", "I2", "I3", "I4"]:
        criteria_rows.append({
            "criterion": c, "description": CRITERIA_LABELS[c], "type": "inclusion",
            "n_met": inc_met_counts[c], "n_failed": inc_failed_counts[c], "n_triggered": "",
        })
    for c in ["E1", "E2", "E3", "E4", "E5"]:
        criteria_rows.append({
            "criterion": c, "description": CRITERIA_LABELS[c], "type": "exclusion",
            "n_met": "", "n_failed": "", "n_triggered": exc_triggered_counts[c],
        })
    criteria_path = output_path.replace(".csv", "_criteria_summary.csv")
    pd.DataFrame(criteria_rows).to_csv(criteria_path, index=False)
    print(f"  Criteria summary CSV saved to: {criteria_path}\n")

    _save_included_titles(df, output_path)


# ── Parallel orchestration ────────────────────────────────────────────────────

def screen_records(input_path, output_path, n_workers=DEFAULT_WORKERS, resume=True):
    log = _get_logger()

    df = pd.read_csv(input_path)
    required = {"record_id", "title", "abstract"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    for col in INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Determine which records still need processing across ALL shards
    results_dir = os.path.dirname(output_path) or "."
    os.makedirs(results_dir, exist_ok=True)

    base      = os.path.splitext(os.path.basename(output_path))[0]
    shard_dir = os.path.join(results_dir, f"{base}_shards")
    os.makedirs(shard_dir, exist_ok=True)

    shard_paths = [
        os.path.join(shard_dir, f"shard_{i}.csv") for i in range(n_workers)
    ]

    # ── Resume: collect already-done IDs from ALL sources ────────
    # Priority 1: existing final output CSV (from a previous completed or
    #             interrupted single-worker run, or a previous parallel run)
    # Priority 2: existing shard CSVs (from an interrupted parallel run)
    # Both are checked so that no matter how the script was stopped last
    # time, it always resumes correctly.
    done_ids   = set()
    prior_rows = []   # rows already processed — used to pre-seed shards below

    if resume:
        # Check final output CSV first
        if os.path.exists(output_path):
            prior_df   = pd.read_csv(output_path)
            prior_ids  = set(prior_df["record_id"].astype(str))
            done_ids  |= prior_ids
            prior_rows = prior_df.to_dict("records")
            log.info(
                f"Resuming: found {len(prior_ids)} records in existing output CSV "
                f"({output_path})."
            )

        # Also check shards (catches records written after the last merge)
        shard_extra = set()
        for sp in shard_paths:
            if os.path.exists(sp):
                shard_df    = pd.read_csv(sp)
                shard_ids   = set(shard_df["record_id"].astype(str))
                new_in_shard = shard_ids - done_ids
                if new_in_shard:
                    prior_rows.extend(
                        shard_df[shard_df["record_id"].astype(str).isin(new_in_shard)]
                        .to_dict("records")
                    )
                    shard_extra |= new_in_shard
                done_ids |= shard_ids

        if shard_extra:
            log.info(
                f"Resuming: found {len(shard_extra)} additional records in shard files."
            )

        if done_ids:
            log.info(f"Total already processed: {len(done_ids)} records — skipping these.")

        # Pre-seed shard files with prior rows so each worker's own resume
        # logic stays consistent (workers only read their own shard file).
        if prior_rows:
            prior_df_all = pd.DataFrame(prior_rows, columns=INPUT_COLUMNS + LLM_COLUMNS)
            for i, sp in enumerate(shard_paths):
                if not os.path.exists(sp):
                    # Distribute prior rows across shards so they're roughly balanced
                    chunk = prior_df_all.iloc[i::n_workers]
                    if not chunk.empty:
                        chunk.to_csv(sp, index=False)
                        log.info(
                            f"Pre-seeded shard {i} with {len(chunk)} prior rows."
                        )

    all_records = df.to_dict("records")
    pending     = [r for r in all_records if str(r["record_id"]) not in done_ids]
    total       = len(pending)

    if total == 0:
        log.info("All records already processed. Merging and printing summary.")
        _merge_shards(shard_paths, output_path, input_path)
        _print_summary(output_path)
        return

    log.info(f"Pending: {total} records across {n_workers} workers.")

    # Distribute pending records across workers (round-robin preserves rough balance)
    slices = [pending[i::n_workers] for i in range(n_workers)]

    # Shared queue for progress updates from workers
    progress_queue = multiprocessing.Queue()

    processes = []
    for i, slice_ in enumerate(slices):
        p = multiprocessing.Process(
            target=_worker,
            args=(i, slice_, shard_paths[i], progress_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)
        log.info(f"Worker {i} started (PID {p.pid}) — {len(slice_)} records.")

    # Main process: unified progress bar fed by the queue
    done_workers = 0
    with tqdm(total=total, desc="Screening", unit="record",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        while done_workers < n_workers:
            msg = progress_queue.get()
            worker_id, rec_id, status = msg
            if status == "DONE":
                done_workers += 1
            elif status == "SKIP":
                pbar.update(1)
            else:
                failed_at = ""
                label = status
                if status == "EXCLUDE":
                    label = f"EXCLUDE"
                pbar.set_postfix_str(f"W{worker_id} → {label}", refresh=True)
                pbar.update(1)

    for p in processes:
        p.join()

    log.info("All workers finished. Merging shards...")
    _merge_shards(shard_paths, output_path, input_path)
    _print_summary(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global MODEL

    parser = argparse.ArgumentParser(
        description="Stage 1 — Title & Abstract Screening (parallel two-gate)"
    )
    parser.add_argument("--input",        default="results/slr_stage1_screening.csv")
    parser.add_argument("--output",       default="results/stage1_results.csv")
    parser.add_argument("--model",        default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--workers",      type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-resume",    action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    MODEL = args.model

    if args.summary_only:
        if not os.path.exists(args.output):
            print(f"Error: output file '{args.output}' not found.")
            sys.exit(1)
        _print_summary(args.output)
    else:
        screen_records(
            args.input, args.output,
            n_workers=args.workers,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()