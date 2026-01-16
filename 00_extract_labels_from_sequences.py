#!/usr/bin/env python3
"""
extract_anatomical_questions_from_sequences_json.py

Reads per-report JSON files that contain multiple sequence-level text chunks,
asks an LLM (via Ollama) the SAME anatomical-level QUESTIONS per sequence,
and writes ONE new JSON file per source JSON file (in the same directory).

Input dir (default):
  /data/Deep_Angiography/Reports/Report_List_v01_01_sequences_json

Output:
  For each <name>.json, writes <name>_anatomy.json (unless --suffix changed)

Notes:
- This script tries to be robust to different JSON schemas by searching for
  sequences in common keys and for per-sequence text in common fields.
- If a sequence text is missing/empty, it records "Not stated" with 0 confidence.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from tqdm import tqdm

# -----------------------------
# Defaults / Configuration
# -----------------------------
DEFAULT_IN_DIR = Path("/data/Deep_Angiography/Reports/Report_List_v01_01_sequences_json")

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL_NAME = "qwen2.5vl:32b"
DEFAULT_TIMEOUT_S = 120

QUESTIONS = [
    "Which artery is catheterized?",
    "Is variant anatomy present?",
    "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
    "Is there evidence of arterial or venous dissection?",
    "Is stenosis present in any visualized vessel?",
    "Is an endovascular stent visible in this sequence?",
]

BASE_PROMPT = """ROLE
You are a meticulous clinical information extraction engine for radiology/interventional radiology narrative reports. Your job is to convert unstructured text into a high-precision, audit-friendly answer to a single targeted question.

WHY THIS MATTERS (INCENTIVE)
Your output will be used to build a research-grade labeled dataset. High precision is more important than guessing.

SOURCE OF TRUTH
Use ONLY the provided sequence text (not the full report).

TASK
Answer exactly ONE question:
Question: {QUESTION}

OUTPUT FORMAT (JSON ONLY)
Return:
- answer
- confidence (0–100)
- evidence (≤3 short excerpts)
- notes

CONTEXT (SEQUENCE TEXT):
<<<SEQUENCE_TEXT_START>>>
{SEQUENCE_TEXT}
<<<SEQUENCE_TEXT_END>>>
"""

# Common places where "sequences" might live
SEQUENCE_CONTAINER_KEYS = [
    "sequences",
    "sequence_infos",
    "sequence_level_infos",
    "sequence_level_info",
    "runs",
    "run_infos",
    "chunks",
    "sequence_chunks",
]

# Common fields where per-sequence text might live
SEQUENCE_TEXT_KEYS = [
    "sequence_text",
    "text",
    "chunk",
    "verbatim_chunk",
    "verbatim",
    "content",
    "report_chunk",
    "sequence_chunk",
    "sequence",
]

# -----------------------------
# Helpers
# -----------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse model output as JSON; salvage first {...} block if needed."""
    if not text:
        return None
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def build_prompt(sequence_text: str, question: str) -> str:
    return BASE_PROMPT.format(QUESTION=question, SEQUENCE_TEXT=sequence_text)


def ollama_chat(prompt: str, model: str, url: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def looks_like_sequence_list(x: Any) -> bool:
    if not isinstance(x, list) or len(x) == 0:
        return False
    # allow list of dicts or list of strings
    if all(isinstance(i, dict) for i in x):
        return True
    if all(isinstance(i, str) for i in x):
        return True
    return False


def extract_sequences(root: Any) -> Tuple[List[Any], str]:
    """
    Returns (sequences, path_hint).
    - sequences: list[dict] or list[str]
    - path_hint: where we found them (for metadata/debug)
    """
    # 1) direct list at top level
    if looks_like_sequence_list(root):
        return list(root), "$"

    # 2) dict with known container keys
    if isinstance(root, dict):
        for k in SEQUENCE_CONTAINER_KEYS:
            v = root.get(k)
            if looks_like_sequence_list(v):
                return list(v), f"$.{k}"

    # 3) try a shallow scan for any list that looks like sequences
    if isinstance(root, dict):
        for k, v in root.items():
            if looks_like_sequence_list(v):
                return list(v), f"$.{k}"

    # 4) nothing found
    return [], ""


def get_sequence_text(seq: Any) -> str:
    """
    Given a sequence object (dict or string), return best-effort text.
    """
    if isinstance(seq, str):
        return seq.strip()

    if isinstance(seq, dict):
        for k in SEQUENCE_TEXT_KEYS:
            v = seq.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # sometimes text is nested, e.g., {"sequence": {"text": "..."}}
        for k, v in seq.items():
            if isinstance(v, dict):
                for kk in SEQUENCE_TEXT_KEYS:
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()

    return ""


def get_sequence_id(seq: Any, fallback_idx: int) -> Union[str, int]:
    """Best-effort stable identifier for a sequence."""
    if isinstance(seq, dict):
        for k in ["sequence_id", "seq_id", "id", "run_id", "index", "sequence_index"]:
            v = seq.get(k)
            if isinstance(v, (str, int)) and str(v) != "":
                return v
    return fallback_idx


# -----------------------------
# Main processing
# -----------------------------
def process_one_file(
    in_path: Path,
    out_path: Path,
    model: str,
    url: str,
    timeout_s: int,
    delay_s: float,
) -> Dict[str, Any]:
    src = load_json(in_path)
    sequences, found_at = extract_sequences(src)

    out: Dict[str, Any] = {
        "source_file": str(in_path),
        "sequences_found_at": found_at or None,
        "num_sequences": len(sequences),
        "questions": QUESTIONS,
        "results": [],
    }

    for i, seq in enumerate(sequences):
        seq_text = get_sequence_text(seq)
        seq_id = get_sequence_id(seq, i)

        seq_result: Dict[str, Any] = {
            "sequence_id": seq_id,
            "sequence_text_present": bool(seq_text),
            # keep original sequence payload for traceability (optional but helpful)
            "source_sequence": seq,
            "qa": [],
        }

        if not seq_text:
            for q in QUESTIONS:
                seq_result["qa"].append(
                    {
                        "question": q,
                        "answer": "Not stated",
                        "confidence": 0,
                        "evidence": [],
                        "notes": "Empty or missing sequence text in source JSON",
                    }
                )
            out["results"].append(seq_result)
            continue

        for q in QUESTIONS:
            try:
                prompt = build_prompt(seq_text, q)
                raw = ollama_chat(prompt=prompt, model=model, url=url, timeout_s=timeout_s)
                parsed = safe_parse_json(raw)
                if not parsed:
                    raise ValueError("Invalid JSON returned by model")

                seq_result["qa"].append(
                    {
                        "question": q,
                        "answer": parsed.get("answer"),
                        "confidence": parsed.get("confidence"),
                        "evidence": parsed.get("evidence", []),
                        "notes": parsed.get("notes", ""),
                        "raw_model_text": raw if parsed is None else None,  # usually None
                    }
                )
            except Exception as e:
                seq_result["qa"].append(
                    {
                        "question": q,
                        "answer": "Unclear",
                        "confidence": 0,
                        "evidence": [],
                        "notes": f"Error: {str(e)[:300]}",
                    }
                )

            if delay_s > 0:
                time.sleep(delay_s)

        out["results"].append(seq_result)

    write_json(out_path, out)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract anatomical-level Q/A from sequence JSON files (per-sequence) using Ollama."
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_IN_DIR, help="Directory containing source JSON files")
    parser.add_argument("--suffix", type=str, default="_anatomy", help="Suffix for output JSON filenames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist")
    parser.add_argument("--limit_files", type=int, default=None, help="Process only first N JSON files")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0, help="Delay (seconds) between model calls")

    args = parser.parse_args()

    in_dir: Path = args.dir
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    json_files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])

    # Avoid re-processing already-generated output files (common pattern: *_anatomy.json)
    json_files = [p for p in json_files if not p.stem.endswith(args.suffix)]

    if args.limit_files is not None:
        json_files = json_files[: args.limit_files]

    if not json_files:
        print(f"No input JSON files found in {in_dir} (excluding already-suffixed outputs).")
        return

    total = len(json_files)
    print(f"Found {total} source JSON files in: {in_dir}")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.url}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output suffix: {args.suffix}")
    print("Generating outputs in the SAME directory.\n")

    with tqdm(total=total, desc="Files", unit="file") as pbar:
        for in_path in json_files:
            out_path = in_path.with_name(f"{in_path.stem}{args.suffix}.json")

            if out_path.exists() and not args.overwrite:
                pbar.update(1)
                continue

            try:
                _ = process_one_file(
                    in_path=in_path,
                    out_path=out_path,
                    model=args.model,
                    url=args.url,
                    timeout_s=args.timeout,
                    delay_s=args.delay,
                )
            except Exception as e:
                # Write a minimal failure artifact so you can see which file failed
                fail_obj = {
                    "source_file": str(in_path),
                    "error": str(e),
                }
                write_json(out_path, fail_obj)

            pbar.update(1)

    print("\nDone ✔ Outputs saved alongside sources (one *_anatomy.json per source file).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — already-written output JSON files are preserved.", file=sys.stderr)
        raise
