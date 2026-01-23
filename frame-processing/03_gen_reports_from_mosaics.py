#!/usr/bin/env python3
"""
extract_labels_from_mosaics.py

Stage 2 only: Discover INNER sequence dirs (same rule as stage 1),
read an EXISTING mosaic image from each sequence dir, and query an LLM (Ollama)
to generate an angiography-style report *based only on what is visible in the mosaic*.

CSV behavior (FINAL)
- Adds: Timestamp, Model Name
- Appends row-by-row (never rewrites the full file)
- If CSV exists, new rows are appended
- Output directory is ALWAYS:
    <base_path>_Output/
"""

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

# Default model if user does not pass --model
DEFAULT_MODEL_NAME = "qwen3-vl:32b"
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# Report-generation prompt (UPDATED)
# -----------------------------
BASE_PROMPT = """ROLE
You are a careful, image-grounded interventional radiology (IR) angiography reporting assistant.

GOAL
Generate a concise angiography-style report for the single provided mosaic image (tiled frames).
This will be used for research dataset construction, so accuracy and transparency matter more than completeness.

SOURCE OF TRUTH (VERY IMPORTANT)
- Use ONLY what is visible in the provided mosaic image.
- Do NOT assume clinical history, indication, laterality, catheter type, projections, or phases unless you can see it.
- Do NOT add diagnoses that are not supported by visible evidence.
- If something cannot be determined from the image, explicitly mark it as "Unclear".

WHAT YOU ARE GIVEN (MOSAIC)
- You receive ONE mosaic image that contains multiple frames arranged in reading order:
  left-to-right, top-to-bottom.
- Treat each tile as an individual frame from the same angiographic run/sequence.

WHAT TO PRODUCE
Produce BOTH:
1) A short free-text angiography report (research-friendly, no PHI).
2) A structured summary of key elements you can infer from the image.

STRICT RULES
1) Do not guess. Prefer "Unclear" over speculation.
2) If evidence is contradictory across tiles, mark the relevant field as "Conflicting".
3) Avoid overconfident language. Use hedging when appropriate (e.g., "suggests", "may represent").
4) Do NOT include any patient identifiers or invented demographics.
5) Keep it self-contained and understandable without external context.

REPORT CONTENT GUIDANCE (write what you can, omit what you can't)
- Technique / Description of sequence (image-based only): e.g., "angiographic run with contrast opacification..."
- Catheter position / injected territory (if visible): e.g., "catheter tip appears in ..."
- Vessels opacified (if visible): e.g., "arterial tree of ..."
- Key findings (only if visible): stenosis/occlusion, aneurysm/pseudoaneurysm, dissection, extravasation/hemorrhage, AV shunting/early venous drainage, vasospasm, stent/coil/embolization material, other devices.
- Impression: 1–3 bullets, image-grounded, with uncertainty noted.
- Limitations: e.g., "single mosaic; phase/projection unclear; limited field of view..."

OUTPUT FORMAT (JSON ONLY — no extra text)
Return a single JSON object with EXACTLY these keys:
{
  "report_text": string,                 # concise, angiography-style paragraph(s)
  "catheterized_territory": string,      # one of: specific territory / "Unclear" / "Conflicting"
  "vessels_opacified": [string, ...],    # list; empty list if none can be determined
  "key_findings": [string, ...],         # list of short findings; include "None visible" if appropriate
  "impression": [string, ...],           # 1–3 bullets; can include uncertainty
  "limitations": [string, ...],          # short bullets
  "confidence": integer,                 # 0–100 overall confidence based on image clarity
  "evidence": [string, ...],             # up to 5 short cues tied to what you see (e.g., "contrast pooling outside vessel", "early venous filling")
  "notes": string                        # optional; use "" if no extra notes
}

QUALITY CHECK BEFORE YOU ANSWER
- Is every claim supported by something visible?
- Did you avoid guessing and PHI?
- Is the output valid JSON and only JSON?
"""

# -----------------------------
# CSV helpers (append-only)
# -----------------------------
def ensure_csv_header(out_path: Path, columns: List[str]) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


# -----------------------------
# Directory discovery
# -----------------------------
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    """
    A "sequence dir" is any directory under base_path that contains frames_subdir/
    with at least one image file.
    """
    seq_dirs: List[Path] = []
    for d in base_path.rglob("*"):
        if not d.is_dir():
            continue
        frames_dir = d / frames_subdir
        if not frames_dir.exists():
            continue
        if any(p.suffix.lower() in IMAGE_EXTS for p in frames_dir.iterdir()):
            seq_dirs.append(d)
    return sorted(seq_dirs, key=lambda p: p.as_posix())


# -----------------------------
# Helpers
# -----------------------------
def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_prompt() -> str:
    return BASE_PROMPT


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to parse JSON robustly even if the model wraps it with extra text.
    """
    try:
        return json.loads(text.strip())
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def ollama_chat_with_images(prompt: str, images_b64: List[str], model: str, url: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SequenceMosaicInfo:
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    ok: bool
    error: Optional[str] = None


def load_mosaics(seq_dirs: List[Path], base_path: Path, mosaic_name: str) -> List[SequenceMosaicInfo]:
    infos: List[SequenceMosaicInfo] = []
    for d in seq_dirs:
        rel = d.relative_to(base_path).as_posix()
        mp = d / mosaic_name
        infos.append(
            SequenceMosaicInfo(
                seq_dir=d,
                seq_rel=rel,
                mosaic_path=mp,
                ok=mp.exists(),
                error=None if mp.exists() else "Missing mosaic",
            )
        )
    return infos


# -----------------------------
# Main processing loop
# -----------------------------
def run_llm(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    columns: List[str],
    model: str,
    url: str,
    timeout: int,
    delay: float,
) -> None:
    total = len(infos)

    with tqdm(total=total, desc="Generating reports from mosaics", unit="seq") as pbar:
        for info in infos:
            row: Dict[str, Any] = {
                "Timestamp": utc_timestamp(),
                "Model Name": model,
                "sequence_dir": info.seq_rel,
            }

            if not info.ok:
                # No mosaic to analyze
                row.update(
                    {
                        "report_text": "",
                        "catheterized_territory": "Not stated",
                        "vessels_opacified": "[]",
                        "key_findings": "[]",
                        "impression": "[]",
                        "limitations": json.dumps(["Missing mosaic image; report not generated."]),
                        "confidence": 0,
                        "evidence": "[]",
                        "notes": info.error or "Missing mosaic",
                    }
                )
            else:
                try:
                    images = [b64_image(info.mosaic_path)]
                    raw = ollama_chat_with_images(build_prompt(), images, model, url, timeout)
                    parsed = safe_parse_json(raw)
                    if not parsed:
                        raise ValueError("Non-JSON response")

                    # Normalize list-like fields to JSON strings for CSV stability
                    vessels = parsed.get("vessels_opacified", [])
                    findings = parsed.get("key_findings", [])
                    impression = parsed.get("impression", [])
                    limitations = parsed.get("limitations", [])
                    evidence = parsed.get("evidence", [])

                    row.update(
                        {
                            "report_text": parsed.get("report_text", ""),
                            "catheterized_territory": parsed.get("catheterized_territory", "Unclear"),
                            "vessels_opacified": json.dumps(vessels if isinstance(vessels, list) else []),
                            "key_findings": json.dumps(findings if isinstance(findings, list) else []),
                            "impression": json.dumps(impression if isinstance(impression, list) else []),
                            "limitations": json.dumps(limitations if isinstance(limitations, list) else []),
                            "confidence": parsed.get("confidence", 0),
                            "evidence": json.dumps(evidence if isinstance(evidence, list) else []),
                            "notes": parsed.get("notes", ""),
                        }
                    )
                except Exception as e:
                    row.update(
                        {
                            "report_text": "",
                            "catheterized_territory": "Unclear",
                            "vessels_opacified": "[]",
                            "key_findings": "[]",
                            "impression": "[]",
                            "limitations": json.dumps(["Model error; report not generated."]),
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": str(e)[:200],
                        }
                    )

            append_csv_row(out_path, row, columns)
            pbar.update(1)
            if delay:
                time.sleep(delay)


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--frames_subdir", default="frames")
    parser.add_argument("--mosaic_name", default="mosaic.png")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[: args.limit]
    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)

    # 🔥 OUTPUT PATH FIX (as requested)
    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "mosaics_generated_reports.csv"

    columns = [
        "Timestamp",
        "Model Name",
        "sequence_dir",
        "report_text",
        "catheterized_territory",
        "vessels_opacified",
        "key_findings",
        "impression",
        "limitations",
        "confidence",
        "evidence",
        "notes",
    ]

    ensure_csv_header(out_csv, columns)

    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")

    run_llm(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=args.model,
        url=args.url,
        timeout=args.timeout,
        delay=args.delay,
    )

    print("Done ✔ Incremental results preserved.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        raise
