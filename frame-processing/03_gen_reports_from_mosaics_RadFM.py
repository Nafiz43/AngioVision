#!/usr/bin/env python3
"""
Stage 2 only: Discover INNER sequence dirs (same rule as stage 1),
read an EXISTING mosaic image from each sequence dir, and query RadFM
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
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_RADFM_URL = "http://localhost:8000/generate"  # Adjust to your RadFM endpoint

# Default model configuration
DEFAULT_MODEL_NAME = "RadFM"
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
# RadFM-specific helpers
# -----------------------------
def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode image to base64 string for RadFM API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def radfm_generate_report(
    prompt: str, 
    image_path: Path, 
    url: str, 
    timeout: int,
    model_name: str = "RadFM"
) -> str:
    """
    Call RadFM API to generate a report from an image.
    
    RadFM typically expects:
    - image: base64-encoded image or image path
    - prompt/question: text prompt
    - Additional parameters may vary based on your RadFM setup
    
    Adjust the payload structure based on your specific RadFM deployment.
    """
    
    # Method 1: Using base64 encoding (common for API calls)
    image_b64 = encode_image_to_base64(image_path)
    
    payload = {
        "image": image_b64,
        "prompt": prompt,
        "model": model_name,
        "max_tokens": 2048,
        "temperature": 0.0,  # Low temperature for consistent medical reporting
    }
    
    # Alternative payload structures you might need:
    # 
    # Method 2: If RadFM uses a different structure:
    # payload = {
    #     "inputs": {
    #         "image": image_b64,
    #         "text": prompt
    #     },
    #     "parameters": {
    #         "max_new_tokens": 2048,
    #         "temperature": 0.0
    #     }
    # }
    #
    # Method 3: If using file upload:
    # files = {"image": open(image_path, "rb")}
    # data = {"prompt": prompt, "temperature": 0.0}
    # response = requests.post(url, files=files, data=data, timeout=timeout)
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        # Parse response - adjust based on your RadFM API response format
        result = response.json()
        
        # Common response formats:
        # Format 1: {"output": "generated text"}
        if "output" in result:
            return result["output"]
        
        # Format 2: {"generated_text": "..."}
        if "generated_text" in result:
            return result["generated_text"]
        
        # Format 3: {"response": "..."}
        if "response" in result:
            return result["response"]
        
        # Format 4: Direct text response
        if isinstance(result, str):
            return result
        
        # If none of the above, try to extract any text field
        for key in ["text", "result", "prediction", "report"]:
            if key in result:
                return result[key]
        
        raise ValueError(f"Unexpected RadFM response format: {result}")
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"RadFM API request failed: {str(e)}")


def build_prompt() -> str:
    return BASE_PROMPT


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tries to parse JSON robustly even if the model wraps it with extra text.
    """
    try:
        return json.loads(text.strip())
    except Exception:
        # Try to find JSON object within the text
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        
        # Try to find JSON array within the text
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    
    return None


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
def run_radfm(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    columns: List[str],
    model: str,
    url: str,
    timeout: int,
    delay: float,
) -> None:
    total = len(infos)

    with tqdm(total=total, desc="Generating reports from mosaics (RadFM)", unit="seq") as pbar:
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
                    # Call RadFM to generate report
                    raw_response = radfm_generate_report(
                        prompt=build_prompt(),
                        image_path=info.mosaic_path,
                        url=url,
                        timeout=timeout,
                        model_name=model
                    )
                    
                    # Parse the response
                    parsed = safe_parse_json(raw_response)
                    if not parsed:
                        raise ValueError(f"Non-JSON response from RadFM: {raw_response[:200]}")

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
                    error_msg = str(e)[:200]
                    row.update(
                        {
                            "report_text": "",
                            "catheterized_territory": "Unclear",
                            "vessels_opacified": "[]",
                            "key_findings": "[]",
                            "impression": "[]",
                            "limitations": json.dumps(["RadFM error; report not generated."]),
                            "confidence": 0,
                            "evidence": "[]",
                            "notes": error_msg,
                        }
                    )
                    print(f"Error processing {info.seq_rel}: {error_msg}", file=sys.stderr)

            append_csv_row(out_path, row, columns)
            pbar.update(1)
            if delay:
                time.sleep(delay)


# -----------------------------
# Entrypoint
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate angiography reports from mosaic images using RadFM"
    )
    parser.add_argument(
        "--base_path", 
        type=Path, 
        default=DEFAULT_BASE_PATH,
        help="Base path containing sequence directories"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Model name (for tracking purposes)"
    )
    parser.add_argument(
        "--url", 
        type=str, 
        default=DEFAULT_RADFM_URL,
        help="RadFM API endpoint URL"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=DEFAULT_TIMEOUT_S,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.0,
        help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--frames_subdir", 
        default="frames",
        help="Subdirectory name containing frames"
    )
    parser.add_argument(
        "--mosaic_name", 
        default="mosaic.png",
        help="Name of the mosaic image file"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of sequences to process (for testing)"
    )

    args = parser.parse_args()

    # Discover sequence directories
    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    if args.limit:
        seq_dirs = seq_dirs[: args.limit]
    
    # Load mosaic information
    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)

    # Set output path
    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "radfm_generated_reports.csv"

    # Define CSV columns
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

    # Ensure CSV header exists
    ensure_csv_header(out_csv, columns)

    print(f"RadFM Report Generation")
    print(f"=" * 50)
    print(f"Sequences found: {len(seq_dirs)}")
    print(f"Output CSV: {out_csv}")
    print(f"Model: {args.model}")
    print(f"RadFM Endpoint: {args.url}")
    print(f"=" * 50)

    # Run the processing
    run_radfm(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=args.model,
        url=args.url,
        timeout=args.timeout,
        delay=args.delay,
    )

    print("\nDone ✔ Incremental results preserved.")
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
