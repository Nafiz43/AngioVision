#!/usr/bin/env python3
"""
03_gen_reports_from_mosaics_flammingo.py.py (Med-Flamingo version)

Stage 2: Discover INNER sequence dirs, read EXISTING mosaic images,
and query Med-Flamingo to generate angiography-style reports.

Med-Flamingo is a medical vision-language model that's better suited
for medical imaging tasks compared to general-purpose VLMs.

Requirements:
    pip install torch torchvision transformers Pillow pandas tqdm huggingface_hub --break-system-packages

CSV behavior:
- Adds: Timestamp, Model Name
- Appends row-by-row
- Output directory: <base_path>_Output/
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_BASE_PATH = Path("/data/Deep_Angiography/DICOM_Sequence_Processed")
DEFAULT_MODEL_NAME = "med-flamingo/med-flamingo"  # Hugging Face model ID
DEFAULT_TIMEOUT_S = 180

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -----------------------------
# Report-generation prompt (Medical focus)
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
  "evidence": [string, ...],             # up to 5 short cues tied to what you see
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
    """Create CSV with header if it doesn't exist."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)


def append_csv_row(out_path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    """Append a single row to CSV."""
    ordered = {c: row.get(c) for c in columns}
    pd.DataFrame([ordered]).to_csv(out_path, mode="a", header=False, index=False)


# -----------------------------
# Directory discovery
# -----------------------------
def find_sequence_dirs(base_path: Path, frames_subdir: str) -> List[Path]:
    """
    Find all sequence directories containing frames_subdir/ with image files.
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
# Med-Flamingo Model Manager
# -----------------------------
class MedFlamingoModel:
    """Wrapper for Med-Flamingo model inference."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """
        Initialize Med-Flamingo model.
        
        Args:
            model_name: Hugging Face model ID (e.g., "med-flamingo/med-flamingo")
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        print(f"Loading Med-Flamingo model: {model_name}...")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model and processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTrying alternative loading method...")
            # Fallback to CPU with lower precision
            self.device = "cpu"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            print("✓ Model loaded on CPU")
    
    def generate_report(self, image_path: Path, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate report from image using Med-Flamingo.
        
        Args:
            image_path: Path to mosaic image
            prompt: Text prompt for report generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated report text
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            # Med-Flamingo expects format: "<image>{prompt}"
            full_prompt = f"<image>{prompt}"
            
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                    temperature=0.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0]
            
            # Remove the prompt from the output (model might echo it)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from model output, handling extra text.
    """
    try:
        return json.loads(text.strip())
    except Exception:
        # Try to extract JSON block
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def utc_timestamp() -> str:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SequenceMosaicInfo:
    """Information about a sequence and its mosaic."""
    seq_dir: Path
    seq_rel: str
    mosaic_path: Path
    ok: bool
    error: Optional[str] = None


def load_mosaics(seq_dirs: List[Path], base_path: Path, mosaic_name: str) -> List[SequenceMosaicInfo]:
    """
    Check which sequence directories have mosaics available.
    """
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
def run_inference(
    infos: List[SequenceMosaicInfo],
    out_path: Path,
    columns: List[str],
    model: MedFlamingoModel,
    model_name: str,
    delay: float,
) -> None:
    """
    Run Med-Flamingo inference on all mosaics and save results to CSV.
    """
    total = len(infos)

    with tqdm(total=total, desc="Generating reports with Med-Flamingo", unit="seq") as pbar:
        for info in infos:
            row: Dict[str, Any] = {
                "Timestamp": utc_timestamp(),
                "Model Name": model_name,
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
                    # Generate report using Med-Flamingo
                    raw = model.generate_report(
                        info.mosaic_path, 
                        BASE_PROMPT,
                        max_tokens=1024
                    )
                    
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
    parser = argparse.ArgumentParser(
        description="Generate angiography reports using Med-Flamingo"
    )
    parser.add_argument("--base_path", type=Path, default=DEFAULT_BASE_PATH,
                        help="Base path for sequence directories")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help="Hugging Face model ID for Med-Flamingo")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between sequences (seconds)")
    parser.add_argument("--frames_subdir", default="frames",
                        help="Subdirectory name containing frames")
    parser.add_argument("--mosaic_name", default="mosaic.png",
                        help="Mosaic filename")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sequences to process")

    args = parser.parse_args()

    # Find sequences
    print(f"Searching for sequences in: {args.base_path}")
    seq_dirs = find_sequence_dirs(args.base_path, args.frames_subdir)
    print(f"Found {len(seq_dirs)} sequence directories")
    
    if args.limit:
        seq_dirs = seq_dirs[: args.limit]
        print(f"Limited to first {args.limit} sequences")
    
    # Load mosaic information
    infos = load_mosaics(seq_dirs, args.base_path, args.mosaic_name)
    available = sum(1 for i in infos if i.ok)
    print(f"Available mosaics: {available}/{len(infos)}")

    # Setup output
    output_root = Path(f"{args.base_path}_Output")
    out_csv = output_root / "medflamingo_generated_reports.csv"

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
    print(f"Output CSV: {out_csv}")

    # Load model
    try:
        model = MedFlamingoModel(args.model, device=args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nPlease ensure you have installed the required packages:")
        print("  pip install torch torchvision transformers Pillow pandas tqdm huggingface_hub --break-system-packages")
        sys.exit(1)

    # Run inference
    print("\nStarting report generation...")
    run_inference(
        infos=infos,
        out_path=out_csv,
        columns=columns,
        model=model,
        model_name=args.model,
        delay=args.delay,
    )

    print("\n✔ Done! Incremental results preserved.")
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — partial results saved.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
