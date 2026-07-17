"""Shared prompts and question configurations used across pipeline scripts."""

# ---------- Questions with options (used by CLIP / BioMedCLIP / MedCLIP) ----------

QUESTIONS_WITH_OPTIONS = [
    {
        "question": "Which artery is catheterized?",
        "options": [
            "No catheter visible",
            "Femoral artery",
            "Radial artery",
            "Brachial artery",
            "Carotid artery",
            "Vertebral artery",
            "Coronary artery",
            "Renal artery",
            "Mesenteric artery",
            "Iliac artery",
            "Unclear or other artery",
        ],
    },
    {
        "question": "Is variant anatomy present?",
        "options": [
            "No variant anatomy visible",
            "Yes, variant anatomy present",
            "Unclear if variant anatomy present",
        ],
    },
    {
        "question": "Is there evidence of hemorrhage or contrast extravasation in this sequence?",
        "options": [
            "No hemorrhage or extravasation",
            "Yes, hemorrhage present",
            "Yes, contrast extravasation present",
            "Yes, both hemorrhage and extravasation present",
            "Unclear",
        ],
    },
    {
        "question": "Is there evidence of arterial or venous dissection?",
        "options": [
            "No dissection visible",
            "Yes, arterial dissection present",
            "Yes, venous dissection present",
            "Unclear if dissection present",
        ],
    },
    {
        "question": "Is stenosis present in any visualized vessel?",
        "options": [
            "No stenosis visible",
            "Yes, mild stenosis present",
            "Yes, moderate stenosis present",
            "Yes, severe stenosis present",
            "Unclear if stenosis present",
        ],
    },
    {
        "question": "Is an endovascular stent visible in this sequence?",
        "options": [
            "No stent visible",
            "Yes, stent visible",
            "Unclear if stent present",
        ],
    },
]


# ---------- YES/NO VLM prompt ----------

def build_yesno_prompt(question: str) -> str:
    """Build the standard YES/NO prompt used by Ollama and Bedrock pipelines."""
    return (
        "You are analyzing a medical angiography mosaic image.\n"
        "Answer the question using only visible image evidence.\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Respond with exactly one word: YES or NO\n"
        "- Do not explain your answer\n"
        "- Do not add punctuation or extra words\n"
        "- If the finding is not clearly visible, answer NO\n"
        "- If the image is ambiguous, uncertain, low-quality, incomplete, or cannot confirm the finding, answer NO\n"
        "- Only answer YES when the finding is clearly supported by the image\n"
        "- Output must be exactly YES or NO\n"
    )


# ---------- Report generation prompt ----------

REPORT_GENERATION_PROMPT = """ROLE
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
- Impression: 1-3 bullets, image-grounded, with uncertainty noted.
- Limitations: e.g., "single mosaic; phase/projection unclear; limited field of view..."

OUTPUT FORMAT (JSON ONLY - no extra text)
Return a single JSON object with EXACTLY these keys:
{
  "report_text": string,                 # concise, angiography-style paragraph(s)
  "catheterized_territory": string,      # one of: specific territory / "Unclear" / "Conflicting"
  "vessels_opacified": [string, ...],    # list; empty list if none can be determined
  "key_findings": [string, ...],         # list of short findings; include "None visible" if appropriate
  "impression": [string, ...],           # 1-3 bullets; can include uncertainty
  "limitations": [string, ...],          # short bullets
  "confidence": integer,                 # 0-100 overall confidence based on image clarity
  "evidence": [string, ...],             # up to 5 short cues tied to what you see (e.g., "contrast pooling outside vessel", "early venous filling")
  "notes": string                        # optional; use "" if no extra notes
}

QUALITY CHECK BEFORE YOU ANSWER
- Is every claim supported by something visible?
- Did you avoid guessing and PHI?
- Is the output valid JSON and only JSON?
"""


# ---------- Report response columns ----------

REPORT_CSV_COLUMNS = [
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


def build_report_row_from_parsed(parsed: dict) -> dict:
    """Extract and JSON-serialise report fields from a parsed LLM response."""
    import json

    vessels = parsed.get("vessels_opacified", [])
    findings = parsed.get("key_findings", [])
    impression = parsed.get("impression", [])
    limitations = parsed.get("limitations", [])
    evidence = parsed.get("evidence", [])

    return {
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


def build_report_error_row(error_msg: str, source: str = "Model") -> dict:
    """Return placeholder report fields for an error case."""
    import json

    return {
        "report_text": "",
        "catheterized_territory": "Unclear",
        "vessels_opacified": "[]",
        "key_findings": "[]",
        "impression": "[]",
        "limitations": json.dumps([f"{source} error; report not generated."]),
        "confidence": 0,
        "evidence": "[]",
        "notes": error_msg,
    }


def build_report_missing_mosaic_row(error: str = "Missing mosaic") -> dict:
    """Return placeholder report fields when the mosaic is missing."""
    import json

    return {
        "report_text": "",
        "catheterized_territory": "Not stated",
        "vessels_opacified": "[]",
        "key_findings": "[]",
        "impression": "[]",
        "limitations": json.dumps(["Missing mosaic image; report not generated."]),
        "confidence": 0,
        "evidence": "[]",
        "notes": error,
    }
