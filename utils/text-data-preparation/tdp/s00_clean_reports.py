"""
Step 00 — Report cleaning (from 21_cleaning_reports.py).

Cleaning steps, in order:
    1. Unicode / encoding normalization
    2. PHI removal via Microsoft Presidio + custom medical recognizers
       (skipped with a warning if presidio/spaCy are not installed or
       enable_phi_removal is false)
    3. Attestation / signature block removal
    4. Header metadata removal (patient/MRN/DOB/... block at the top)
    5. Medical abbreviation + dictation-shorthand expansion
    6. Punctuation / list-marker / whitespace cleanup
    7. Section-header normalization
    8. Sentence-case conversion

Outputs:
    data_dir/cleaned_reports.csv                       (all rows + cleaned_<col>)
    run_dir/00_clean_reports/cleaning_comparison.docx  (first N, Original|Cleaned)
"""

from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from tdp.common import (
    DocxColumn, build_comparison_docx, detect_report_column, normalize_text,
)

CHUNK_SIZE = 16


# =========================================================
# Presidio setup (optional dependency)
# =========================================================
def build_presidio_engines():
    """Return (analyzer, anonymizer) or raise ImportError."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    custom = [
        PatternRecognizer(
            supported_entity="MRN",
            patterns=[Pattern("mrn_labeled", r"\bMRN[:\s#]*\d{4,10}\b", 0.95)],
            context=["mrn", "medical record"],
        ),
        PatternRecognizer(
            supported_entity="ACCESSION",
            patterns=[
                Pattern("acc_labeled", r"\bACCESSION\s*#?[:\s]*[A-Z0-9\-]+\b", 0.9),
                Pattern("acc_short", r"\bACC\s*#?[:\s]*[A-Z0-9\-]+\b", 0.8),
            ],
            context=["accession"],
        ),
        PatternRecognizer(
            supported_entity="DOB",
            patterns=[Pattern(
                "dob_labeled",
                r"\bDOB[:\s]*\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b", 0.95)],
            context=["dob", "date of birth"],
        ),
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern("phone_paren", r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}", 0.9),
                Pattern("phone_dash", r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", 0.85),
            ],
        ),
        PatternRecognizer(
            supported_entity="FACILITY",
            patterns=[Pattern(
                "med_center",
                r"\b[A-Z][A-Za-z\.]+(?:\s+[A-Z][A-Za-z\.]+){0,4}\s+"
                r"(?:Medical Center|Hospital|Clinic|Health System|"
                r"Medical Centre|General Hospital)\b", 0.7)],
        ),
    ]
    for rec in custom:
        analyzer.registry.add_recognizer(rec)
    return analyzer, AnonymizerEngine()


def remove_phi(text: str, analyzer, anonymizer) -> str:
    from presidio_anonymizer.entities import OperatorConfig

    entities = [
        "PERSON", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS",
        "LOCATION", "US_SSN", "CREDIT_CARD", "IP_ADDRESS", "URL",
        "MRN", "ACCESSION", "DOB", "FACILITY",
    ]
    results = analyzer.analyze(text=text, language="en", entities=entities)
    placeholders = {
        "PERSON": "[PERSON]", "DATE_TIME": "[DATE]", "PHONE_NUMBER": "[PHONE]",
        "EMAIL_ADDRESS": "[EMAIL]", "LOCATION": "[LOCATION]", "US_SSN": "[SSN]",
        "MRN": "[MRN]", "ACCESSION": "[ACCESSION]", "DOB": "[DOB]",
        "FACILITY": "[FACILITY]", "DEFAULT": "[REDACTED]",
    }
    operators = {k: OperatorConfig("replace", {"new_value": v})
                 for k, v in placeholders.items()}
    return anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators
    ).text


# =========================================================
# Rule-based cleaning (no external deps)
# =========================================================
_ATTESTATION_HEADS = [
    r"^\s*ATTEST(?:ATION)?\b", r"^\s*SIGN-?OFF\b", r"^\s*SIGNATURE\b",
    r"\bPreliminary\s+Report\s+Electronically\s+Signed\b",
    r"\bFinal\s+Report\s+Electronically\s+Signed\b",
    r"\bReport\s+Electronically\s+Signed\b",
    r"\bElectronically\s+signed\s+by\b", r"\bE-?signed\s+by\b",
    r"\bDigitally\s+signed\s+by\b", r"\bSigned\s+by\s*[:\-]",
    r"\bDictated\s+by\s*[:\-]", r"\bTranscribed\s+by\s*[:\-]",
    r"\bI\s+have\s+(?:personally\s+)?(?:reviewed|examined|seen)\b",
    r"\bI\s+(?:was|have\s+been)\s+present\b",
    r"\bI\s+agree\s+with\s+the\s+(?:findings|above|report|plan)\b",
    r"\bAttending\s+(?:attestation|note|statement)\b",
    r"^\s*---\s*END\s+OF\s+REPORT\s*---\s*$", r"\bDate/Time\s+signed\b",
]

_BODY_START_HEADERS = [
    r"PROCEDURE", r"INDICATIONS?", r"HISTORY", r"CLINICAL\s+HISTORY",
    r"TECHNIQUE", r"FINDINGS", r"CORONARY\s+FINDINGS", r"HEMODYNAMICS",
    r"IMPRESSION", r"COMPARISON", r"EXAMINATION",
]

_ABBREVIATIONS = [
    # Anatomy / vessels — coronary
    (r"\bLM\b", "left main"),
    (r"\bLAD\b", "left anterior descending"),
    (r"\bLCX\b", "left circumflex"),
    (r"\bRCA\b", "right coronary artery"),
    (r"\bPDA\b", "posterior descending artery"),
    (r"\bPLV\b", "posterior left ventricular"),
    (r"\bOM\d*\b", "obtuse marginal"),
    (r"\bD\d+\b", "diagonal branch"),
    (r"\bRI\b", "ramus intermedius"),
    # Anatomy / vessels — non-coronary / interventional
    (r"\bGDA\b", "gastroduodenal artery"),
    (r"\bRAS\b", "renal artery stenosis"),
    (r"\bSMA\b", "superior mesenteric artery"),
    (r"\bIMA\b", "inferior mesenteric artery"),
    (r"\bCFA\b", "common femoral artery"),
    (r"\bSFA\b", "superficial femoral artery"),
    (r"\bIVC\b", "inferior vena cava"),
    (r"\bSVC\b", "superior vena cava"),
    (r"\bAV\s+shunting\b", "arteriovenous shunting"),
    # Function / hemodynamics
    (r"\bLVEF\b", "left ventricular ejection fraction"),
    (r"\bLVEDP\b", "left ventricular end-diastolic pressure"),
    (r"\bLV\s+gram\b", "left ventriculogram"),
    (r"\bAo\b", "aorta"),
    (r"\bMAP\b", "mean arterial pressure"),
    (r"\bTIMI\b", "thrombolysis in myocardial infarction grade"),
    # Conditions / pathology
    (r"\bHTN\b", "hypertension"),
    (r"\bDM2\b", "type 2 diabetes mellitus"),
    (r"\bDM\b", "diabetes mellitus"),
    (r"\bHLD\b", "hyperlipidemia"),
    (r"\bCAD\b", "coronary artery disease"),
    (r"\bCHF\b", "congestive heart failure"),
    (r"\bMI\b", "myocardial infarction"),
    (r"\bCP\b", "chest pain"),
    (r"\bSOB\b", "shortness of breath"),
    (r"\bHCC\b", "hepatocellular carcinoma"),
    (r"\bUGI\s+bleed\b", "upper gastrointestinal bleed"),
    (r"\bAKI\b", "acute kidney injury"),
    (r"\bESRD\b", "end-stage renal disease"),
    (r"\bNSTEMI\b", "non-ST-elevation myocardial infarction"),
    (r"\bSTEMI\b", "ST-elevation myocardial infarction"),
    (r"\bPE\b", "pulmonary embolism"),
    (r"\bDVT\b", "deep vein thrombosis"),
    # Procedures / techniques
    (r"\bPCI\b", "percutaneous coronary intervention"),
    (r"\bCABG\b", "coronary artery bypass grafting"),
    (r"\bIR\b", "interventional radiology"),
    (r"\bDSA\b", "digital subtraction angiography"),
    (r"\bEGD\b", "esophagogastroduodenoscopy"),
    (r"\bMRA\b", "magnetic resonance angiography"),
    (r"\bTACE\b", "transarterial chemoembolization"),
    (r"\bTARE\b", "transarterial radioembolization"),
    # Devices / stents
    (r"\bDES\b", "drug-eluting stent"),
    (r"\bBMS\b", "bare-metal stent"),
    # Drugs
    (r"\bASA\b", "aspirin"),
    # Measurements / units
    (r"\bEBL\b", "estimated blood loss"),
    (r"\bmGy\b", "milligray"),
    (r"\bcGycm", "centigray-cm"),
    (r"\bmmHg\b", "millimeters of mercury"),
    (r"(\d+)\s*French\b", r"\1 French (catheter size)"),
    (r"(\d+)\s*F\b(?!\w)", r"\1 French (catheter size)"),
    # Couinaud liver segments
    (r"\bSegment\s+([1-8])\b", r"Couinaud liver segment \1"),
    (r"\bSeg\.?\s*([1-8])\b", r"Couinaud liver segment \1"),
    # Standalone LV (after LV-compound terms above)
    (r"\bLV\b", "left ventricular"),
    # Dictation shorthand
    (r"\bw/o\b", "without"),
    (r"\bw/(?=\s|$)", "with"),
    (r"\bh/o\b", "history of"),
    (r"\bs/p\b", "status post"),
    (r"\bp/w\b", "presenting with"),
    (r"\by/?o\b", "year old"),
    (r"\bpk-?yr\b", "pack-year"),
    (r"\bhx\b", "history"),
    (r"\bf/u\b", "follow up"),
    (r"\bwk(s)?\b", r"week\1"),
    (r"\bqhs\b", "at bedtime"),
    (r"\bpt\b", "patient"),
]

_SECTION_HEADERS = [
    "PROCEDURE", "INDICATIONS", "HEMODYNAMICS", "CORONARY FINDINGS",
    "FINDINGS", "IMPRESSION", "PLAN", "HISTORY", "TECHNIQUE",
    "COMPARISON", "COMPLICATIONS",
]


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for u, a in {
        "–": "-", "—": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "…": "...",
        "°": " degrees ", "±": "+/-", "×": "x",
        "→": "->", " ": " ",
    }.items():
        text = text.replace(u, a)
    return text


def remove_attestation(text: str) -> str:
    earliest = len(text)
    for pat in _ATTESTATION_HEADS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            earliest = min(earliest, m.start())
    return text[:earliest].rstrip()


def remove_header_metadata(text: str) -> str:
    pattern = r"(?im)^\s*(?:" + "|".join(_BODY_START_HEADERS) + r")\s*[:\-]"
    match = re.search(pattern, text)
    return text[match.start():].lstrip() if match else text


def expand_abbreviations(text: str) -> str:
    for pat, repl in _ABBREVIATIONS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def clean_symbols_and_lists(text: str) -> str:
    text = re.sub(r"^[\s]*[\*•\-–—]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"-{3,}.*?-{3,}", "", text)
    text = re.sub(r"[=_]{3,}", "", text)
    text = re.sub(r"([!?.,;:])\1{1,}", r"\1", text)
    return text


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_section_headers(text: str) -> str:
    for h in _SECTION_HEADERS:
        text = re.sub(rf"(?im)^\s*{h}\s*[:\-]?\s*$", f"{h}:", text)
        text = re.sub(rf"(?im)^\s*{h}\s*[:\-]\s*", f"{h}: ", text)
    header_pat = "|".join(re.escape(h) for h in _SECTION_HEADERS)
    text = re.sub(rf"(?m)(?<!\n\n)^((?:{header_pat}):)", r"\n\1", text)
    return text.lstrip("\n")


def sentence_case(text: str) -> str:
    """Lowercase everything except ALL-CAPS words; capitalize sentence starts."""
    def _selective_lower(word: str) -> str:
        alpha_only = re.sub(r"[^A-Za-z]", "", word)
        if len(alpha_only) >= 2 and alpha_only.isupper():
            return word
        return word.lower()

    result = re.sub(r"[^\s]+", lambda m: _selective_lower(m.group(0)), text)

    def _cap(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    result = re.sub(r"^(\s*)([a-z])", _cap, result)
    result = re.sub(r"([.!?]\s+|\n+\s*)([a-z])", _cap, result)
    result = re.sub(r"\[([a-z_][a-z0-9_]*)\]",
                    lambda m: f"[{m.group(1).upper()}]", result)
    return result


def clean_report(text: str, analyzer=None, anonymizer=None) -> str:
    """Full cleaning pipeline; PHI removal only if engines are provided."""
    text = normalize_unicode(text)
    if analyzer is not None and anonymizer is not None:
        text = remove_phi(text, analyzer, anonymizer)
    text = remove_attestation(text)
    text = remove_header_metadata(text)
    text = expand_abbreviations(text)
    text = clean_symbols_and_lists(text)
    text = collapse_whitespace(text)
    text = normalize_section_headers(text)
    text = sentence_case(text)
    return collapse_whitespace(text)


# =========================================================
# Parallel worker plumbing
# =========================================================
# Presidio engines are heavy (spaCy model load) and unpicklable — built once
# per worker process in the initializer.
_WORKER_ANALYZER = None
_WORKER_ANONYMIZER = None


def _worker_init(enable_phi: bool) -> None:
    global _WORKER_ANALYZER, _WORKER_ANONYMIZER
    if enable_phi:
        _WORKER_ANALYZER, _WORKER_ANONYMIZER = build_presidio_engines()


def _clean_one(text: str) -> str:
    if not text:
        return ""
    return clean_report(text, _WORKER_ANALYZER, _WORKER_ANONYMIZER)


# =========================================================
# Step entry point
# =========================================================
def run(cfg, run_dir: Path, data_dir: Path) -> Dict:
    step_dir = run_dir / "00_clean_reports"

    df = pd.read_csv(cfg.input_csv, dtype=str, keep_default_na=False)
    report_col = cfg.report_column or detect_report_column(list(df.columns))
    if report_col not in df.columns:
        raise KeyError(f"Column '{report_col}' not in {cfg.input_csv} "
                       f"(available: {list(df.columns)})")
    if cfg.max_reports and cfg.max_reports > 0:
        df = df.head(cfg.max_reports)

    originals = [normalize_text(t) for t in df[report_col]]

    # PHI removal is optional: fail soft if presidio isn't installed.
    enable_phi = cfg.enable_phi_removal
    phi_status = "enabled"
    if enable_phi:
        try:
            import presidio_analyzer  # noqa: F401
            import presidio_anonymizer  # noqa: F401
        except ImportError:
            enable_phi = False
            phi_status = "SKIPPED — presidio not installed"
            print("[00] WARNING: enable_phi_removal is true but presidio is "
                  "not installed; continuing WITHOUT PHI removal.")
    else:
        phi_status = "disabled via config"

    cleaned = [""] * len(originals)
    with ProcessPoolExecutor(max_workers=cfg.workers,
                             initializer=_worker_init,
                             initargs=(enable_phi,)) as ex:
        results = ex.map(_clean_one, originals, chunksize=CHUNK_SIZE)
        for i, text in enumerate(tqdm(results, total=len(originals),
                                      desc="[00] Cleaning", unit="report")):
            cleaned[i] = text

    cleaned_col = f"cleaned_{report_col}"
    df[cleaned_col] = cleaned
    data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_csv = data_dir / "cleaned_reports.csv"
    df.to_csv(cleaned_csv, index=False)

    n_docx = min(cfg.num_docx_samples, len(df))
    docx_path = build_comparison_docx(
        rows=[{"original": originals[i], "cleaned": cleaned[i]}
              for i in range(n_docx)],
        columns=[
            DocxColumn("original", "Original Report", 4.9),
            DocxColumn("cleaned", "Cleaned Report", 4.9),
        ],
        title="Angiographic Report Cleaning — Original vs. Cleaned",
        subtitle=(f"PHI removal: {phi_status} · attestation stripped · "
                  f"abbreviations expanded · sentence-cased · "
                  f"first {n_docx} of {len(df)} reports"),
        out_path=step_dir / "cleaning_comparison.docx",
        landscape=True,
    )

    summary = {
        "reports_cleaned": len(df),
        "report_column": report_col,
        "phi_removal": phi_status,
        "cleaned_csv": str(cleaned_csv),
        "comparison_docx": str(docx_path),
    }
    print(f"[00] {summary}")
    return summary
