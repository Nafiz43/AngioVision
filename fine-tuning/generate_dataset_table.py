#!/usr/bin/env python3
"""
generate_dataset_table.py

Generate Table 1 (Dataset Characteristics) for the AngioVision contrastive-learning
paper with REAL demographic values, and write it as a compilable LaTeX file.

Where the numbers come from (auto-discovered from the AngioVision pipeline)
--------------------------------------------------------------------------
TRAINING cohort   The study/sequence identifiers in the training meta CSV, i.e.
                  the --meta_csv passed to the custom_framework_train_* /
                  finetune_clip_on_mosaics_* scripts. This is the consolidated
                  metadata CSV produced by
                  utils/04_generate_consolidated_metadata.py.
                  Columns used: an accession column + "SOPInstanceUIDs"
                  (a comma-separated list of SOPInstanceUIDs per study).

TEST cohort       The ground-truth validation CSV resolved from
                  configs/settings.py:VALIDATION_CSV -- the same file that
                  fine-tuning/calculate_score.py scores predictions against.
                  Columns used: "SOPInstanceUID" (+ "Accession").

DEMOGRAPHICS      The DICOM metadata SQLite DB (table dicom_files) built by
                  utils/metadata_db (path = ingestion/config.py:SQLITE_DB).
                  Age, sex, study date, referring physician, manufacturer,
                  contrast agent, frame count (and institution, if the column
                  exists) are joined to each cohort by SOPInstanceUID.

The training and test CSVs carry only identifiers -- no age/sex -- so every
demographic value is looked up in the SQLite DB. The per-institution breakdown
of the test set needs an institution column in dicom_files; DICOM
InstitutionName (0008,0080) is NOT extracted by the current parser, so the
script falls back to --institution-col (default tries institution_name, then
station_name) and prints a warning if none is present.

Usage
-----
    python3 generate_dataset_table.py                       # all defaults
    python3 generate_dataset_table.py \
        --meta_csv    /data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_GT.csv \
        --test_csv    /data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv \
        --db          /data/Deep_Angiography/AngioVision/dicom_staging.db \
        --reports_csv /data/Deep_Angiography/Reports/Report_List_v01_01.csv \
        --out_tex     output/table1_dataset.tex

The script depends only on the standard library + pandas + numpy.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Repo-relative defaults ──────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent  # .../AngioVision

DEFAULT_META_CSV = "/data/Deep_Angiography/DICOM-metadata-stats/consolidated_metadata_GT.csv"
DEFAULT_DB = "/data/Deep_Angiography/AngioVision/dicom_staging.db"
DEFAULT_REPORTS_CSV = "/data/Deep_Angiography/Reports/Report_List_v01_01.csv"
DEFAULT_SETTINGS = str(_REPO_ROOT / "configs" / "settings.py")
DEFAULT_OUT_TEX = str(_THIS.parent / "output" / "table1_dataset.tex")

# Candidate column names (first match wins).
SOP_LIST_COLS = ["SOPInstanceUIDs", "SOPInstanceUID", "UID", "sopinstanceuid"]
SOP_SINGLE_COLS = ["SOPInstanceUID", "SOPInstanceUIDs", "UID", "sopinstanceuid"]
ACC_COLS = ["AccessionNumber", "Accession", "Anon Acc #", "accession", "accessionnumber"]
REPORT_ACC_COLS = ["Anon Acc #", "AccessionNumber", "Accession", "accession"]
REPORT_TEXT_COLS = ["radrpt", "Report", "report", "report_text"]

# DB columns we want, mapped from dicom_files -> internal name.
DB_FIELDS = {
    "sop_instance_uid": "sop",
    "accession_number": "accession",
    "study_instance_uid": "study_uid",
    "patient_id": "patient_id",
    "patient_age": "patient_age",
    "patient_sex": "patient_sex",
    "study_date": "study_date",
    "referring_physician": "referring_physician",
    "manufacturer": "manufacturer",
    "frame_count": "frame_count",
    "contrast_bolus_agent": "contrast_bolus_agent",
    "station_name": "station_name",
}
# Candidate institution columns, in order of preference. station_name is a
# per-machine identifier, NOT a true institution, so it is deliberately excluded
# from auto-detection; pass --institution-col station_name to use it as a proxy.
INSTITUTION_COLS = ["institution_name", "institutional_department_name"]

_AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([dwmyDWMY])?")


# ═══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════════

def pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate present in df.columns (case-insensitive)."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower:
            return lower[key]
    return None


def norm_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def parse_sop_list(val) -> List[str]:
    """Split a 'SOPInstanceUIDs' cell (comma/semicolon/whitespace separated)."""
    s = norm_id(val)
    if not s:
        return []
    parts = re.split(r"[,;\s]+", s)
    return [p for p in (norm_id(p) for p in parts) if p]


def parse_age_years(raw) -> Optional[float]:
    """Convert a DICOM PatientAge string ('045Y', '009M', '32', ...) to years."""
    s = norm_id(raw)
    if not s:
        return None
    m = _AGE_RE.search(s)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "Y").upper()
    years = {
        "Y": num,
        "M": num / 12.0,
        "W": num / 52.1775,
        "D": num / 365.25,
    }.get(unit, num)
    if years <= 0 or years > 120:
        return None
    return years


def normalize_sex(raw) -> str:
    s = norm_id(raw).upper()
    if s.startswith("F"):
        return "F"
    if s.startswith("M"):
        return "M"
    return "Other/Unknown"


def parse_year(raw) -> Optional[int]:
    s = norm_id(raw)
    if len(s) >= 4 and s[:4].isdigit():
        y = int(s[:4])
        if 1900 <= y <= 2100:
            return y
    return None


def to_int(raw) -> Optional[int]:
    s = norm_id(raw)
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def latex_escape(text: str) -> str:
    if text is None:
        return ""
    out = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Cohort identifier loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_train_ids(meta_csv: Path) -> Tuple[set, set]:
    """Return (sop_uids, accessions) referenced by the training meta CSV."""
    df = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    sop_col = pick_column(df, SOP_LIST_COLS)
    acc_col = pick_column(df, ACC_COLS)
    if sop_col is None:
        raise SystemExit(
            f"[ERROR] No SOPInstanceUID(s) column in {meta_csv}. "
            f"Looked for {SOP_LIST_COLS}; found {list(df.columns)}"
        )

    sops: set = set()
    for cell in df[sop_col]:
        sops.update(parse_sop_list(cell))

    accs: set = set()
    if acc_col is not None:
        for cell in df[acc_col]:
            # AccessionNumber may itself be a comma-joined list (see 04_generate_*).
            for a in parse_sop_list(cell):
                accs.add(a)

    print(f"[TRAIN] {meta_csv}")
    print(f"[TRAIN]   sop column='{sop_col}', accession column='{acc_col}'")
    print(f"[TRAIN]   {len(sops):,} unique SOPInstanceUIDs, {len(accs):,} unique accessions")
    return sops, accs


def load_test_ids(test_csv: Path) -> Tuple[set, set]:
    """Return (sop_uids, accessions) referenced by the GT validation CSV."""
    df = pd.read_csv(test_csv, dtype=str, keep_default_na=False)
    sop_col = pick_column(df, SOP_SINGLE_COLS)
    acc_col = pick_column(df, ACC_COLS)
    if sop_col is None:
        raise SystemExit(
            f"[ERROR] No SOPInstanceUID column in {test_csv}. "
            f"Looked for {SOP_SINGLE_COLS}; found {list(df.columns)}"
        )

    sops = {norm_id(v) for v in df[sop_col] if norm_id(v)}
    accs: set = set()
    if acc_col is not None:
        accs = {norm_id(v) for v in df[acc_col] if norm_id(v)}

    print(f"[TEST]  {test_csv}")
    print(f"[TEST]    sop column='{sop_col}', accession column='{acc_col}'")
    print(f"[TEST]    {len(sops):,} unique SOPInstanceUIDs, {len(accs):,} unique accessions")
    return sops, accs


def resolve_validation_csv(settings_py: Path) -> str:
    """Load VALIDATION_CSV from a settings.py module (same as calculate_score.py)."""
    spec = importlib.util.spec_from_file_location("angio_settings", settings_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[ERROR] Could not load settings module: {settings_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "VALIDATION_CSV"):
        raise SystemExit(f"[ERROR] VALIDATION_CSV not defined in {settings_py}")
    return str(module.VALIDATION_CSV)


# ═══════════════════════════════════════════════════════════════════════════════
# Demographics lookup (SQLite dicom_files)
# ═══════════════════════════════════════════════════════════════════════════════

def db_columns(con: sqlite3.Connection, table: str = "dicom_files") -> List[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def fetch_demographics(
    db_path: Path,
    sop_uids: Iterable[str],
    institution_col: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Pull one row per SOPInstanceUID from dicom_files for the given SOP UIDs."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        present = set(db_columns(con))
        if "sop_instance_uid" not in present:
            raise SystemExit(
                f"[ERROR] Table dicom_files in {db_path} has no sop_instance_uid column."
            )

        # Decide which institution column to use.
        inst_col: Optional[str] = None
        if institution_col:
            inst_col = institution_col if institution_col in present else None
            if inst_col is None:
                print(f"[WARN] --institution-col '{institution_col}' not in dicom_files.")
        if inst_col is None:
            for cand in INSTITUTION_COLS:
                if cand in present:
                    inst_col = cand
                    break

        select_cols = [c for c in DB_FIELDS if c in present]
        if inst_col and inst_col not in select_cols:
            select_cols.append(inst_col)

        sop_list = [s for s in sop_uids if s]
        rows: List[sqlite3.Row] = []
        CHUNK = 800
        col_sql = ", ".join(select_cols)
        for i in range(0, len(sop_list), CHUNK):
            chunk = sop_list[i : i + CHUNK]
            ph = ",".join("?" * len(chunk))
            q = (
                f"SELECT {col_sql} FROM dicom_files "
                f"WHERE sop_instance_uid IN ({ph})"
            )
            rows.extend(con.execute(q, chunk).fetchall())
    finally:
        con.close()

    if not rows:
        return pd.DataFrame(), inst_col

    recs = []
    for r in rows:
        rec = {DB_FIELDS.get(k, k): r[k] for k in r.keys()}
        if inst_col:
            rec["institution"] = norm_id(r[inst_col])
        recs.append(rec)
    df = pd.DataFrame(recs)

    # Ensure every column the derived block reads exists (defensive: a DB may
    # lack an optional column). Missing ones become empty strings.
    for needed in (
        "sop", "accession", "study_uid", "patient_id", "patient_age",
        "patient_sex", "study_date", "frame_count", "contrast_bolus_agent",
        "referring_physician", "manufacturer",
    ):
        if needed not in df.columns:
            df[needed] = ""

    # Derived, cleaned columns.
    df["sop"] = df["sop"].map(norm_id)
    df = df.drop_duplicates(subset=["sop"]).reset_index(drop=True)
    df["age_years"] = df.get("patient_age", "").map(parse_age_years)
    df["sex"] = df.get("patient_sex", "").map(normalize_sex)
    df["year"] = df.get("study_date", "").map(parse_year)
    df["frames"] = df.get("frame_count", "").map(to_int)
    df["has_contrast"] = df.get("contrast_bolus_agent", "").map(lambda v: bool(norm_id(v)))
    df["accession"] = df.get("accession", "").map(norm_id)
    df["patient_id"] = df.get("patient_id", "").map(norm_id)
    df["study_uid"] = df.get("study_uid", "").map(norm_id)
    df["referring_physician"] = df.get("referring_physician", "").map(norm_id)
    df["manufacturer"] = df.get("manufacturer", "").map(norm_id)
    if "institution" not in df.columns:
        df["institution"] = ""
    return df, inst_col


def load_report_accessions(reports_csv: Path) -> set:
    """Return the set of normalized accessions that have a non-empty report."""
    if not reports_csv.exists():
        print(f"[WARN] reports_csv not found ({reports_csv}); 'paired reports' row -> --")
        return set()
    df = pd.read_csv(reports_csv, dtype=str, keep_default_na=False)
    acc_col = pick_column(df, REPORT_ACC_COLS)
    txt_col = pick_column(df, REPORT_TEXT_COLS)
    if acc_col is None:
        print(f"[WARN] no accession column in {reports_csv}; 'paired reports' row -> --")
        return set()
    accs: set = set()
    for _, row in df.iterrows():
        acc = norm_id(row[acc_col])
        if not acc:
            continue
        if txt_col is None or norm_id(row[txt_col]):
            accs.add(acc)
    print(f"[REPORTS] {len(accs):,} accessions with reports (col='{acc_col}', text='{txt_col}')")
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "--"


def _fmt_pct(k: int, total: int) -> str:
    if not total:
        return "--"
    return f"{_fmt_int(k)} ({100.0 * k / total:.1f})"


def _age_stats(ages: pd.Series) -> Dict[str, str]:
    vals = ages.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return {"mean_sd": "--", "median_iqr": "--", "range": "--"}
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    q1, med, q3 = (float(x) for x in np.percentile(vals, [25, 50, 75]))
    return {
        "mean_sd": f"{mean:.1f}\\,$\\pm$\\,{sd:.1f}",
        "median_iqr": f"{med:.0f} [{q1:.0f}--{q3:.0f}]",
        "range": f"{vals.min():.0f}--{vals.max():.0f}",
    }


def _frames_per_seq(frames: pd.Series) -> str:
    vals = frames.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return "--"
    med = float(np.median(vals))
    return f"{med:.0f} [{vals.min():.0f}--{vals.max():.0f}]"


def cohort_stats(df: pd.DataFrame, report_accs: set) -> Dict[str, str]:
    """Compute every Table-1 row value for one cohort sub-frame."""
    if df.empty:
        return {}

    # Patient-level dedup for age/sex so multi-sequence patients are not double-counted.
    pat = df.drop_duplicates(subset=["patient_id"]) if df["patient_id"].any() else df
    n_patients = df["patient_id"].replace("", np.nan).nunique()
    n_studies = df["study_uid"].replace("", np.nan).nunique()
    n_sequences = df["sop"].nunique()
    n_frames = int(df["frames"].dropna().sum()) if df["frames"].notna().any() else None

    sex_counts = pat["sex"].value_counts().to_dict()
    n_sex_total = int(sum(sex_counts.values()))

    years = df["year"].dropna()
    year_range = f"{int(years.min())}--{int(years.max())}" if not years.empty else "--"

    n_phys = df["referring_physician"].replace("", np.nan).nunique()

    cohort_accs = {a for a in df["accession"].unique() if a}
    n_reports = len(cohort_accs & report_accs) if report_accs else None

    n_contrast = int(df["has_contrast"].sum())

    age = _age_stats(pat["age_years"])
    stats = {
        "patients": _fmt_int(n_patients),
        "studies": _fmt_int(n_studies),
        "sequences": _fmt_int(n_sequences),
        "frames": _fmt_int(n_frames) if n_frames is not None else "--",
        "reports": _fmt_int(n_reports) if n_reports is not None else "--",
        "age_mean_sd": age["mean_sd"],
        "age_median_iqr": age["median_iqr"],
        "age_range": age["range"],
        "female": _fmt_pct(int(sex_counts.get("F", 0)), n_sex_total),
        "male": _fmt_pct(int(sex_counts.get("M", 0)), n_sex_total),
        "sex_other": _fmt_pct(int(sex_counts.get("Other/Unknown", 0)), n_sex_total),
        "year_range": year_range,
        "physicians": _fmt_int(n_phys),
        "frames_per_seq": _frames_per_seq(df["frames"]),
        "contrast": _fmt_pct(n_contrast, n_sequences),
    }
    return stats


def institution_rows(df_test: pd.DataFrame, inst_col: Optional[str]) -> List[List[str]]:
    """Build the Panel-B rows (one per institution in the test cohort + Total)."""
    def _row(label: str, sub: pd.DataFrame) -> List[str]:
        pat = sub.drop_duplicates(subset=["patient_id"]) if sub["patient_id"].any() else sub
        sex_counts = pat["sex"].value_counts().to_dict()
        n_sex = int(sum(sex_counts.values()))
        years = sub["year"].dropna()
        yr = f"{int(years.min())}--{int(years.max())}" if not years.empty else "--"
        age = _age_stats(pat["age_years"])
        frames = int(sub["frames"].dropna().sum()) if sub["frames"].notna().any() else None
        return [
            label,
            _fmt_int(sub["patient_id"].replace("", np.nan).nunique()),
            _fmt_int(sub["study_uid"].replace("", np.nan).nunique()),
            _fmt_int(sub["sop"].nunique()),
            _fmt_int(frames) if frames is not None else "--",
            _fmt_pct(int(sex_counts.get("F", 0)), n_sex),
            age["mean_sd"],
            yr,
        ]

    rows: List[List[str]] = []
    if inst_col and df_test["institution"].replace("", np.nan).notna().any():
        order = (
            df_test.assign(_i=df_test["institution"].replace("", "Unknown"))
            .groupby("_i")["sop"].nunique().sort_values(ascending=False).index.tolist()
        )
        for inst in order:
            sub = df_test[df_test["institution"].replace("", "Unknown") == inst]
            rows.append(_row(latex_escape(inst), sub))
    else:
        rows.append(_row("Institution not recorded$^{\\dagger}$", df_test))

    rows.append(_row("\\textbf{Total}", df_test))
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX emission
# ═══════════════════════════════════════════════════════════════════════════════

def _cells(label: str, tr: Dict[str, str], te: Dict[str, str], ov: Dict[str, str], key: str) -> str:
    g = lambda d: d.get(key, "--") if d else "--"
    return f"{label} & {g(tr)} & {g(te)} & {g(ov)} \\\\"


def build_latex(
    train: Dict[str, str],
    test: Dict[str, str],
    overall: Dict[str, str],
    inst_rows: List[List[str]],
    inst_col: Optional[str],
    meta: Dict[str, str],
) -> str:
    L: List[str] = []
    A = L.append

    A("% Auto-generated by fine-tuning/generate_dataset_table.py -- do not edit by hand.")
    A(f"% Generated: {meta['timestamp']}")
    A(f"% Train meta CSV : {meta['meta_csv']}")
    A(f"% Test CSV       : {meta['test_csv']}")
    A(f"% Metadata DB    : {meta['db']}")
    A(f"% Train/Test SOP coverage in DB: {meta['train_cov']} / {meta['test_cov']}")
    A("% Requires \\usepackage{booktabs} in the document preamble.")
    A("")
    A("\\begin{table}[ht]")
    A("\\centering")
    A("\\caption{\\textbf{Dataset characteristics.} \\textbf{(A)}~Patient and "
      "acquisition characteristics of the training and external test cohorts. "
      "\\textbf{(B)}~Composition of the external test set by contributing "
      "institution. Values are $n$ (\\%) unless otherwise indicated; age in "
      "years. Demographics derived from DICOM \\texttt{PatientAge} (0010,1010), "
      "\\texttt{PatientSex} (0010,0040), \\texttt{StudyDate} (0008,0020) and "
      "\\texttt{ReferringPhysicianName} (0008,0090) joined by SOPInstanceUID.}")
    A("\\label{tab:dataset}")
    A("\\small")
    A("\\renewcommand{\\arraystretch}{1.2}")
    A("")
    A("% ---- Panel A: characteristics by cohort ----")
    A("\\begin{tabular}{@{}l rrr@{}}")
    A("\\toprule")
    A("\\textbf{Characteristic} & \\textbf{Training} & \\textbf{External test} & \\textbf{Overall} \\\\")
    A("\\midrule")
    A("\\multicolumn{4}{@{}l}{\\textit{Cohort size}}\\\\")
    A(_cells("\\quad Patients, $n$", train, test, overall, "patients"))
    A(_cells("\\quad Studies, $n$", train, test, overall, "studies"))
    A(_cells("\\quad DSA sequences, $n$", train, test, overall, "sequences"))
    A(_cells("\\quad Frames, $n$", train, test, overall, "frames"))
    A(_cells("\\quad Paired radiology reports, $n$", train, test, overall, "reports"))
    A("\\addlinespace[2pt]")
    A("\\multicolumn{4}{@{}l}{\\textit{Age, years}}\\\\")
    A(_cells("\\quad Mean $\\pm$ SD", train, test, overall, "age_mean_sd"))
    A(_cells("\\quad Median [IQR]", train, test, overall, "age_median_iqr"))
    A(_cells("\\quad Range", train, test, overall, "age_range"))
    A("\\addlinespace[2pt]")
    A("\\multicolumn{4}{@{}l}{\\textit{Sex}}\\\\")
    A(_cells("\\quad Female, $n$ (\\%)", train, test, overall, "female"))
    A(_cells("\\quad Male, $n$ (\\%)", train, test, overall, "male"))
    A(_cells("\\quad Unknown/other, $n$ (\\%)", train, test, overall, "sex_other"))
    A("\\addlinespace[2pt]")
    A("\\multicolumn{4}{@{}l}{\\textit{Acquisition}}\\\\")
    A(_cells("\\quad Study year range", train, test, overall, "year_range"))
    A(_cells("\\quad Referring physicians, $n$", train, test, overall, "physicians"))
    A(_cells("\\quad Frames per sequence, median [range]", train, test, overall, "frames_per_seq"))
    A(_cells("\\quad Contrast-enhanced sequences, $n$ (\\%)", train, test, overall, "contrast"))
    A("\\bottomrule")
    A("\\end{tabular}")
    A("")
    A("\\vspace{0.8em}")
    A("")
    A("% ---- Panel B: external test set by institution ----")
    A("\\footnotesize")
    A("\\setlength{\\tabcolsep}{5pt}")
    A("\\begin{tabular}{@{}l rrrr r r r@{}}")
    A("\\multicolumn{8}{@{}l}{\\textbf{(B) External test set by contributing institution}}\\\\[2pt]")
    A("\\toprule")
    A("\\textbf{Institution} & \\textbf{Patients} & \\textbf{Studies} & "
      "\\textbf{Sequences} & \\textbf{Frames} & \\textbf{Female, $n$ (\\%)} & "
      "\\textbf{Age (mean$\\pm$SD)} & \\textbf{Year range} \\\\")
    A("\\midrule")
    for i, row in enumerate(inst_rows):
        if i == len(inst_rows) - 1:  # Total row
            A("\\midrule")
        A(" & ".join(row) + " \\\\")
    A("\\bottomrule")
    A("\\end{tabular}")
    if not inst_col:
        A("")
        A("\\vspace{2pt}")
        A("{\\footnotesize $^{\\dagger}$DICOM \\texttt{InstitutionName} (0008,0080) "
          "is not present in the metadata database, so the external test set "
          "cannot be split by institution. Add \\texttt{InstitutionName} to the "
          "ingestion parser and re-ingest to populate this panel.}")
    A("\\end{table}")
    A("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta_csv", type=Path, default=Path(DEFAULT_META_CSV),
                   help="Training meta CSV (consolidated metadata used as --meta_csv at train time).")
    p.add_argument("--test_csv", type=Path, default=None,
                   help="GT validation CSV. If omitted, resolved from --settings VALIDATION_CSV.")
    p.add_argument("--settings", type=Path, default=Path(DEFAULT_SETTINGS),
                   help="Path to configs/settings.py used to resolve VALIDATION_CSV.")
    p.add_argument("--db", type=Path, default=Path(DEFAULT_DB),
                   help="SQLite metadata DB (dicom_files table).")
    p.add_argument("--reports_csv", type=Path, default=Path(DEFAULT_REPORTS_CSV),
                   help="Reports CSV (for the 'paired reports' row). Optional.")
    p.add_argument("--institution-col", type=str, default=None,
                   help="dicom_files column to use as institution for Panel B "
                        "(default tries institution_name, then station_name).")
    p.add_argument("--out_tex", type=Path, default=Path(DEFAULT_OUT_TEX),
                   help="Output LaTeX file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve test CSV.
    test_csv = args.test_csv
    if test_csv is None:
        if not args.settings.exists():
            raise SystemExit(f"[ERROR] --test_csv not given and settings.py missing: {args.settings}")
        test_csv = Path(resolve_validation_csv(args.settings))
        print(f"[INFO] VALIDATION_CSV resolved from {args.settings}: {test_csv}")

    for label, path in (("meta_csv", args.meta_csv), ("test_csv", test_csv), ("db", args.db)):
        if not Path(path).exists():
            raise SystemExit(f"[ERROR] {label} does not exist: {path}")

    train_sops, _train_accs = load_train_ids(args.meta_csv)
    test_sops, _test_accs = load_test_ids(test_csv)
    report_accs = load_report_accessions(args.reports_csv)

    union_sops = train_sops | test_sops
    df, inst_col = fetch_demographics(args.db, union_sops, args.institution_col)
    if df.empty:
        raise SystemExit(
            "[ERROR] No matching rows in dicom_files for any cohort SOPInstanceUID. "
            "Check that --db points at the DB that ingested these studies."
        )

    df["in_train"] = df["sop"].isin(train_sops)
    df["in_test"] = df["sop"].isin(test_sops)

    train_cov = f"{int(df['in_train'].sum()):,}/{len(train_sops):,}"
    test_cov = f"{int(df['in_test'].sum()):,}/{len(test_sops):,}"
    print(f"[MATCH] Train SOPs found in DB: {train_cov}")
    print(f"[MATCH] Test  SOPs found in DB: {test_cov}")
    if inst_col:
        print(f"[INFO] Institution column for Panel B: '{inst_col}'")
    else:
        print("[WARN] No institution column found; Panel B will show a single 'not recorded' row.")

    df_train = df[df["in_train"]].copy()
    df_test = df[df["in_test"]].copy()

    train_stats = cohort_stats(df_train, report_accs)
    test_stats = cohort_stats(df_test, report_accs)
    overall_stats = cohort_stats(df, report_accs)
    inst_rows = institution_rows(df_test, inst_col) if not df_test.empty else [["\\textbf{Total}"] + ["--"] * 7]

    meta = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "meta_csv": str(args.meta_csv),
        "test_csv": str(test_csv),
        "db": str(args.db),
        "train_cov": train_cov,
        "test_cov": test_cov,
    }
    latex = build_latex(train_stats, test_stats, overall_stats, inst_rows, inst_col, meta)

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(latex, encoding="utf-8")

    # Console preview.
    print("\n" + "=" * 70)
    print("PANEL A  (Characteristic | Training | External test | Overall)")
    print("=" * 70)
    order = [
        ("Patients", "patients"), ("Studies", "studies"), ("Sequences", "sequences"),
        ("Frames", "frames"), ("Reports", "reports"), ("Age mean+-SD", "age_mean_sd"),
        ("Age median[IQR]", "age_median_iqr"), ("Age range", "age_range"),
        ("Female", "female"), ("Male", "male"), ("Sex other", "sex_other"),
        ("Year range", "year_range"), ("Referring physicians", "physicians"),
        ("Frames/seq", "frames_per_seq"), ("Contrast", "contrast"),
    ]
    for label, key in order:
        tr = train_stats.get(key, "--") if train_stats else "--"
        te = test_stats.get(key, "--") if test_stats else "--"
        ov = overall_stats.get(key, "--") if overall_stats else "--"
        print(f"  {label:<22} | {tr:<18} | {te:<18} | {ov}")

    print(f"\n[OK] Wrote LaTeX table to: {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
