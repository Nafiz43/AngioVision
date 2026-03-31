import os
import glob
import json
import pandas as pd
from datetime import datetime

# ----------------------------
# Paths (as you provided)
# ----------------------------
JSON_DIR = "/data/Deep_Angiography/Reports/Report_List_v01_01_sequences_json"
CSV_PATH = "/data/Deep_Angiography/DICOM_Sequence_Processed_Output/mosaics_extracted_labels_clip_strict.csv"

# Output
OUT_CSV = "aligned_answers.csv"


# ----------------------------
# Helpers
# ----------------------------
def normalize_answer(a: str) -> str:
    """
    Normalize answers so comparisons are robust.
    Example: "N/A", "n/a", "na" -> "n/a"
    """
    if a is None:
        return ""
    a = str(a).strip().lower()
    # unify common NA variants
    if a in {"na", "n.a.", "n/a", "n\\a", "n-a"}:
        return "n/a"
    return a


def parse_uid_from_sequence_dir(sequence_dir: str) -> str:
    """
    From: 01_5sDSA/<UID>  -> returns UID
    """
    if sequence_dir is None:
        return ""
    s = str(sequence_dir).strip()
    # everything after last slash
    return s.split("/")[-1]


def parse_timestamp(ts: str):
    """
    CSV timestamps look like: 2026-01-27T19:42:41Z
    """
    if not ts or pd.isna(ts):
        return pd.NaT
    try:
        # handle Zulu 'Z'
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return pd.NaT


# ----------------------------
# 1) Load JSON QA into a DataFrame
# ----------------------------
json_rows = []

json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
if not json_files:
    raise FileNotFoundError(f"No .json files found in: {JSON_DIR}")

for fp in json_files:
    with open(fp, "r") as f:
        data = json.load(f)

    sequences = data.get("sequences", [])
    for seq in sequences:
        uid = seq.get("sequence_instance_uid", "")
        qa_list = seq.get("qa", [])

        for qa in qa_list:
            question = qa.get("question", "")
            answer = qa.get("answer", "")

            json_rows.append({
                "uid": uid,
                "question": question,
                "answer_json": answer,
                "answer_json_norm": normalize_answer(answer),
                "json_source_file": os.path.basename(fp),
            })

df_json = pd.DataFrame(json_rows)

if df_json.empty:
    raise RuntimeError("Extracted 0 QA rows from JSON. Check JSON structure/path.")


# ----------------------------
# 2) Load CSV and prep fields
# ----------------------------
df_csv = pd.read_csv(CSV_PATH)

required_cols = {"sequence_dir", "question", "answer"}
missing = required_cols - set(df_csv.columns)
if missing:
    raise ValueError(f"CSV missing required columns: {missing}. Found columns: {list(df_csv.columns)}")

df_csv["uid"] = df_csv["sequence_dir"].apply(parse_uid_from_sequence_dir)
df_csv["answer_csv_norm"] = df_csv["answer"].apply(normalize_answer)

# timestamp + confidence are optional but useful for dedup
if "Timestamp" in df_csv.columns:
    df_csv["Timestamp_parsed"] = df_csv["Timestamp"].apply(parse_timestamp)
else:
    df_csv["Timestamp_parsed"] = pd.NaT

if "confidence" in df_csv.columns:
    # ensure numeric
    df_csv["confidence_num"] = pd.to_numeric(df_csv["confidence"], errors="coerce")
else:
    df_csv["confidence_num"] = pd.NA


# ----------------------------
# 3) Deduplicate CSV rows per (sequence_dir, question)
#    Choose ONE row if multiple models/timestamps exist.
#
#    Strategy options:
#      A) "latest_timestamp"  (default): keep most recent Timestamp
#      B) "highest_confidence": keep max confidence
# ----------------------------
DEDUP_STRATEGY = "latest_timestamp"  # change to "highest_confidence" if preferred

group_keys = ["sequence_dir", "question"]

if DEDUP_STRATEGY == "highest_confidence":
    # Sort so max confidence is first; tie-breaker: latest timestamp
    df_csv_sorted = df_csv.sort_values(
        by=["confidence_num", "Timestamp_parsed"],
        ascending=[False, False],
        na_position="last"
    )
elif DEDUP_STRATEGY == "latest_timestamp":
    # Sort so latest timestamp is first; tie-breaker: highest confidence
    df_csv_sorted = df_csv.sort_values(
        by=["Timestamp_parsed", "confidence_num"],
        ascending=[False, False],
        na_position="last"
    )
else:
    raise ValueError(f"Unknown DEDUP_STRATEGY: {DEDUP_STRATEGY}")

df_csv_dedup = df_csv_sorted.drop_duplicates(subset=group_keys, keep="first").copy()


# ----------------------------
# 4) Merge JSON and CSV on (uid, question)
# ----------------------------
df_merged = df_csv_dedup.merge(
    df_json,
    on=["uid", "question"],
    how="inner",
    suffixes=("_csv", "_json")
)

# ----------------------------
# 5) Keep only aligned answers
# ----------------------------
aligned = df_merged[df_merged["answer_csv_norm"] == df_merged["answer_json_norm"]].copy()

# ----------------------------
# 6) Output exactly what you asked:
#    SequenceID Location, Question, Answer
#
# "SequenceID Location" -> using CSV's 'sequence_dir' (it includes the location + UID)
# "Answer" -> the aligned answer (same from both), so we can output JSON's raw or CSV's raw
# ----------------------------
out = aligned[["sequence_dir", "question", "answer_json"]].rename(columns={
    "sequence_dir": "SequenceID Location",
    "question": "Question",
    "answer_json": "Answer"
})

# Save
out.to_csv(OUT_CSV, index=False)

print(f"Done.")
print(f"JSON QA rows: {len(df_json):,}")
print(f"CSV rows (raw): {len(df_csv):,}")
print(f"CSV rows (dedup): {len(df_csv_dedup):,}")
print(f"Merged rows: {len(df_merged):,}")
print(f"Aligned rows: {len(out):,}")
print(f"Saved: {OUT_CSV}")
