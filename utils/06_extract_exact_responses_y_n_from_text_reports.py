import os
import glob
import json
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
JSON_DIR = "/data/Deep_Angiography/Reports/LlamaMed-Report_List_v01_01_sequences_json"
OUT_CSV  = "frame_extracted_exact_labels.csv"

# --------------------------------------------------
# Helper
# --------------------------------------------------
def normalize_answer(a):
    if a is None:
        return ""
    return str(a).strip().lower()

# --------------------------------------------------
# Main extraction
# --------------------------------------------------
rows = []

json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
if not json_files:
    raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

for json_path in json_files:
    with open(json_path, "r") as f:
        data = json.load(f)

    for seq in data.get("sequences", []):
        uid = str(seq.get("sequence_instance_uid", "")).strip()

        # 🔴 DROP if no sequence ID
        if not uid:
            continue

        for qa in seq.get("qa", []):
            question = str(qa.get("question", "")).strip()
            answer_raw = qa.get("answer", "")
            answer_norm = normalize_answer(answer_raw)

            # keep only exact yes / no
            if answer_norm in {"yes", "no"}:
                rows.append({
                    "sequence_instance_uid": uid,
                    "question": question,
                    "answer": answer_norm
                })

# --------------------------------------------------
# Save
# --------------------------------------------------
df = pd.DataFrame(rows)

df.to_csv(OUT_CSV, index=False)

print("Done.")
print(f"JSON files processed : {len(json_files)}")
print(f"Rows saved           : {len(df):,}")
print(f"Output file          : {OUT_CSV}")
