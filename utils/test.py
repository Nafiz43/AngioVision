import csv
import glob
import os
import pandas as pd

# ------------------------------------------------------------------
# 1️⃣  Find all metadata.csv files (unchanged)
# ------------------------------------------------------------------
base = "/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
csvs = glob.glob(f"{base}/**/metadata.csv", recursive=True)
print(f"Found {len(csvs)} metadata.csv files")

# ------------------------------------------------------------------
# 2️⃣  Pull the accession numbers (unchanged)
# ------------------------------------------------------------------
accessions = set()
errors = []

for f in csvs:
    try:
        df = pd.read_csv(f, header=0, dtype=str)
        row = df[df["Information"] == "accession_number"]
        if not row.empty:
            val = row["Value"].iloc[0]
            if pd.notna(val):
                accessions.add(val)
        else:
            errors.append(f"No accession_number row: {f}")
    except Exception as e:
        errors.append(f"Error reading {f}: {e}")

if errors:
    print(f"\n⚠ Issues ({len(errors)}):")
    for e in errors:
        print(f"  {e}")

print(f"\nTotal metadata.csv files  : {len(csvs)}")
print(f"Unique accession numbers   : {len(accessions)}")

# ------------------------------------------------------------------
# 3️⃣  Write the results to the requested path
# ------------------------------------------------------------------
output_file = "/data/Deep_Angiography/DICOM-metadata-stats/missing_accessions_potential_dsas.csv"

# Make sure the parent folder exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_accessions = pd.DataFrame({'accession_number': sorted(accessions)})

# Quote non‑numeric values just to be safe
df_accessions.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"\n✅  Wrote {len(df_accessions)} unique accession numbers to:\n    {output_file}")
