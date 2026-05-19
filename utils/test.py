import glob, pandas as pd

base = "/data/Deep_Angiography/DICOM_Sequence_Processed/00_potential_dsas"
csvs = glob.glob(f"{base}/**/metadata.csv", recursive=True)

print(f"Found {len(csvs)} metadata.csv files")

accessions = set()
errors = []

for f in csvs:
    try:
        df = pd.read_csv(f, header=0, dtype=str)  # columns: Information, Value
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
