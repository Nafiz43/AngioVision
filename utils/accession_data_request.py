import pandas as pd

# Load both files
f1 = pd.read_csv('/data/Deep_Angiography/DICOM-metadata-stats/missing_accessions_has_visual.csv')
f2 = pd.read_csv('/data/Deep_Angiography/DICOM-metadata-stats/missing_accessions_potential_dsas.csv')

# Find overlapping accessions
overlap = set(f1['AccessionNumber']).intersection(set(f2['AccessionNumber']))

print(f"F1 total: {len(f1)}")
print(f"F2 total: {len(f2)}")
print(f"Overlapping accessions: {len(overlap)}")

if overlap:
    print("\nOverlapping AccessionNumbers:")
    for acc in sorted(overlap):
        print(f"  {acc}")

# Remove overlapping accessions from f1
f1_filtered = f1[~f1['AccessionNumber'].isin(overlap)]

print(f"\nF1 after removing overlaps: {len(f1_filtered)}")

# Save result
output_path = '/data/Deep_Angiography/DICOM-metadata-stats/accession_data_request.csv'
f1_filtered.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")