import pandas as pd

# Load both files
f1 = pd.read_csv('/data/Deep_Angiography/DICOM-metadata-stats/missing_accessions_has_visual.csv')
f2 = pd.read_csv('/data/Deep_Angiography/DICOM-metadata-stats/missing_accessions_potential_dsas.csv')

# Find overlapping accessions
overlap = set(f1['AccessionNumber']).intersection(set(f2['AccessionNumber']))

print(f"F1 total: {len(f1)}")
print(f"F2 total: {len(f2)}")
print(f"Overlapping accessions: {len(overlap)}")
c=0
if overlap:
    print("\nOverlapping AccessionNumbers:")
    
    for acc in sorted(overlap):
        c+=1
    print(f"  {acc}")