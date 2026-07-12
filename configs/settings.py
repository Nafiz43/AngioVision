# from pathlib import Path

# # VALIDATION_CSV = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"
# VALIDATION_CSV = "/data/Deep_Angiography/Validation_Data/test-data/gt.csv"

# # Construct DATA_DIR dynamically
# DATA_DIR = Path(VALIDATION_CSV).parent / "DICOM_Sequence_Processed"

# # Optional: convert to string if needed
# DATA_DIR = str(DATA_DIR)




# VALIDATION_CSV = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/VLM_Test_Data_2026_03_04_v01.csv"
# DATA_DIR = "/data/Deep_Angiography/Validation_Data/Validation_Data_2026_03_04/DICOM_Sequence_Processed"

# VLM_Validation_Data_2026_06_17 snapshot (no generated mosaics — superseded):
# VALIDATION_CSV = "/data/Deep_Angiography/VLM_Validation_Data_2026_06_17/VLM_Validation_Data_2026_06_17_v01_filtered.csv"
# DATA_DIR = "/data/Deep_Angiography/VLM_Validation_Data_2026_06_17/VLM_Validation_Data_2026_06_17/DICOM_Sequence_Processed"

# Validation_VDP snapshot: has generated mosaics under DSA_Split/<split>/
# <accession>/<SOP>/mosaic.png. CSV is the same curated *_filtered QA set,
# now sitting directly under the validation root.
VALIDATION_CSV = "/data/Deep_Angiography/Validation_VDP/VLM_Validation_Data_2026_06_17_v01_filtered.csv"
DATA_DIR = "/data/Deep_Angiography/Validation_VDP"