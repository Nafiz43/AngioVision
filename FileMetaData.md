# Deep Angiography Dataset – CSV Documentation

This document describes the structure and semantics of the key CSV files used in the **Deep Angiography** dataset. These files link anonymized interventional radiology (IR) studies, DICOM image series, angiography run–level labels, and corresponding radiology reports.

---

## 1. `labeled_DSA_2023_10_24.csv`

This file contains **DSA (Digital Subtraction Angiography) run–level labels** aligned with specific DICOM series and file paths. It is primarily used for curated, labeled angiography runs suitable for analysis or model training.

### Columns

- **Accession**  
  An anonymized identifier for the patient/procedure (e.g., `PXgCA2nz3E`).

- **code**  
  Standardized IR procedure code (e.g., `IRTRANEMBO`, `IRURETEMBO`, `IRVISCSEL`, `IRHEPSIRTP`).

- **SeriesNumber**  
  Numerical identifier grouping related images within a procedure.

- **SeriesUID**  
  DICOM Series Instance UID uniquely identifying the image series.

- **Synology_folder_path**  
  Network path to the folder containing the images on a shared drive or Synology NAS.

- **angio_run**  
  Descriptive label for the angiography run (vessel or anatomical region).

- **run_type**  
  Imaging run type, almost always `DSA`.

- **Best_Image**  
  Frame number or quality-ranked image selected as best.

- **First_Diag_Image**  
  Starting diagnostic frame number.

- **Last_Diag_Image**  
  Ending diagnostic frame number.

- **Comments**  
  Optional notes about the run.

- **jackstraw_folder_path**  
  Local Linux filesystem path to the image folder.

- **file_path**  
  Full path to a representative DICOM (`.dcm`) file.

---

## 2. `IR_Folder_List_Labeling_V07_2023_10_24.csv`

Inventory of IR imaging runs, including DSA, fluoroscopy, and other image types.

### Columns

- **(Unnamed first column)** – Sequential index  
- **Accession** – Anonymized patient/procedure ID  
- **code** – IR procedure code  
- **SeriesNumber** – Series identifier  
- **SeriesUID** – DICOM Series Instance UID  
- **dirpath** – Network path to the image folder  
- **Angio Run** – Descriptive run label  
- **Run Type** – Imaging modality (DSA, Fluoro Run, etc.)  
- **Best_Image** – Best frame (if applicable)  
- **First_Diag_Image** – Diagnostic start frame  
- **Last_Diag_Image** – Diagnostic end frame  
- **Comments** – Optional notes  

---

## 3. `Deep_Angiography_DB_Download_V01_log.csv`

Tracks anonymization and download of imaging studies.

### Key Fields

- Original vs anonymized patient name, MRN, sex, DOB
- Original vs anonymized accession number
- Original vs anonymized exam date/time
- Image request status
- Original vs anonymized Study Instance UID
- Number of images per study

---

## 4. `Report_List_v01_01.csv`

Contains anonymized radiology reports corresponding to IR studies.

### Key Fields

- **Orig Acc # / Anon Acc #** – Link reports to imaging studies
- **radrpt** – Full free-text radiology report, including procedure, technique, findings, impression, and plan


<img width="935" height="600" alt="image" src="https://github.com/user-attachments/assets/ba28ed68-b56f-407e-9c93-5b82ea30b6ee" />



Dtaset Image

