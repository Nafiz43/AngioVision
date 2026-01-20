## Weekly Update — Week of 5 January

### Work Completed
1. Identified and extracted **six anatomy-focused clinical questions** from radiology reports.
2. Generated **per-accession / per-study videos** from DICOM images using the `DICOM2Video` pipeline.
3. Extracted frames from each video and queried **Qwen-2.5 VL** with the same six anatomical questions using the extracted frames as input.

### Key Observations
1. The reports are **consolidated at the study level**, aggregating findings across multiple imaging sequences. This limits our ability to directly associate reported findings with specific sequences.
2. For reliable evaluation, **anatomical questions must be answered at the sequence level**. Passing an entire study-level (consolidated) video is neither traceable nor clinically verifiable with respect to individual sequences.

### Next Steps
- Develop a strategy to **distinguish and isolate individual sequences** within each study.
- Align each sequence with its corresponding anatomical context to enable **sequence-level question answering and validation**.
- Revisit and refine the question set to improve **clinical specificity and diagnostic relevance**.

## Weekly Update — Week of 12 January
1. Extraction of FRAMES, MetaData from DICOM files
2. Summary of the sequences that can be processed
3. Passsing the frames to video language model to extract anatomical questions
4. Used language model (llama-3 8B) to extract sequences from reports 
    Used Individual extracted sequences to answer anatomical questions
5. Spliced up the frames that are present in individaul sequences; and then passed those to the models; then asked seuqence level questions


## Weekly Update — Week of 19 January
1. Identified 20763 DSA sequences out of all DICOM files
2. Created consolidated metadata (DICOM_Sequence_Processed/consolidated_metadata.csv) for validating whether we can extract the number of seuqences




### Work Completed

### Key Observations

### Next Steps

