# Processing of Digital Subtraction Angiography (DSA): A Structured Review

## 1. Introduction

### 1.1 Why DSA is a Unique Computational Target


### 1.2 Clinical Decision Points Where DSA Processing Matters

### 1.3 Scope and Contributions of This Review

---

## 2. Review Methodology

### 2.1 Search Strategy and Inclusion/Exclusion Criteria

### 2.2 Data Extraction Schema
Each paper was summarized using a structured schema mirrored in the spreadsheet, including:
- Short paper summary  
- DSA data handling (frames vs sequences, subtraction assumptions)  
- Methods and techniques employed  
- Reported results and evaluation metrics  
- Remarks and limitations noted by the authors  

This schema enables transparent comparison across studies.

### 2.3 Synthesis Approach
---

## 3. DSA Data Handling and Representation Choices

### 3.1 Input Unit: Key-Frame vs Full Sequence

### 3.2 Subtraction Variants: Classical vs Learned

### 3.3 Preprocessing for Robustness

### 3.4 Annotation Targets Implied by the Studies
---

## 4. Technique Taxonomy by Problem Type

### 4.1 Image Quality Enhancement and Artifact Reduction

#### 4.1.1 U-Net Style Subtraction and Denoising Pipelines

#### 4.1.2 GAN-Based Subtraction

#### 4.1.3 Hybrid Segmentation–Synthesis Approaches

#### 4.1.4 Reported Failure Modes

### 4.2 Detection of Active Bleeding / Extravasation

#### 4.2.1 Frame-Level Triage

#### 4.2.2 Two-Stage vs End-to-End Pipelines

#### 4.2.3 Evaluation Practices

### 4.3 Anatomical Labeling and Classification

### 4.4 Prediction and Outcome Modeling

### 4.5 Emerging Frontier Methods

---

## 5. Evaluation Practices and Strength of Evidence

### 5.1 Validation Design Patterns

### 5.2 Metrics and Clinical Realism

### 5.3 Reproducibility Signals

---

## 6. Cross-Cutting Gaps and Unresolved Problems

### 6.1 Data Handling Gaps

### 6.2 Generalization Gaps

### 6.3 Labeling Gaps

### 6.4 Deployment Gaps

---

## 7. Practical Implications and Recommendations

### 7.1 For Researchers

### 7.2 For Clinicians and System Designers

### 7.3 For Dataset and Benchmark Builders

---

## 8. Conclusion
