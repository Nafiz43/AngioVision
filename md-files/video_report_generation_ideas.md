# Video-to-Report Generation System Design Diagrams

**Research-Grounded Design**: Each diagram below is inspired by and grounded in published research papers in medical imaging, video understanding, and multimodal AI.

---

## 1. High-Level System Architecture

**Inspired by**: 
- Chen et al. (2020) - "Generating Radiology Reports via Memory-driven Transformer" (R2Gen, EMNLP 2020)
- Nature Digital Medicine (2025) - "Multimodal generative AI for interpreting 3D medical images and videos"

**Key insight**: Medical report generation follows an encoder-decoder paradigm where video encoders extract spatial-temporal features and language decoders generate coherent clinical text.

```mermaid
graph TB
    subgraph Input
        V1[Video Sequence 1]
        V2[Video Sequence 2]
        Vn[Video Sequence n]
        R[Text Report]
    end
    
    subgraph "Feature Extraction"
        FE1[Frame Encoder<br/>CNN/ViT]
        FE2[Temporal Encoder<br/>LSTM/Transformer]
        FE3[Cross-Video Aggregator]
    end
    
    subgraph "Fusion & Generation"
        MM[Multi-Modal Fusion<br/>Cross-Attention]
        TG[Text Generator<br/>Transformer Decoder]
    end
    
    subgraph Output
        TR[Generated Report]
    end
    
    V1 --> FE1
    V2 --> FE1
    Vn --> FE1
    FE1 --> FE2
    FE2 --> FE3
    R -.Training Only.-> MM
    FE3 --> MM
    MM --> TG
    TG --> TR
    
    style R fill:#ffe6e6
    style TR fill:#e6ffe6
```

## 2. Detailed Training Pipeline

**Inspired by**: 
- Pang et al. (2023) - "A survey on automatic generation of medical imaging reports based on deep learning" (PMC)
- arXiv (2026) - "arXiv:2602.17112" (https://arxiv.org/abs/2602.17112)

**Key insight**: Training medical image captioning models requires careful data preprocessing, multi-stage encoding (frame→temporal→study), and iterative validation with both NLG and clinical efficacy metrics.

```mermaid
flowchart TD
    Start([Start Training]) --> LoadData[Load Study Data]
    LoadData --> PreProc[Preprocess Videos & Reports]
    
    PreProc --> BatchGen[Generate Training Batch]
    BatchGen --> VidEnc[Encode Video Sequences]
    
    subgraph "Video Encoding Path"
        VidEnc --> FrameEmb[Frame-level Embeddings<br/>Per Video]
        FrameEmb --> TempAgg[Temporal Aggregation<br/>Per Video]
        TempAgg --> StudyAgg[Study-level Aggregation<br/>Across Videos]
    end
    
    StudyAgg --> Fusion[Multi-Modal Fusion Layer]
    
    subgraph "Report Generation Path"
        Fusion --> Decoder[Transformer Decoder]
        Decoder --> TokenGen[Token-by-Token Generation]
        TokenGen --> RepOut[Report Output]
    end
    
    RepOut --> Loss[Compute Loss<br/>Cross-Entropy + Optional]
    Loss --> BackProp[Backpropagation]
    BackProp --> UpdateWeights[Update Model Weights]
    
    UpdateWeights --> CheckEpoch{Epoch Complete?}
    CheckEpoch -->|No| BatchGen
    CheckEpoch -->|Yes| Validate[Validation Step]
    
    Validate --> CheckConverge{Converged?}
    CheckConverge -->|No| BatchGen
    CheckConverge -->|Yes| SaveModel[Save Best Model]
    SaveModel --> End([End Training])
    
    style Loss fill:#ffcccc
    style Validate fill:#ccffcc
```

## 3. Model Architecture (Encoder-Decoder)

**Inspired by**: 
- Chen et al. (2020) - "R2Gen: Memory-driven Transformer for Radiology Report Generation" (EMNLP 2020)
- Xu et al. (2015) - "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" (ICML 2015)
- Bertasius et al. (2021) - "Is Space-Time Attention All You Need for Video Understanding?" (TimeSformer, ICML 2021)

**Key insight**: Combining CNN/ViT for spatial features with temporal transformers, followed by cross-attention between visual and textual modalities enables high-quality report generation. R2Gen's memory-driven approach and TimeSformer's divided space-time attention are particularly effective.

```mermaid
graph TB
    subgraph "Input Layer"
        V1[Video 1<br/>Frames: T×H×W×C]
        V2[Video 2<br/>Frames: T×H×W×C]
        Vk[Video k<br/>Frames: T×H×W×C]
    end
    
    subgraph "Video Encoder Stack"
        subgraph "Per-Video Processing"
            CNN[3D CNN / ViT<br/>Spatial-Temporal Features]
            Pool[Temporal Pooling<br/>or Attention]
            VidEmb[Video Embedding<br/>d-dimensional]
        end
        
        subgraph "Study-Level Processing"
            SetTrans[Set Transformer /<br/>Cross-Video Attention]
            StudyEmb[Study Embedding<br/>d-dimensional]
        end
    end
    
    subgraph "Report Decoder Stack"
        CrossAttn[Cross-Attention<br/>Query: Report tokens<br/>Key/Value: Study embedding]
        SelfAttn[Self-Attention<br/>Causal Masking]
        FFN[Feed-Forward Network]
        Output[Output Projection<br/>Vocabulary Distribution]
    end
    
    V1 --> CNN
    V2 --> CNN
    Vk --> CNN
    CNN --> Pool
    Pool --> VidEmb
    VidEmb --> SetTrans
    SetTrans --> StudyEmb
    
    StudyEmb --> CrossAttn
    CrossAttn --> SelfAttn
    SelfAttn --> FFN
    FFN --> Output
    
    Output --> Token[Generated Token]
    Token -.Autoregressive.-> CrossAttn
    
    style StudyEmb fill:#bbdefb
    style Token fill:#c8e6c9
```

## 4. Data Processing Pipeline

**Inspired by**: 
- Nature Digital Medicine (2025) - "Multimodal generative AI for interpreting 3D medical images and videos"
- Medical Image Captioning surveys showing preprocessing requirements for DICOM windowing and normalization

**Key insight**: Medical videos require specialized preprocessing (DICOM windowing, proper normalization, frame sampling) different from natural videos. Data augmentation must preserve clinical validity.

```mermaid
flowchart LR
    subgraph "Raw Data"
        RV[(Raw Videos<br/>DICOM/MP4)]
        RR[(Raw Reports<br/>Text/XML)]
    end
    
    subgraph "Preprocessing"
        VidProc[Video Processing<br/>- Resize frames<br/>- Normalize<br/>- Sample frames]
        TextProc[Text Processing<br/>- Tokenization<br/>- Clean & normalize<br/>- Add special tokens]
    end
    
    subgraph "Data Augmentation"
        VidAug[Video Augmentation<br/>- Random crop<br/>- Brightness/contrast<br/>- Temporal jittering]
        TextAug[Text Augmentation<br/>- Optional]
    end
    
    subgraph "Dataset"
        DS[(Training Dataset<br/>Study ID → Videos + Report)]
    end
    
    RV --> VidProc
    RR --> TextProc
    VidProc --> VidAug
    TextProc --> TextAug
    VidAug --> DS
    TextAug --> DS
    
    DS --> DL[DataLoader<br/>Batching & Shuffling]
    
    style DS fill:#fff9c4
```

## 5. Inference Pipeline

**Inspired by**: 
- Standard encoder-decoder inference patterns from NLP/Vision literature
- Chen et al. (2020) R2Gen inference methodology
- Autoregressive generation with beam search from transformer literature

**Key insight**: Inference follows autoregressive token generation with cross-attention to video embeddings at each step. Study-level aggregation of multiple videos happens before report generation begins.

```mermaid
sequenceDiagram
    participant User
    participant System
    participant VideoEncoder
    participant ReportGenerator
    participant PostProc
    
    User->>System: Upload Video Sequences
    System->>System: Validate inputs
    
    loop For each video
        System->>VideoEncoder: Encode video frames
        VideoEncoder->>VideoEncoder: Extract spatial features
        VideoEncoder->>VideoEncoder: Extract temporal features
        VideoEncoder-->>System: Video embedding
    end
    
    System->>System: Aggregate video embeddings
    System->>ReportGenerator: Study-level embedding
    
    ReportGenerator->>ReportGenerator: Initialize with [START] token
    
    loop Until [END] or max_length
        ReportGenerator->>ReportGenerator: Cross-attend to video features
        ReportGenerator->>ReportGenerator: Generate next token
        ReportGenerator->>ReportGenerator: Append to sequence
    end
    
    ReportGenerator-->>PostProc: Generated token sequence
    PostProc->>PostProc: Detokenize
    PostProc->>PostProc: Format report
    PostProc-->>User: Final medical report
    
    Note over User,PostProc: Beam search or sampling can be used
```

## 6. Multi-Video Aggregation Strategies

**Inspired by**: 
- Lee et al. (2019) - "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" (ICML 2019)
- Attention mechanisms from Vaswani et al. (2017) - "Attention is All You Need" (NeurIPS 2017)
- Various pooling strategies from deep learning literature

**Key insight**: Multiple videos from one study require permutation-invariant aggregation. Set Transformers and attention-based pooling preserve information better than simple averaging, especially when videos have different clinical significance.

```mermaid
graph TD
    subgraph "Input: Multiple Video Embeddings"
        E1[Video 1 Embedding]
        E2[Video 2 Embedding]
        En[Video n Embedding]
    end
    
    subgraph "Strategy 1: Simple Pooling"
        AP1[Average Pooling]
        MP1[Max Pooling]
        CP1[Concatenation]
    end
    
    subgraph "Strategy 2: Attention-Based"
        SA[Self-Attention<br/>Transformer Encoder]
        CA[Cross-Attention<br/>with Learnable Queries]
        WA[Weighted Average<br/>Learned Weights]
    end
    
    subgraph "Strategy 3: Set Encoding"
        ST[Set Transformer<br/>Permutation Invariant]
        ISAB[Induced Set Attention]
    end
    
    E1 --> AP1
    E2 --> AP1
    En --> AP1
    
    E1 --> SA
    E2 --> SA
    En --> SA
    
    E1 --> ST
    E2 --> ST
    En --> ST
    
    AP1 --> Fusion[Fusion Layer]
    MP1 --> Fusion
    CP1 --> Fusion
    SA --> Fusion
    CA --> Fusion
    WA --> Fusion
    ST --> Fusion
    ISAB --> Fusion
    
    Fusion --> Out[Study Representation]
    
    style SA fill:#e1f5dd
    style ST fill:#e1f5dd
```

## 7. Loss Function Architecture

**Inspired by**: 
- Chen et al. (2020) R2Gen - Combined NLG and clinical efficacy losses
- Radford et al. (2021) CLIP - "Learning Transferable Visual Models from Natural Language Supervision" (contrastive alignment)
- Standard cross-entropy for language modeling

**Key insight**: Medical report generation benefits from multi-task learning: primary cross-entropy for token prediction, plus auxiliary losses for video-text alignment (contrastive) and clinical accuracy (domain-specific metrics).

```mermaid
graph TB
    subgraph "Predictions"
        Pred[Generated Tokens<br/>Probability Distribution]
    end
    
    subgraph "Ground Truth"
        GT[Reference Report<br/>Token Sequence]
    end
    
    subgraph "Primary Loss"
        CEL[Cross-Entropy Loss<br/>Token-level]
    end
    
    subgraph "Optional Auxiliary Losses"
        RL[ROUGE/BLEU Loss<br/>Sequence-level]
        CL[Contrastive Loss<br/>Video-Report Alignment]
        PL[Perplexity Loss<br/>Language Modeling]
    end
    
    subgraph "Regularization"
        WD[Weight Decay]
        DO[Dropout]
    end
    
    Pred --> CEL
    GT --> CEL
    
    Pred --> RL
    GT --> RL
    
    Pred --> CL
    GT --> CL
    
    CEL --> Total[Total Loss<br/>Weighted Sum]
    RL --> Total
    CL --> Total
    
    Total --> Opt[Optimizer<br/>Adam/AdamW]
    WD --> Opt
    
    style CEL fill:#ffcdd2
    style Total fill:#ef5350
```

## 8. Training Strategy Comparison

**Inspired by**: 
- Transfer learning literature (ImageNet → medical imaging)
- Google Research (2026) - MedGemma 1.5: demonstrating effectiveness of pretrained multimodal models for medical imaging
- Two-stage training from R2Gen and similar medical report generation papers

**Key insight**: Transfer learning from pretrained vision (ImageNet, Kinetics) and language models (BERT, GPT, Clinical-BERT) significantly outperforms training from scratch. Two-stage training (pretrain encoder, then train generator) offers good balance between quality and compute.

```mermaid
graph LR
    subgraph "Approach 1: End-to-End"
        E2E1[Video Encoder<br/>Random Init]
        E2E2[Report Decoder<br/>Random Init]
        E2E3[Train Together<br/>Directly on Task]
    end
    
    subgraph "Approach 2: Two-Stage"
        TS1[Stage 1: Pretrain<br/>Video Encoder]
        TS2[Stage 2: Train<br/>Report Generation]
        TS3[Optional: Fine-tune<br/>End-to-End]
    end
    
    subgraph "Approach 3: Transfer Learning"
        TL1[Pretrained Vision<br/>ImageNet/Kinetics]
        TL2[Pretrained Language<br/>GPT/BERT/Clinical]
        TL3[Freeze/Fine-tune<br/>Adapters]
    end
    
    E2E1 --> E2E2
    E2E2 --> E2E3
    
    TS1 --> TS2
    TS2 --> TS3
    
    TL1 --> TL3
    TL2 --> TL3
    
    style E2E3 fill:#fff59d
    style TS3 fill:#81c784
    style TL3 fill:#64b5f6
```

## 9. Evaluation Metrics Framework

**Inspired by**: 
- Chen et al. (2020) R2Gen - Natural Language Generation (NLG) + Clinical Efficacy (CE) metrics
- Pang et al. (2023) survey - Comprehensive evaluation framework for medical report generation
- Jing et al. (2019) - "On the Automatic Generation of Medical Imaging Reports" (clinical accuracy metrics)

**Key insight**: Medical report generation requires dual evaluation: (1) linguistic quality (BLEU, ROUGE, METEOR, CIDEr) and (2) clinical accuracy (medical entity F1, RadGraph, clinical correctness). Human evaluation remains gold standard.

```mermaid
mindmap
    root((Evaluation<br/>Metrics))
        Text Generation
            BLEU Score
            ROUGE-L
            METEOR
            CIDEr
        Clinical Accuracy
            Medical Entity F1
            Finding Detection
            Terminology Accuracy
        Human Evaluation
            Clinical Coherence
            Factual Correctness
            Completeness
        Embedding-Based
            BERTScore
            Clinical BERT Similarity
        Specialized Medical
            RadGraph F1
            RadCliQ
            Clinical Efficacy
```

## 10. Deployment Architecture

**Inspired by**: 
- Standard ML system deployment patterns (MLOps best practices)
- Production medical AI systems requiring high availability and auditability
- Microservices architecture from cloud computing literature

**Key insight**: Production medical AI requires: GPU-accelerated model serving, load balancing for multiple requests, caching for efficiency, comprehensive monitoring and logging for clinical validation, and version control for models and data.

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web/Desktop Interface]
        API[REST API Client]
    end
    
    subgraph "API Gateway"
        LB[Load Balancer]
        Auth[Authentication]
    end
    
    subgraph "Application Server"
        Val[Input Validation]
        Queue[Task Queue<br/>Redis/RabbitMQ]
    end
    
    subgraph "ML Inference Service"
        subgraph "Model Serving"
            MS1[Model Server 1<br/>GPU Instance]
            MS2[Model Server 2<br/>GPU Instance]
            MSn[Model Server n<br/>GPU Instance]
        end
        
        Cache[Result Cache]
    end
    
    subgraph "Storage"
        VS[(Video Storage<br/>S3/Blob)]
        RS[(Results DB<br/>PostgreSQL)]
        ML[(Model Registry)]
    end
    
    subgraph "Monitoring"
        Logs[Logging Service]
        Metrics[Metrics Dashboard]
        Alerts[Alert System]
    end
    
    UI --> LB
    API --> LB
    LB --> Auth
    Auth --> Val
    Val --> Queue
    
    Queue --> MS1
    Queue --> MS2
    Queue --> MSn
    
    MS1 --> Cache
    MS2 --> Cache
    MSn --> Cache
    
    MS1 -.Load Model.-> ML
    MS1 -.Read Videos.-> VS
    MS1 -.Store Results.-> RS
    
    MS1 --> Logs
    MS1 --> Metrics
    Metrics --> Alerts
    
    Cache --> Val
    RS --> UI
    
    style MS1 fill:#c5e1a5
    style Cache fill:#ffe082
```

---

## Key Research Papers Referenced

### Core Medical Report Generation:
1. **Chen, Z., Song, Y., Chang, T. H., & Wan, X. (2020)**. "Generating Radiology Reports via Memory-driven Transformer." *EMNLP 2020*. https://arxiv.org/abs/2010.16056
   - Introduces R2Gen with relational memory and memory-driven conditional layer normalization

2. **Pang, T., Li, P., & Zhao, L. (2023)**. "A survey on automatic generation of medical imaging reports based on deep learning." *Biomedical Engineering Online*. PMC10195007

3. **arXiv (2026)**. "arXiv:2602.17112". https://arxiv.org/abs/2602.17112

### Vision-Language Foundation Models:
4. **Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., & Bengio, Y. (2015)**. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." *ICML 2015*. https://arxiv.org/abs/1502.03044
   - Pioneering attention mechanism for image captioning

5. **Radford, A., Kim, J. W., Hallacy, C., et al. (2021)**. "Learning Transferable Visual Models from Natural Language Supervision (CLIP)." *OpenAI*. https://arxiv.org/abs/2103.00020
   - Contrastive language-image pretraining that aligns vision and language

### Video Understanding:
6. **Bertasius, G., Wang, H., & Torresani, L. (2021)**. "Is Space-Time Attention All You Need for Video Understanding?" *ICML 2021*. https://arxiv.org/abs/2102.05095
   - TimeSformer: divided space-time attention for efficient video modeling

7. **Nature Digital Medicine (2025)**. "Multimodal generative AI for interpreting 3D medical images and videos." https://www.nature.com/articles/s41746-025-01649-4
   - Adapting video-text models for 3D medical imaging by treating volumes as video sequences

### Medical AI Products:
8. **Google Research (2026)**. "Next generation medical image interpretation with MedGemma 1.5 and medical speech to text with MedASR."
   - State-of-the-art medical multimodal models

### Architecture Components:
9. **Vaswani, A., et al. (2017)**. "Attention is All You Need." *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
   - Foundational transformer architecture

10. **Lee, J., et al. (2019)**. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." *ICML 2019*.
    - Permutation-invariant aggregation for sets

### Clinical Datasets:
- **IU X-Ray Dataset**: 7,470 chest X-rays with 3,955 reports (Indiana University)
- **MIMIC-CXR**: Large-scale chest X-ray dataset with reports
- **Kinetics-400/600**: Video action recognition datasets used for pretraining

---

## Implementation Notes

For your specific task of video-to-report generation:

1. **Start with TimeSformer (Diagram 3)** for video encoding - it's specifically designed for medical imaging where you need to process multiple sequences
2. **Use R2Gen architecture (Diagram 3)** as your decoder with memory-driven attention
3. **Apply CLIP-style contrastive learning (Diagram 7)** to align video and text representations
4. **Implement Set Transformer (Diagram 6)** for aggregating multiple video sequences per study
5. **Follow the two-stage training strategy (Diagram 8)**: pretrain encoders, then train end-to-end

The combination of these research-backed approaches should give you a strong foundation for your medical video-to-report system.


