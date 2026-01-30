graph LR
    subgraph Study["Study (One Report)"]
        direction LR
        
        Start["Study Input"]
        
        subgraph Sequences["Video Sequences"]
            direction TB
            S1["Sequence 1<br/>(DICOM Video)"]
            S2["Sequence 2<br/>(DICOM Video)"]
            Sdots["..."]
            SN["Sequence N<br/>(DICOM Video)"]
        end
        
        subgraph Encoders["Vision Encoders (Trainable)"]
            direction TB
            VE1["Vision Encoder 1"]
            VE2["Vision Encoder 2"]
            VEdots["..."]
            VEN["Vision Encoder N"]
        end
        
        subgraph Features["Extracted Features"]
            direction TB
            F1["Features 1"]
            F2["Features 2"]
            Fdots["..."]
            FN["Features N"]
        end
        
        Merge["Merge Layer<br/>(Fusion)"]
        
        MergedFeatures["Merged<br/>Visual Features"]
        
        LLM["Language Model<br/>(Frozen)"]
        
        Report["Angiographic<br/>Report"]
    end
    
    %% Connections
    Start --> Sequences
    
    S1 --> VE1
    S2 --> VE2
    SN --> VEN
    
    VE1 --> F1
    VE2 --> F2
    VEN --> FN
    
    F1 --> Merge
    F2 --> Merge
    FN --> Merge
    
    Merge --> MergedFeatures
    MergedFeatures --> LLM
    LLM --> Report
    
    %% Styling
    classDef trainable fill:#90EE90,stroke:#2E8B57,stroke-width:3px
    classDef frozen fill:#FFB6C1,stroke:#C71585,stroke-width:3px
    classDef data fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    classDef process fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    classDef dots fill:#F0F0F0,stroke:#808080,stroke-width:1px,stroke-dasharray: 3 3
    
    class VE1,VE2,VEN trainable
    class LLM frozen
    class Start,S1,S2,SN,Report data
    class Merge,MergedFeatures,F1,F2,FN process
    class Sdots,VEdots,Fdots dots
