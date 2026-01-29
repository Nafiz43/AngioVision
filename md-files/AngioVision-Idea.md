---
config:
  layout: fixed
---
flowchart TB
 subgraph Data["<br>"]
        Study["<b>AngioVision</b>"]
        Sequences["<b>Angiogram Sequences</b><br>Multiple sequences per study"]
        Report["<b>Textual Report</b><br>One report per study"]
        Frames["<b>Frames</b><br>n frames per sequence<br>Temporal cardiac anatomy"]
        RepContent["<b>Clinical observations<br></b>Diagnostic findings"]
  end
 subgraph Challenge["<b>Challenge</b>"]
        Problem["Multi-modal: Visual + Textual"]
  end
 subgraph Goals["<br>"]
    direction LR
        Goal1["<b>Goal 1: Anatomically Relevant Questions</b><br>Generate questions about anatomical structures<br><i>Clinically meaningful queries from visual content</i>"]
        Goal2["<b>Goal 2: Report Generation</b><br>Input: Angiographic video sequences<br><i>Output: Clinical report text</i>"]
        Goal3["<b>Goal 3: Implicit Findings</b><br>Extract unstated information<br><i>Infer hidden clinical insights</i>"]
  end
    Study --> Sequences & Report
    Sequences --> Frames
    Report --> RepContent
    Goal1 ~~~ Goal2
    Goal2 ~~~ Goal3
    Data --> Challenge
    Challenge --> Goals

    style Study fill:#E1BEE7,stroke:none
    style Sequences fill:#E1BEE7,stroke:none
    style Report fill:#FFCDD2,stroke:none
    style Frames fill:#FFF9C4,stroke:none
    style RepContent fill:#FFE0B2,stroke:none
    style Problem fill:#C8E6C9,stroke:none
    style Goal1 fill:#e8f5e9,stroke:none,stroke-width:2px
    style Goal2 fill:#fff9c4,stroke:none,stroke-width:2px
    style Goal3 fill:#fce4ec,stroke:none,stroke-width:2px
    style Data fill:#e1f5ff,stroke:none,stroke-width:2px
    style Challenge fill:#ffebee,stroke:none,stroke-width:2px
    style Goals fill:#fff4e1,stroke:none,stroke-width:2px