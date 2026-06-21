# AngioVision

AngioVision is a research codebase for **multi-modal analysis of coronary angiography
studies**, combining angiographic image sequences (DICOM) with their free-text
radiology reports. It spans the full research lifecycle: DICOM ingestion and
cleaning, frame/mosaic processing, vision-language model inference and fine-tuning,
an interactive natural-language query engine over the curated database, and tooling
for a systematic literature review.

> An angiography **study** contains multiple **sequences** (cine loops); each
> sequence has *n* **frames** capturing temporal cardiac anatomy, and each study has
> one textual **report**. The central challenge is multi-modal — aligning visual
> content with clinical text.

## Research goals

1. **Anatomically relevant questions** — generate clinically meaningful questions
   about anatomical structures from visual content.
2. **Report generation** — turn angiographic video sequences into clinical report text.
3. **Implicit findings** — extract unstated / inferred clinical insights.

See [`md-files/AngioVision-Goal.md`](./md-files/AngioVision-Goal.md) for the full
goal diagram.

## Repository layout

```
.
├── utils/                  # DICOM curation & analysis pipeline (numbered stages)
│   └── metadata_db/        # ★ DICOM Query Web App (NL→SQL + Image RAG) + ingestion
├── frame-processing/       # Mosaic creation, frame cleanup, label/report extraction
├── batch-processing/       # Batch DICOM processing, frame-text aggregation, eval
├── text-report-processing/ # Parse reports → sequences, sequence-level QA generation
├── bedrock-inference/      # AWS Bedrock model inference & multi-region testing
├── fine-tuning/            # CLIP/SigLIP fine-tuning + custom temporal framework
├── slr/                    # Systematic Literature Review tooling
├── configs/                # Shared question bank + settings
├── md-files/               # Project documentation & notes
├── z-deprecated-scripts/   # Retired scripts kept for reference
├── index.html              # Generated SLR Network Explorer (interactive graph)
└── requirements.txt        # Base Python deps (subsystems add their own)
```

> **Conventions:** scripts in pipeline folders are **numbered by stage**
> (`00_…`, `01_…`, …) and intended to run in order. Files prefixed `d_`/`d-` are
> diagnostic / data-prep helpers; `z-` / `zz_` are deprecated or scratch.

## Subsystems

### ★ DICOM Query Web App — `utils/metadata_db/`

A Flask web app + REST API for exploring the curated DICOM database in natural
language:

- **Agentic NL→SQL** via [smolagents](https://github.com/huggingface/smolagents)
  driving a local **Ollama** model (qwen3 / llama3.1+ / mistral) that can explore the
  schema, inspect values, and self-correct SQL across multiple turns.
- **Image RAG** — upload an angiography image to retrieve visually similar sequences
  using **RAD-DINO** embeddings stored in **ChromaDB**, enriched with SQLite metadata
  and report excerpts.
- **DICOM frame rendering** — thumbnails, full-resolution PNGs, and multi-frame strips
  per sequence.
- Maintained entry points: **`run_server.py`** (web server, `qa_app/` package) and
  **`run_ingest.py`** (data ingestion, `ingestion/` package). The legacy single-file
  equivalents `qa_pipe.py` / `ingest.py` are kept for reference with identical behavior.
- Also includes a knowledge-graph retrieval evaluation (`eval_kn_retrieval.py`, with
  Neo4j helper scripts).

Full setup and usage: [`utils/metadata_db/how-to-run-query.txt`](./utils/metadata_db/how-to-run-query.txt).

```bash
cd utils/metadata_db
pip install -r requirements.txt
python3 run_ingest.py          # build SQLite (metadata + reports) + ChromaDB (embeddings)
python3 run_server.py          # serve the UI at http://localhost:5050
```

### DICOM curation & analysis — `utils/`

A large staged pipeline that turns raw DICOM into a clean, labeled dataset:
consistency checks, sequence filtering & DSA identification, frame statistics, mosaic
creation, consolidated metadata generation, report cleaning, data augmentation, and an
interactive viewer (`23_interactive_angio.py`).

### Frame & mosaic processing — `frame-processing/`

Trim black borders, build per-sequence / per-study **mosaics**, extract labels from
mosaics with CLIP / BiomedCLIP / MedCLIP, and generate reports from mosaics via RadFM,
Med-Flamingo, or Ollama models.

### Batch processing — `batch-processing/`

Batch DICOM processing, frame-text extraction & aggregation, label extraction from
validation mosaics, model evaluation, and CSV merging.

### Text report processing — `text-report-processing/`

Extract per-sequence descriptions from radiology reports and build sequence-level
question/answer pairs using ground-truth reports.

### Cloud inference — `bedrock-inference/`

Run label extraction and model comparisons through **AWS Bedrock**, including
multi-region model testing and single-frame probes.

### Fine-tuning — `fine-tuning/`

Fine-tune CLIP / SigLIP on mosaics (report-level and sequence-level) and train a
**custom temporal multimodal framework** (time-aware training, generation, validation,
scoring). `pipeline.py` is a minimal, dependency-free scaffold illustrating the
architecture; see [`fine-tuning/readme.md`](./fine-tuning/readme.md).

### Systematic Literature Review — `slr/`

End-to-end SLR tooling: PDF preprocessing, citation-network graph generation, paper
fetching, two-stage screening / extraction, CLAIM-2024 checklist extraction, and
temporal ablation analysis. The root **`index.html`** is the generated interactive
**SLR Network Explorer**.

## Installation

There is no single environment for the whole repo — each subsystem declares its own
dependencies. Start with the base, then add the extras you need:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# e.g. for the DICOM Query Web App:
pip install -r utils/metadata_db/requirements.txt
```

Most model-inference and fine-tuning scripts require **PyTorch** (GPU recommended) and
`transformers`; cloud scripts require AWS credentials; the query app requires a running
**Ollama** server.

## Documentation

- [`md-files/AngioVision-Goal.md`](./md-files/AngioVision-Goal.md) — project goals diagram
- [`md-files/Updates.md`](./md-files/Updates.md) — progress log
- [`utils/metadata_db/how-to-run-query.txt`](./utils/metadata_db/how-to-run-query.txt) — query app guide
- [`utils/metadata_db/how-to-run-kn-retrieval.txt`](./utils/metadata_db/how-to-run-kn-retrieval.txt) — knowledge-graph retrieval guide
- [`fine-tuning/readme.md`](./fine-tuning/readme.md) — fine-tuning prototype

## License

Research code. Third-party models (e.g. Qwen2.5-VL, RAD-DINO, CLIP/SigLIP, RadFM,
Med-Flamingo) are subject to their respective licenses — consult each model card
before use.
