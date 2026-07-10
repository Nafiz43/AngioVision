"""
imatch — AngioVision Image Matching pipeline.

Embedding-based DSA sequence matching: DICOM frames are embedded with a
vision model (RAD-DINO / ViT / OpenCLIP / any HF AutoModel), indexed in an
in-memory ChromaDB collection, and query sequences are matched by cosine
nearest-neighbour + majority vote. Evaluated via K@N accuracy under n-fold
stratified cross-validation or a single stratified holdout split.

Modularized from ``utils/metadata_db/eval_kn_retrieval.py`` (which it
supersedes as the maintained code path — the original stays in place for
provenance).

Module map
──────────
  config        constants, model aliases, label codes, output naming
  data_loading  labeled CSV loading, DICOM index, path resolution
  splits        stratified CV folds / holdout split
  embedding     embedding-model loader + temporal (mean+std) pooling
  frames        DICOM pixel extraction (best / fl / all frame modes)
  vector_store  ChromaDB collection, embedding precompute, fold ingestion
  evaluation    K@N matching evaluation, CV aggregation, K=1 examples
  reporting     console table, bar chart, Markdown, retrieval-examples docx
  cli           argparse + orchestration (entry: ``imatch.cli.main``)
"""

__version__ = "1.0.0"
