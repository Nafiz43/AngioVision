#!/usr/bin/env python3
"""AngioVision Image Matching — entry point.

Embedding-based DSA sequence matching with K@N cross-validated evaluation.
Thin wrapper over ``imatch.cli.main`` (see README.md for full usage).

Usage
─────
  python run_matching.py
  python run_matching.py --model vit-b16 --frame-mode fl --n-folds 10
  python run_matching.py --model openclip-b32 --frame-mode all --max-frames 20
  python run_matching.py --temporal --n-folds 5
  python run_matching.py --k-values 1 3 5 --workers 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from imatch.cli import main

if __name__ == "__main__":
    main()
