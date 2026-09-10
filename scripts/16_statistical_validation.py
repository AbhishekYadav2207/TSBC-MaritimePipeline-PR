"""
Stage 16: Statistical Validation Wrapper
Maritime Accident Corpus Generation Pipeline Version 2.1

This module provides a thin compatibility entry point around the canonical
scripts/16_statistical_analysis.py implementation.
"""

import sys
import importlib
from pathlib import Path

# Ensure scripts directory is in sys.path
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

canonical_module = importlib.import_module("16_statistical_analysis")
main = canonical_module.main

if __name__ == "__main__":
    main()
