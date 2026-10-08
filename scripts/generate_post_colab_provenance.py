"""
Post-Colab Provenance & Hash Audit Generator
TSBC-MaritimePipeline-Version2.1

Generates outputs/final_audit/post_colab_provenance.json verifying SHA-256 of:
- corpus
- representations
- benchmark configuration
- primary Stage14 cache (175 cells)
- PLL outputs
- domain-aware outputs
- relevant code
"""

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit(root: Path) -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL)
        return res.decode("utf-8").strip()
    except Exception:
        return "unknown"

def generate_post_colab_provenance():
    prov = {
        "title": "Post-Colab Primary Benchmark Provenance & Hash Audit",
        "git_commit": get_git_commit(PROJECT_ROOT),
        "protocol": {
            "masking_strategy": "random_15",
            "masking_mode": "subword",
            "evaluation_unit": "subword",
            "mask_rate": 0.15,
            "seed": 42
        },
        "corpus_artifacts": {},
        "representation_artifacts": {},
        "benchmark_configuration": {},
        "primary_stage14_cache": {
            "path": "outputs/stage-14/evaluations/cache_legacy_subword_15",
            "total_cells": 175,
            "cells": {}
        },
        "auxiliary_benchmarks": {},
        "code_artifacts": {}
    }

    # 1. Corpus
    corpus_files = [
        "outputs/stage-08/maritime_corpus.txt",
        "outputs/stage-08/maritime_corpus.jsonl",
        "outputs/stage-08/manifest.json"
    ]
    for rel_path in corpus_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            prov["corpus_artifacts"][rel_path] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size
            }

    # 2. Representations
    rep_files = [
        "outputs/stage-11/corpus_representations/json.jsonl",
        "outputs/stage-11/corpus_representations/key_value.jsonl",
        "outputs/stage-11/corpus_representations/mixed.jsonl",
        "outputs/stage-11/corpus_representations/narrative.jsonl",
        "outputs/stage-11/corpus_representations/template.jsonl"
    ]
    for rel_path in rep_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            prov["representation_artifacts"][rel_path] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size
            }

    # 3. Benchmark Config
    cfg_files = [
        "config/config.json",
        "outputs/stage-13/selected_models.json"
    ]
    for rel_path in cfg_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            prov["benchmark_configuration"][rel_path] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size
            }

    # 4. Primary Stage14 Cache
    cache_dir = PROJECT_ROOT / "outputs" / "stage-14" / "evaluations" / "cache_legacy_subword_15"
    if cache_dir.exists():
        cells = sorted([f for f in cache_dir.glob("*.json") if not f.is_dir()])
        prov["primary_stage14_cache"]["total_cells"] = len(cells)
        for cell_file in cells:
            prov["primary_stage14_cache"]["cells"][cell_file.name] = {
                "sha256": compute_sha256(cell_file),
                "size_bytes": cell_file.stat().st_size
            }

    # 5. Auxiliary Benchmarks (PLL & Domain-Aware)
    aux_files = [
        "outputs/stage-14/pll_results.json",
        "outputs/stage-14/pll_selection.json",
        "outputs/stage-14/focused_domain_aware_results.json",
        "outputs/stage-14/masking_comparison.json",
        "outputs/stage-14/bert_mlm_evaluation.json"
    ]
    for rel_path in aux_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            prov["auxiliary_benchmarks"][rel_path] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size
            }

    # 6. Relevant Code
    code_files = [
        "scripts/14_mlm_evaluation.py",
        "scripts/15_cross_model_benchmarking.py",
        "scripts/16_statistical_analysis.py",
        "scripts/17_decision_engine.py",
        "scripts/18_lint_corpus.py",
        "scripts/audit_independent_calculations.py",
        "scripts/dapt_leakage_audit.py",
        "scripts/validate_stage14_ingestion.py"
    ]
    for rel_path in code_files:
        p = PROJECT_ROOT / rel_path
        if p.exists():
            prov["code_artifacts"][rel_path] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size
            }

    out_file = PROJECT_ROOT / "outputs" / "final_audit" / "post_colab_provenance.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"Generated {out_file} with {len(prov['primary_stage14_cache']['cells'])} primary cells hashed.")

if __name__ == "__main__":
    generate_post_colab_provenance()
