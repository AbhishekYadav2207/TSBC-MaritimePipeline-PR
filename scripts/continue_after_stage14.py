"""
Post-Colab Ingestion, Validation & Pipeline Continuation Script
MaritimeBench / MaritimeBERT Pipeline Version 2.1

Usage:
  python scripts/continue_after_stage14.py --validate-only
  python scripts/continue_after_stage14.py --continue
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def run_command(cmd: list, cwd: Path, desc: str):
    print(f"\n>>> Running: {desc}...")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {res.returncode}: {desc}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {desc} completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Ingest Stage 14 Colab outputs and continue downstream stages.")
    parser.add_argument("--validate-only", action="store_true", help="Validate ingested cache without running downstream stages.")
    parser.add_argument("--continue", dest="run_continue", action="store_true", help="Validate cache and execute Stages 15 through 18.")
    args = parser.parse_args()

    if not args.validate_only and not args.run_continue:
        print("Please specify either --validate-only or --continue.")
        print("Example: python scripts/continue_after_stage14.py --continue")
        sys.exit(1)

    root = get_project_root()
    cache_dir = root / "outputs" / "stage-14" / "evaluations" / "cache_legacy_subword_15"

    print("=" * 70)
    print("MARITIMEBENCH POST-STAGE-14 INGESTION & CONTINUATION GATE")
    print("=" * 70)

    # 1. Check Cache Existence & Completeness
    if not cache_dir.exists():
        print(f"\n[ERROR] Target cache directory not found:")
        print(f"  {cache_dir}")
        print("\nPlease run Stage 14 in Google Colab, download cache_legacy_subword_15.zip,")
        print("and extract it into the directory above.")
        print("See instructions in: outputs/final_audit/COLAB_STAGE14_RUN_INSTRUCTIONS.md")
        sys.exit(1)

    cell_files = list(cache_dir.glob("*.json"))
    print(f"Found {len(cell_files)} cell files in {cache_dir.name}/")

    if len(cell_files) != 175:
        print(f"\n[GUARD ACTIVE] WAITING_FOR_COMPLETE_STAGE14")
        print(f"Incomplete cache! Expected exactly 175 cell files, but found {len(cell_files)}.")
        print("Downstream Stage 15-18 execution is strictly blocked until all 175/175 cells complete in Colab.")
        sys.exit(1)

    # 2. Ingestion Validation
    validate_script = root / "scripts" / "validate_stage14_ingestion.py"
    run_command([sys.executable, str(validate_script), "--cache_dir", str(cache_dir)], root, "Stage 14 Ingestion & Schema Validation")

    # 3. Independent Calculation Audit
    audit_script = root / "scripts" / "audit_independent_calculations.py"
    run_command([sys.executable, str(audit_script), "--cache_dir", str(cache_dir)], root, "Stage 14 Independent Calculation Audit")

    if args.validate_only:
        print("\n" + "=" * 70)
        print("[SUCCESS] Cache validation and independent calculation audit passed!")
        print("Run with --continue when you are ready to execute Stages 15 through 18.")
        print("=" * 70)
        return

    # 4. Execute Downstream Stages 15 through 18
    s15 = root / "scripts" / "15_cross_model_benchmarking.py"
    run_command([sys.executable, str(s15)], root, "Stage 15: Cross-Model Benchmarking & MECS")

    s16 = root / "scripts" / "16_statistical_analysis.py"
    run_command([sys.executable, str(s16)], root, "Stage 16: Statistical Validation & Crossed ANOVA")

    s17 = root / "scripts" / "17_decision_engine.py"
    run_command([sys.executable, str(s17)], root, "Stage 17: Multi-Dimensional Evidence Decision Engine")

    s18 = root / "scripts" / "18_lint_corpus.py"
    run_command([sys.executable, str(s18)], root, "Stage 18: Corpus Verification & Linting")

    print("\n" + "=" * 70)
    print("[SUCCESS] FULL PIPELINE CONTINUATION COMPLETED SUCCESSFULLY!")
    print("All downstream outputs regenerated and independently verified.")
    print("=" * 70)

if __name__ == "__main__":
    main()
