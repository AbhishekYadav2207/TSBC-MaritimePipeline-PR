import os
import sys
import json
import argparse
from pathlib import Path

# Add scripts directory to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from validate_stage14_cell_schema import validate_stage14_cell_dict
except ImportError:
    from scripts.validate_stage14_cell_schema import validate_stage14_cell_dict

EXPECTED_MODELS = [
    "allenai_scibert_scivocab_uncased",
    "answerdotai_ModernBERT_base",
    "bert_base_uncased",
    "dmis_lab_biobert_base_cased_v1.2",
    "microsoft_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext",
    "nlpaueb_legal_bert_base_uncased",
    "roberta_base"
]

EXPECTED_REPRESENTATIONS = ["json", "key_value", "mixed", "narrative", "template"]
EXPECTED_SUBSETS = ["balanced_knowledge", "high_knowledge", "low_knowledge", "medium_knowledge", "random_baseline"]

def validate_stage14_directory(cache_dir: Path) -> tuple[bool, dict]:
    """
    Rigorously validates a Stage-14 output directory against the strict ingestion contract.
    Returns (passed: bool, report: dict).
    """
    report = {
        "cache_directory": str(cache_dir).replace("\\", "/"),
        "total_files_found": 0,
        "valid_cells_count": 0,
        "missing_cells": [],
        "duplicate_cells": [],
        "schema_violations": [],
        "protocol_violations": [],
        "provenance_violations": [],
        "cohort_check": {},
        "overall_status": "PENDING"
    }

    if not cache_dir.exists():
        report["overall_status"] = "DIRECTORY_NOT_FOUND"
        report["error"] = f"Directory does not exist: {cache_dir}"
        return False, report

    json_files = sorted(list(cache_dir.glob("*.json")))
    report["total_files_found"] = len(json_files)

    expected_combinations = {
        (m, r, s) for m in EXPECTED_MODELS for r in EXPECTED_REPRESENTATIONS for s in EXPECTED_SUBSETS
    }
    seen_combinations = set()

    model_counts = {m: 0 for m in EXPECTED_MODELS}
    rep_counts = {r: 0 for r in EXPECTED_REPRESENTATIONS}
    sub_counts = {s: 0 for s in EXPECTED_SUBSETS}

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            report["schema_violations"].append(f"{f.name}: Invalid JSON syntax ({e})")
            continue

        parts = f.stem.split("__")
        if len(parts) != 3:
            report["protocol_violations"].append(f"{f.name}: Filename does not follow <model>__<rep>__<sub> convention")
            continue

        m_clean, rep, sub = parts
        comb = (m_clean, rep, sub)

        if comb in seen_combinations:
            report["duplicate_cells"].append(f"{f.name}: Duplicate cell {comb}")
        seen_combinations.add(comb)

        if m_clean in model_counts:
            model_counts[m_clean] += 1
        if rep in rep_counts:
            rep_counts[rep] += 1
        if sub in sub_counts:
            sub_counts[sub] += 1

        # Protocol fields check
        meta = data.get("experiment_metadata", {})
        cond = data.get("condition", "")
        mask_strat = meta.get("masking_strategy") or data.get("masking_strategy")
        if not mask_strat and cond.endswith("random_15"):
            mask_strat = "random_15"
        mask_mode = data.get("masking_mode") or meta.get("masking_mode")
        eval_unit = data.get("evaluation_unit") or meta.get("evaluation_unit")
        mask_rate = data.get("mask_rate") or meta.get("mask_rate", 0.15)

        if mask_mode != "subword":
            report["protocol_violations"].append(f"{f.name}: masking_mode is '{mask_mode}', expected 'subword'")
        if eval_unit != "subword":
            report["protocol_violations"].append(f"{f.name}: evaluation_unit is '{eval_unit}', expected 'subword'")
        if mask_strat != "random_15":
            report["protocol_violations"].append(f"{f.name}: masking_strategy is '{mask_strat}', expected 'random_15'")
        if abs(mask_rate - 0.15) > 1e-5:
            report["protocol_violations"].append(f"{f.name}: mask_rate is {mask_rate}, expected 0.15")

        # Schema validation
        is_schema_valid, schema_errs = validate_stage14_cell_dict(data)
        if not is_schema_valid:
            for err in schema_errs:
                report["schema_violations"].append(f"{f.name}: {err}")

    # Check for missing cells
    missing = expected_combinations - seen_combinations
    report["missing_cells"] = [f"{m}__{r}__{s}" for m, r, s in sorted(missing)]
    report["valid_cells_count"] = max(0, len(seen_combinations) - len(report["schema_violations"]) - len(report["protocol_violations"]))

    report["cohort_check"] = {
        "models_observed": {k: v for k, v in model_counts.items()},
        "representations_observed": {k: v for k, v in rep_counts.items()},
        "subsets_observed": {k: v for k, v in sub_counts.items()}
    }

    has_errors = bool(
        report["missing_cells"] or
        report["duplicate_cells"] or
        report["schema_violations"] or
        report["protocol_violations"] or
        report["provenance_violations"] or
        report["total_files_found"] != 175
    )

    # Classify state
    if report["total_files_found"] == 0:
        classification = "EMPTY"
    elif report["valid_cells_count"] == 175 and not has_errors:
        classification = "COMPLETE"
    elif len(report["schema_violations"]) > 0 or len(report["protocol_violations"]) > 0 or len(report["duplicate_cells"]) > 0:
        classification = "INVALID_PARTIAL"
    else:
        classification = "VALID_PARTIAL"

    report["classification"] = classification

    if classification == "COMPLETE":
        report["overall_status"] = "PASS"
    elif classification in ("VALID_PARTIAL", "INVALID_PARTIAL", "EMPTY"):
        report["overall_status"] = "WAITING_FOR_COMPLETE_STAGE14"
    else:
        report["overall_status"] = "FAIL"

    return (classification == "COMPLETE"), report

def main():
    parser = argparse.ArgumentParser(description="Strict Stage-14 Ingestion Validator")
    parser.add_argument("--cache-dir", type=str, default="outputs/stage-14/evaluations/cache_legacy_subword_15",
                        help="Path to the primary Stage 14 cache directory to validate.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"=====================================================================")
    print(f"VALIDATING STAGE-14 INGESTION CONTRACT: {cache_dir}")
    print(f"=====================================================================")

    passed, report = validate_stage14_directory(cache_dir)

    print(f"Total files found: {report['total_files_found']} (Expected: 175)")
    print(f"Valid cells count: {report.get('valid_cells_count', 0)}")
    print(f"Classification: {report.get('classification', 'UNKNOWN')}")
    print(f"Missing cells: {len(report['missing_cells'])}")
    print(f"Duplicate cells: {len(report['duplicate_cells'])}")
    print(f"Protocol violations: {len(report['protocol_violations'])}")
    print(f"Schema violations: {len(report['schema_violations'])}")
    print(f"OVERALL STATUS: {report['overall_status']}")
    print(f"=====================================================================")

    if not passed:
        if report.get("error"):
            print(f"ERROR: {report['error']}")
        if report.get("overall_status") == "WAITING_FOR_COMPLETE_STAGE14":
            print(f"PIPELINE GUARD ACTIVE: WAITING_FOR_COMPLETE_STAGE14 (Classification: {report.get('classification')}).")
            print("Downstream Stage 15+ execution is BLOCKED until all 175/175 cells complete in Colab.")
        for m in report["missing_cells"][:10]:
            print(f"  MISSING: {m}")
        for p in report["protocol_violations"][:10]:
            print(f"  PROTOCOL ERROR: {p}")
        for s in report["schema_violations"][:10]:
            print(f"  SCHEMA ERROR: {s}")
        sys.exit(1)
    else:
        print("STAGE-14 INGESTION CONTRACT PASSED PERFECTLY!")
        sys.exit(0)

if __name__ == "__main__":
    main()
