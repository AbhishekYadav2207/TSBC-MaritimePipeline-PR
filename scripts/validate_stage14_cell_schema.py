import json
import math
from pathlib import Path

def validate_stage14_cell_dict(record: dict) -> tuple[bool, list[str]]:
    """
    Validates a single Stage 14 evaluation cell record against the strict output contract.
    Returns (is_valid, error_list).
    """
    errors = []
    
    # 1. Required top-level keys
    required_top = [
        "model_id", "representation", "subset", "condition", "seed",
        "masking_mode", "evaluation_unit", "mask_rate", "scoring_method",
        "overall", "maritime_target", "rare_target", "token_statistics", "provenance"
    ]
    for k in required_top:
        if k not in record:
            errors.append(f"Missing required top-level key: '{k}'")
            
    if errors:
        return False, errors

    # 2. Representation & Subset allowed values
    valid_reps = {"json", "key_value", "mixed", "narrative", "template"}
    valid_subs = {"balanced_knowledge", "high_knowledge", "low_knowledge", "medium_knowledge", "random_baseline"}
    if record["representation"] not in valid_reps:
        errors.append(f"Invalid representation: '{record['representation']}'. Expected one of {sorted(valid_reps)}")
    if record["subset"] not in valid_subs:
        errors.append(f"Invalid subset: '{record['subset']}'. Expected one of {sorted(valid_subs)}")

    # 3. Masking mode & evaluation unit
    valid_mask_modes = {"subword", "whole_word"}
    valid_eval_units = {"subword", "word"}
    if record["masking_mode"] not in valid_mask_modes:
        errors.append(f"Invalid masking_mode: '{record['masking_mode']}'")
    if record["evaluation_unit"] not in valid_eval_units:
        errors.append(f"Invalid evaluation_unit: '{record['evaluation_unit']}'")
    if abs(record["mask_rate"] - 0.15) > 1e-5:
        errors.append(f"Invalid mask_rate: {record['mask_rate']}. Expected 0.15")

    # 4. Overall metrics
    overall = record.get("overall", {})
    for metric in ["top1", "top5", "loss"]:
        if metric not in overall:
            errors.append(f"Missing overall metric: '{metric}'")
        else:
            v = overall[metric]
            if v is not None:
                if not isinstance(v, (int, float)) or (metric in ("top1", "top5") and (v < 0.0 or v > 1.0)):
                    errors.append(f"Invalid overall.{metric} value: {v}")

    # 5. Maritime target metrics
    mar = record.get("maritime_target", {})
    for metric in ["top1", "top5", "loss"]:
        if metric not in mar:
            errors.append(f"Missing maritime_target metric: '{metric}'")
        else:
            v = mar[metric]
            if v is not None:
                if not isinstance(v, (int, float)) or (metric in ("top1", "top5") and (v < 0.0 or v > 1.0)):
                    errors.append(f"Invalid maritime_target.{metric} value: {v}")

    # 6. Rare target metrics
    rare = record.get("rare_target", {})
    for metric in ["top1", "top5", "loss"]:
        if metric not in rare:
            errors.append(f"Missing rare_target metric: '{metric}'")
        else:
            v = rare[metric]
            if v is not None:
                if not isinstance(v, (int, float)) or (metric in ("top1", "top5") and (v < 0.0 or v > 1.0)):
                    errors.append(f"Invalid rare_target.{metric} value: {v}")

    # 7. Token statistics
    tok_stats = record.get("token_statistics", {})
    for k in ["masked_positions", "evaluated_positions", "maritime_target_count", "rare_target_count"]:
        if k not in tok_stats:
            errors.append(f"Missing token_statistics field: '{k}'")
        elif not isinstance(tok_stats[k], int) or tok_stats[k] < 0:
            errors.append(f"Invalid token_statistics.{k}: {tok_stats[k]}")

    # 8. Provenance
    prov = record.get("provenance", {})
    for k in ["model_identifier", "tokenizer_identifier", "execution_type"]:
        if k not in prov or not prov[k]:
            errors.append(f"Missing or empty provenance field: '{k}'")

    # 9. No NaN values allowed (must be None/null)
    def check_nan(obj, path=""):
        if isinstance(obj, dict):
            for sub_k, sub_v in obj.items():
                check_nan(sub_v, f"{path}.{sub_k}" if path else sub_k)
        elif isinstance(obj, float) and math.isnan(obj):
            errors.append(f"NaN detected at '{path}'. Unobserved values must be null/None, never NaN.")

    check_nan(record)

    return len(errors) == 0, errors

def validate_stage14_cell_file(file_path: Path) -> tuple[bool, list[str]]:
    p = Path(file_path)
    if not p.exists():
        return False, [f"File does not exist: {p}"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return validate_stage14_cell_dict(data)
    except Exception as e:
        return False, [f"JSON parsing error: {e}"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        valid, errs = validate_stage14_cell_file(path)
        if valid:
            print(f"PASS: {path.name} is schema-valid.")
            sys.exit(0)
        else:
            print(f"FAIL: {path.name} has {len(errs)} schema errors:")
            for err in errs:
                print(f"  - {err}")
            sys.exit(1)
    else:
        print("Usage: python validate_stage14_cell_schema.py <path_to_cell_json>")
