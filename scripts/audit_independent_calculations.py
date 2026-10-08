import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

def independent_audit_directory(stage14_dir: Path, epsilon: float = 1e-6) -> dict:
    """
    Independently recalculates all summary statistics directly from raw cell JSON files.
    Completely independent of production Stage 15/16 aggregation modules.
    """
    json_files = sorted(list(stage14_dir.glob("*.json")))
    if not json_files:
        return {"error": f"No JSON files found in {stage14_dir}", "status": "FAIL"}

    # Tracking containers
    cell_records = []
    by_model = defaultdict(list)
    by_rep = defaultdict(list)
    by_sub = defaultdict(list)
    by_model_rep = defaultdict(list)

    total_masked_positions = 0
    total_evaluated_positions = 0

    for f in json_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Handle both strict contract fields and legacy nested fields
        model_name = data.get("model_id") or data.get("clean_model_name") or data.get("model_name")
        rep = data.get("representation")
        sub = data.get("subset")

        # Independent extraction of metrics
        if "overall" in data and isinstance(data["overall"], dict):
            overall_top1 = data["overall"].get("top1")
            overall_top5 = data["overall"].get("top5")
            overall_loss = data["overall"].get("loss")
        else:
            ov = data.get("evaluation_metrics", {}).get("overall_summary", {})
            overall_top1 = ov.get("overall_top1_accuracy")
            overall_top5 = ov.get("top5_accuracy", overall_top1)
            overall_loss = ov.get("overall_mlm_loss")

        if "maritime_target" in data and isinstance(data["maritime_target"], dict):
            mar_top1 = data["maritime_target"].get("top1")
            mar_top5 = data["maritime_target"].get("top5")
            mar_loss = data["maritime_target"].get("loss")
        else:
            mar = data.get("evaluation_metrics", {}).get("maritime_tokens_summary", {})
            mar_top1 = mar.get("top1_accuracy")
            mar_top5 = mar.get("top5_accuracy")
            mar_loss = mar.get("mlm_loss")

        if "rare_target" in data and isinstance(data["rare_target"], dict):
            rare_top1 = data["rare_target"].get("top1")
            rare_top5 = data["rare_target"].get("top5")
            rare_loss = data["rare_target"].get("loss")
        else:
            rare = data.get("evaluation_metrics", {}).get("rare_maritime_tokens_summary", {})
            rare_top1 = rare.get("top1_accuracy")
            rare_top5 = rare.get("top5_accuracy")
            rare_loss = rare.get("mlm_loss")

        tok_stats = data.get("token_statistics", {})
        masked_pos = tok_stats.get("masked_positions") or data.get("masked_token_count") or data.get("evaluation_metrics", {}).get("overall_summary", {}).get("total_masked_tokens", 0)
        total_masked_positions += masked_pos

        record = {
            "file": f.name,
            "model": model_name,
            "representation": rep,
            "subset": sub,
            "overall_top1": overall_top1,
            "overall_top5": overall_top5,
            "overall_loss": overall_loss,
            "maritime_top1": mar_top1,
            "maritime_top5": mar_top5,
            "maritime_loss": mar_loss,
            "rare_top1": rare_top1,
            "rare_top5": rare_top5,
            "rare_loss": rare_loss,
            "masked_positions": masked_pos
        }
        cell_records.append(record)
        by_model[model_name].append(record)
        by_rep[rep].append(record)
        by_sub[sub].append(record)
        by_model_rep[(model_name, rep)].append(record)

    # Independent Aggregations
    def mean_valid(vals):
        valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(valid) / len(valid) if valid else None

    model_aggregates = {}
    for m, recs in by_model.items():
        model_aggregates[m] = {
            "cell_count": len(recs),
            "mean_overall_top1": mean_valid([r["overall_top1"] for r in recs]),
            "mean_maritime_top1": mean_valid([r["maritime_top1"] for r in recs]),
            "mean_rare_top1": mean_valid([r["rare_top1"] for r in recs]),
            "mean_overall_loss": mean_valid([r["overall_loss"] for r in recs]),
            "mean_maritime_loss": mean_valid([r["maritime_loss"] for r in recs]),
            "total_masked_positions": sum(r["masked_positions"] for r in recs)
        }

    rep_aggregates = {}
    for r, recs in by_rep.items():
        rep_aggregates[r] = {
            "cell_count": len(recs),
            "mean_overall_top1": mean_valid([r["overall_top1"] for r in recs]),
            "mean_maritime_top1": mean_valid([r["maritime_top1"] for r in recs]),
            "mean_overall_loss": mean_valid([r["overall_loss"] for r in recs])
        }

    sub_aggregates = {}
    for s, recs in by_sub.items():
        sub_aggregates[s] = {
            "cell_count": len(recs),
            "mean_overall_top1": mean_valid([r["overall_top1"] for r in recs]),
            "mean_maritime_top1": mean_valid([r["maritime_top1"] for r in recs]),
            "mean_overall_loss": mean_valid([r["overall_loss"] for r in recs])
        }

    # Model Rankings
    sorted_by_maritime = sorted(
        model_aggregates.items(),
        key=lambda x: (x[1]["mean_maritime_top1"] if x[1]["mean_maritime_top1"] is not None else -1),
        reverse=True
    )
    rankings_maritime = [m for m, _ in sorted_by_maritime]

    sorted_by_overall = sorted(
        model_aggregates.items(),
        key=lambda x: (x[1]["mean_overall_top1"] if x[1]["mean_overall_top1"] is not None else -1),
        reverse=True
    )
    rankings_overall = [m for m, _ in sorted_by_overall]

    return {
        "status": "PASS",
        "total_cells": len(cell_records),
        "total_models": len(by_model),
        "total_representations": len(by_rep),
        "total_subsets": len(by_sub),
        "total_masked_positions": total_masked_positions,
        "model_aggregates": model_aggregates,
        "representation_aggregates": rep_aggregates,
        "subset_aggregates": sub_aggregates,
        "rankings_by_maritime_top1": rankings_maritime,
        "rankings_by_overall_top1": rankings_overall
    }

def main():
    parser = argparse.ArgumentParser(description="Stage-14 Independent Calculations Auditor")
    parser.add_argument("--stage14-dir", type=str, required=True, help="Directory containing Stage 14 JSON cell outputs")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Floating-point comparison tolerance (default: 1e-6)")
    parser.add_argument("--compare-with", type=str, default=None, help="Optional second directory or reference JSON to cross-check")
    args = parser.parse_args()

    stage14_dir = Path(args.stage14_dir)
    print(f"=====================================================================")
    print(f"INDEPENDENT CALCULATIONS AUDITOR: {stage14_dir}")
    print(f"=====================================================================")

    audit = independent_audit_directory(stage14_dir, epsilon=args.epsilon)
    if audit.get("status") != "PASS":
        print(f"AUDIT FAILED: {audit.get('error')}")
        sys.exit(1)

    print(f"Cells evaluated: {audit['total_cells']}")
    print(f"Models: {audit['total_models']}")
    print(f"Representations: {audit['total_representations']}")
    print(f"Subsets: {audit['total_subsets']}")
    print(f"Total masked positions: {audit['total_masked_positions']}")
    print(f"\nModel Rankings (by Maritime Top-1):")
    for idx, m in enumerate(audit['rankings_by_maritime_top1'], start=1):
        agg = audit['model_aggregates'][m]
        print(f"  {idx}. {m:55s} | Mar Top1: {agg['mean_maritime_top1']:.4f} | Overall: {agg['mean_overall_top1']:.4f} | Loss: {agg['mean_overall_loss']:.4f}")

    if args.compare_with:
        comp_path = Path(args.compare_with)
        if comp_path.is_dir():
            print(f"\nCross-checking against reference directory: {comp_path}...")
            ref_audit = independent_audit_directory(comp_path, epsilon=args.epsilon)
            discrepancies = []
            for m, agg in audit["model_aggregates"].items():
                if m in ref_audit["model_aggregates"]:
                    ref_agg = ref_audit["model_aggregates"][m]
                    for metric in ["mean_overall_top1", "mean_maritime_top1", "mean_overall_loss"]:
                        val1 = agg[metric]
                        val2 = ref_agg[metric]
                        if val1 is not None and val2 is not None:
                            diff = abs(val1 - val2)
                            if diff > args.epsilon:
                                discrepancies.append(f"{m} {metric}: {val1} vs {val2} (delta={diff:.8f})")
            if discrepancies:
                print(f"FOUND {len(discrepancies)} NUMERICAL DISCREPANCIES EXCEEDING EPSILON {args.epsilon}:")
                for d in discrepancies[:10]:
                    print(f"  - {d}")
                sys.exit(1)
            else:
                print(f"PERFECT AGREEMENT ACROSS ALL MODELS WITHIN EPSILON {args.epsilon}!")

if __name__ == "__main__":
    main()
