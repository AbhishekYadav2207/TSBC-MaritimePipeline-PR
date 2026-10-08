"""
Generator for True Model-Level Aggregate Summaries for Stage 14
TSBC-MaritimePipeline-Version2.1

Aggregates the authoritative 175-cell cache records (25 cells per model) into true
model-level summary artifacts at outputs/stage-14/evaluations/<clean_model_name>.json.
Updates outputs/maritimebench_output_inventory.json and .md.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from pipeline_utils import get_project_root, setup_logging

logger = setup_logging("generate_stage14_model_summaries")

def generate_stage14_aggregates():
    root = get_project_root()
    eval_dir = root / "outputs" / "stage-14" / "evaluations"
    cache_dir = eval_dir / "cache_legacy_subword_15"

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return False

    records_by_model = defaultdict(list)
    for f in sorted(cache_dir.glob("*.json")):
        if f.is_dir():
            continue
        with open(f, "r", encoding="utf-8") as jf:
            d = json.load(jf)
            m_clean = d.get("clean_model_name")
            records_by_model[m_clean].append(d)

    logger.info(f"Loaded records for {len(records_by_model)} models.")

    for m_clean, cells in records_by_model.items():
        m_name = cells[0].get("model_name", cells[0].get("model", cells[0].get("model_id")))
        logger.info(f"Aggregating {len(cells)} cells for model '{m_name}' ({m_clean})...")

        # Metric lineage explicit aggregations
        def _get_val(cell, category, key, legacy_path):
            v = cell.get(category, {}).get(key)
            if v is None and legacy_path:
                v = cell.get("evaluation_metrics", {})
                for p in legacy_path:
                    if isinstance(v, dict):
                        v = v.get(p)
                    else:
                        v = None
            return v

        maritime_top1_list = [v for c in cells if (v := _get_val(c, "maritime_target", "top1", ["maritime_tokens_summary", "top1_accuracy"])) is not None]
        maritime_top5_list = [v for c in cells if (v := _get_val(c, "maritime_target", "top5", ["maritime_tokens_summary", "top5_accuracy"])) is not None]
        rare_top1_list = [v for c in cells if (v := _get_val(c, "rare_target", "top1", ["rare_maritime_tokens_summary", "top1_accuracy"])) is not None]
        overall_top1_list = [v for c in cells if (v := _get_val(c, "overall", "top1", ["overall_summary", "overall_top1_accuracy"])) is not None]

        maritime_loss_list = [v for c in cells if (v := _get_val(c, "maritime_target", "loss", ["maritime_tokens_summary", "mlm_loss"])) is not None]
        overall_loss_list = [v for c in cells if (v := _get_val(c, "overall", "loss", ["overall_summary", "overall_mlm_loss"])) is not None]
        eval_time_list = [c.get("evaluation_metrics", {}).get("evaluation_time_sec", 0.0) for c in cells]
        doc_count_list = [c.get("evaluated_doc_count", 200.0) for c in cells]

        # Operational latency/throughput
        cell_latencies = [(t / d) * 1000.0 for t, d in zip(eval_time_list, doc_count_list) if d > 0]
        avg_latency_ms = float(np.mean(cell_latencies)) if cell_latencies else 0.0
        avg_throughput = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

        summary_obj = {
            "model_name": m_name,
            "clean_model_name": m_clean,
            "summary_type": "authoritative_model_level_macro_aggregate",
            "evaluation_scope": {
                "total_matched_conditions": len(cells),
                "representations_count": 5,
                "subsets_count": 5,
                "aggregation_method": "unweighted_macro_average_over_25_conditions"
            },
            "protocol_specification": {
                "masking_mode": cells[0].get("masking_mode", "subword"),
                "evaluation_unit": cells[0].get("evaluation_unit", "subword"),
                "mask_rate": cells[0].get("mask_rate", 0.15),
                "scoring_method": cells[0].get("scoring_method", "subword_reconstruction"),
                "masking_strategy": cells[0].get("experiment_metadata", {}).get("masking_strategy", "random_15")
            },
            "macro_averages": {
                # Maritime target metrics
                "maritime_subword_top1_acc": float(np.mean(maritime_top1_list)) if maritime_top1_list else None,
                "maritime_subword_top5_acc": float(np.mean(maritime_top5_list)) if maritime_top5_list else None,
                "rare_maritime_subword_top1_acc": float(np.mean(rare_top1_list)) if rare_top1_list else None,
                "maritime_target_mlm_loss": float(np.mean(maritime_loss_list)) if maritime_loss_list else None,
                # Overall metrics
                "overall_subword_top1_acc": float(np.mean(overall_top1_list)) if overall_top1_list else None,
                "overall_mlm_loss": float(np.mean(overall_loss_list)) if overall_loss_list else None,
                # Operational environment metrics
                "inference_latency_ms": round(avg_latency_ms, 2),
                "throughput_docs_sec": round(avg_throughput, 2)
            },
            "provenance": cells[0].get("provenance", {}),
            "condition_cells_evaluated": [
                {
                    "representation": c.get("representation"),
                    "subset": c.get("subset"),
                    "maritime_subword_top1_acc": c.get("maritime_target", {}).get("top1"),
                    "maritime_target_mlm_loss": c.get("maritime_target", {}).get("loss"),
                    "overall_subword_top1_acc": c.get("overall", {}).get("top1"),
                    "overall_mlm_loss": c.get("overall", {}).get("loss")
                }
                for c in cells
            ]
        }

        out_path = eval_dir / f"{m_clean}.json"
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(summary_obj, out_f, indent=2)

    logger.info("Updated all 7 top-level model evaluation summaries successfully.")

    # Update output inventory
    inv_json_path = root / "outputs" / "maritimebench_output_inventory.json"
    if inv_json_path.exists():
        with open(inv_json_path, "r", encoding="utf-8") as f_inv:
            inv = json.load(f_inv)
        if "stage_14" in inv:
            inv["stage_14"]["authoritative_cache"] = {
                "directory": "outputs/stage-14/evaluations/cache_legacy_subword_15",
                "cell_count": 175,
                "description": "Authoritative 175-cell benchmark evaluation cache (7 models x 5 representations x 5 subsets, Random 15%, subword reconstruction)"
            }
            inv["stage_14"]["model_summaries"] = {
                "directory": "outputs/stage-14/evaluations",
                "count": 7,
                "description": "Macro-averaged 25-condition aggregate evaluation summaries for each active benchmark candidate"
            }
        with open(inv_json_path, "w", encoding="utf-8") as f_inv:
            json.dump(inv, f_inv, indent=2)
        logger.info(f"Updated inventory: {inv_json_path}")

    return True

if __name__ == "__main__":
    generate_stage14_aggregates()
