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
    cache_dir = eval_dir / "cache_wwm_word"

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return False

    records_by_model = defaultdict(list)
    for f in sorted(cache_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as jf:
            d = json.load(jf)
            m_clean = d.get("clean_model_name")
            records_by_model[m_clean].append(d)

    logger.info(f"Loaded records for {len(records_by_model)} models.")

    for m_clean, cells in records_by_model.items():
        m_name = cells[0].get("model_name", cells[0].get("model"))
        logger.info(f"Aggregating {len(cells)} cells for model '{m_name}' ({m_clean})...")

        # Metric lineage explicit aggregations
        maritime_top1_list = [c.get("evaluation_metrics", {}).get("maritime_tokens_summary", {}).get("top1_accuracy", 0.0) for c in cells]
        maritime_top5_list = [c.get("evaluation_metrics", {}).get("maritime_tokens_summary", {}).get("top5_accuracy", 0.0) for c in cells]
        maritime_top10_list = [c.get("evaluation_metrics", {}).get("maritime_tokens_summary", {}).get("top10_accuracy", 0.0) for c in cells]
        rare_top1_list = [c.get("evaluation_metrics", {}).get("rare_maritime_tokens_summary", {}).get("top1_accuracy", 0.0) for c in cells]
        overall_top1_list = [c.get("evaluation_metrics", {}).get("overall_summary", {}).get("overall_top1_accuracy", 0.0) for c in cells]
        overall_top5_list = [c.get("evaluation_metrics", {}).get("overall_summary", {}).get("word_reconstruction_top5_accuracy", 0.0) for c in cells]
        overall_top10_list = [c.get("evaluation_metrics", {}).get("overall_summary", {}).get("word_reconstruction_top10_accuracy", 0.0) for c in cells]

        maritime_loss_list = [c.get("evaluation_metrics", {}).get("maritime_tokens_summary", {}).get("mlm_loss", 0.0) for c in cells]
        overall_loss_list = [c.get("evaluation_metrics", {}).get("overall_summary", {}).get("overall_mlm_loss", 0.0) for c in cells]
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
            "protocol_specification": cells[0].get("provenance", {}).get("protocol_specification", {
                "masking_mode": "whole_word",
                "evaluation_unit": "word",
                "mask_rate": 0.15,
                "scoring_method": "strict_word_reconstruction",
                "internal_mode": "wwm_word",
                "masking_strategy": "wwm_15"
            }),
            "macro_averages": {
                # Maritime target metrics
                "maritime_word_top1_acc": float(np.mean(maritime_top1_list)),
                "maritime_word_top5_acc": float(np.mean(maritime_top5_list)),
                "maritime_word_top10_acc": float(np.mean(maritime_top10_list)),
                "rare_maritime_word_top1_acc": float(np.mean(rare_top1_list)),
                "maritime_target_mlm_loss": float(np.mean(maritime_loss_list)),
                # Overall metrics
                "overall_word_top1_acc": float(np.mean(overall_top1_list)),
                "overall_word_top5_acc": float(np.mean(overall_top5_list)),
                "overall_word_top10_acc": float(np.mean(overall_top10_list)),
                "overall_mlm_loss": float(np.mean(overall_loss_list)),
                # Operational environment metrics
                "inference_latency_ms": round(avg_latency_ms, 2),
                "throughput_docs_sec": round(avg_throughput, 2)
            },
            "provenance": cells[0].get("provenance", {}),
            "condition_cells_evaluated": [
                {
                    "representation": c.get("representation"),
                    "subset": c.get("subset"),
                    "maritime_word_top1_acc": c.get("evaluation_metrics", {}).get("maritime_tokens_summary", {}).get("top1_accuracy"),
                    "maritime_target_mlm_loss": c.get("evaluation_metrics", {}).get("maritime_tokens_summary", {}).get("mlm_loss"),
                    "overall_word_top1_acc": c.get("evaluation_metrics", {}).get("overall_summary", {}).get("overall_top1_accuracy"),
                    "overall_mlm_loss": c.get("evaluation_metrics", {}).get("overall_summary", {}).get("overall_mlm_loss")
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
                "directory": "outputs/stage-14/evaluations/cache_wwm_word",
                "cell_count": 175,
                "description": "Authoritative 175-cell benchmark evaluation cache (7 models x 5 representations x 5 subsets, WWM 15%, strict word reconstruction)"
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
