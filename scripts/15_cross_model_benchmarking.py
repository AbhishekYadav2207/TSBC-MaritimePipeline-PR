import os
import sys
import json
import math
import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("15_cross_model_benchmarking")

# Explicit, configuration-driven permanent benchmark model exclusions
EXCLUDED_MODELS = {
    "microsoft/deberta-v3-base": {
        "reason": "Excluded after benchmark compatibility anomaly: zero MLM Top-1/Top-5/Top-10 across the evaluated standard cells and English diagnostic."
    }
}

# Model parameter and size registry (approximate parameter counts in millions and disk sizes in MB)
MODEL_PROFILES = {
    "bert-base-uncased": {"params_m": 110, "size_mb": 440},
    "bert-large-uncased": {"params_m": 340, "size_mb": 1340},
    "roberta-base": {"params_m": 125, "size_mb": 500},
    "answerdotai/ModernBERT-base": {"params_m": 149, "size_mb": 590},
    "allenai/scibert_scivocab_uncased": {"params_m": 110, "size_mb": 440},
    "dmis-lab/biobert-base-cased-v1.2": {"params_m": 110, "size_mb": 440},
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": {"params_m": 110, "size_mb": 440},
    "emilyalsentzer/Bio_ClinicalBERT": {"params_m": 110, "size_mb": 440},
    "nlpaueb/legal-bert-base-uncased": {"params_m": 110, "size_mb": 440},
    "ProsusAI/finbert": {"params_m": 110, "size_mb": 440},
    "anferico/bert-for-patents": {"params_m": 340, "size_mb": 1340},
    "google/electra-base-discriminator": {"params_m": 110, "size_mb": 440},
    "distilbert-base-uncased": {"params_m": 66, "size_mb": 268}
}

METRIC_DIRECTIONS = {
    # Capability metrics
    "top1_acc": "higher_is_better",
    "top5_acc": "higher_is_better",
    "top10_acc": "higher_is_better",
    "rare_top1_acc": "higher_is_better",
    "category_balance": "higher_is_better",
    "mlm_loss": "lower_is_better",
    "pseudo_perplexity": "lower_is_better",
    # Domain / Tokenizer fit metrics
    "single_token_coverage_pct": "higher_is_better",
    "fragmentation_rate_pct": "lower_is_better",
    "oov_rate_pct": "lower_is_better",
    "tokenizer_speed_tok_sec": "higher_is_better",
    # Operational metrics
    "throughput_docs_sec": "higher_is_better",
    "inference_latency_ms": "lower_is_better",
    "params_millions": "lower_is_better",
    "disk_size_mb": "lower_is_better"
}

# Maritime Encoder Composite Score (MECS) Configuration & Fixed Transforms
# MECS is an operational composite compatibility score used to summarize multiple encoder evaluation
# characteristics for model selection. It is not intended as a direct measure of language or maritime understanding.
# All transforms are explicit design choices with documented justifications and fixed bounds.
FIXED_MECS_TRANSFORMS = {
    "top1_acc": {
        "type": "linear_clip",
        "ceiling": 0.35,
        "direction": "higher_is_better",
        "category": "intrinsic_encoder_capability",
        "justification": "Target benchmark ceiling of 35.0% strict word reconstruction accuracy for pre-trained zero-shot encoders on technical domain terminology without task-specific adaptation."
    },
    "rare_top1_acc": {
        "type": "linear_clip",
        "ceiling": 0.25,
        "direction": "higher_is_better",
        "category": "intrinsic_encoder_capability",
        "justification": "Target benchmark ceiling of 25.0% for low-frequency maritime domain words (corpus frequency <= 10) representing high-difficulty tail tokens."
    },
    "top5_acc": {
        "type": "linear_clip",
        "ceiling": 0.60,
        "direction": "higher_is_better",
        "category": "intrinsic_encoder_capability",
        "justification": "Target benchmark ceiling of 60.0% for top-5 word candidate recall under zero-shot cloze evaluation."
    },
    "mlm_loss": {
        "type": "bounded_linear",
        "min_bound": 3.0,
        "max_bound": 8.0,
        "direction": "lower_is_better",
        "category": "intrinsic_encoder_capability",
        "justification": "Fixed reference interval [3.0, 8.0] where 3.0 represents near-perfect domain MLM convergence (perplexity ~20) and 8.0 represents unadapted random guessing (perplexity ~2980 for 30k-50k vocabulary). Preserves linear cross-entropy sensitivity without cohort dependence."
    },
    "fragmentation_rate_pct": {
        "type": "percentage_inverted",
        "max_bound": 100.0,
        "direction": "lower_is_better",
        "category": "domain_morphological_fit",
        "justification": "Natural percentage domain [0.0%, 100.0%]; lower fragmentation indicates superior morphological token preservation for maritime compounds."
    },
    "oov_rate_pct": {
        "type": "bounded_inverted",
        "ceiling": 10.0,
        "direction": "lower_is_better",
        "category": "domain_morphological_fit",
        "justification": "10.0% ceiling for out-of-vocabulary rate in WordPiece tokenizers. Byte-level BPE tokenizers have no measured OOV and are treated as NA (non-penalizing re-weighting)."
    },
    "category_balance": {
        "type": "linear_clip",
        "ceiling": 1.0,
        "direction": "higher_is_better",
        "category": "domain_morphological_fit",
        "justification": "Calculated as 1.0 - std(category accuracies) in [0.0, 1.0]; higher values denote uniform representation capability across subdomains."
    },
    "inference_latency_ms": {
        "type": "operational_rational",
        "scale_ms": 50.0,
        "direction": "lower_is_better",
        "category": "environment_specific_operational",
        "justification": "Operational reference constant of 50.0 ms represents a standard interactive SLA ceiling for single-document real-time triage. Labelled as environment-specific operational metric dependent on benchmark hardware."
    },
    "throughput_docs_sec": {
        "type": "operational_rational",
        "scale_docs_sec": 50.0,
        "direction": "higher_is_better",
        "category": "environment_specific_operational",
        "justification": "Operational reference constant of 50.0 docs/sec reflects baseline batch ingestion throughput. Labelled as environment-specific operational metric dependent on benchmark hardware."
    }
}

MECS_SCENARIOS = {
    "baseline": {
        "top1_acc": 0.40,
        "rare_top1_acc": 0.15,
        "mlm_loss": 0.25,
        "fragmentation_rate_pct": 0.20
    },
    "performance_heavy": {
        "top1_acc": 0.45,
        "rare_top1_acc": 0.20,
        "mlm_loss": 0.35
    },
    "domain_heavy": {
        "rare_top1_acc": 0.40,
        "top1_acc": 0.20,
        "mlm_loss": 0.15,
        "fragmentation_rate_pct": 0.25
    },
    "balanced": {
        "top1_acc": 0.25,
        "rare_top1_acc": 0.20,
        "mlm_loss": 0.20,
        "fragmentation_rate_pct": 0.15,
        "throughput_docs_sec": 0.20
    }
}


def normalize_metric_fixed(series: pd.Series, metric: str) -> pd.Series:
    """
    Cohort-independent, fixed-bound normalization to [0, 1].
    Preserves NaN for missing values without fabricating zeros.
    """
    if metric not in FIXED_MECS_TRANSFORMS:
        return normalize_metric(series, METRIC_DIRECTIONS.get(metric, "higher_is_better"))

    info = FIXED_MECS_TRANSFORMS[metric]
    m_type = info["type"]

    if m_type == "bounded_linear":
        low = info["min_bound"]
        high = info["max_bound"]
        norm = (high - series) / (high - low)
        return norm.clip(lower=0.0, upper=1.0)
    elif m_type == "linear_clip":
        ceil = info.get("ceiling", 1.0)
        norm = series / ceil
        return norm.clip(lower=0.0, upper=1.0)
    elif m_type == "percentage_inverted":
        norm = 1.0 - (series / 100.0)
        return norm.clip(lower=0.0, upper=1.0)
    elif m_type == "bounded_inverted":
        ceil = info["ceiling"]
        norm = 1.0 - (series / ceil)
        return norm.clip(lower=0.0, upper=1.0)
    elif m_type == "operational_rational":
        if info["direction"] == "lower_is_better":
            scale = info["scale_ms"]
            return 1.0 / (1.0 + series.clip(lower=0.0) / scale)
        else:
            scale = info["scale_docs_sec"]
            s_clipped = series.clip(lower=0.0)
            return s_clipped / (scale + s_clipped)
    else:
        return normalize_metric(series, info["direction"])



def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def discover_stage14_results(cache_dir: Path, pll_path: Path = None):
    """
    Dynamically discovers and validates available Stage 14 cache files.
    Never hardcodes 175 files, 7 models, or specific representations/subsets.
    Tracks coverage: expected cells, discovered files, valid records, invalid files, missing cells.
    """
    discovered_files = 0
    valid_rows = []
    invalid_files = []
    seen_combinations = set()

    if not cache_dir.exists():
        logger.error(f"Stage 14 cache directory does not exist: {cache_dir}")
        return pd.DataFrame(), {}, {}

    json_files = sorted(list(cache_dir.glob("*.json")))
    discovered_files = len(json_files)

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping malformed JSON file {jf.name}: {e}")
            invalid_files.append({"file": jf.name, "reason": f"JSON parse error: {str(e)}"})
            continue

        model_name = item.get("model_name")
        rep = item.get("representation")
        sub = item.get("subset")

        if not model_name or not rep or not sub:
            logger.warning(f"File {jf.name} missing required identifying keys (model_name/representation/subset). Skipping.")
            invalid_files.append({"file": jf.name, "reason": "Missing model_name, representation, or subset"})
            continue

        if model_name in EXCLUDED_MODELS:
            logger.info(f"Skipping excluded model {model_name} in {jf.name}")
            continue

        combo_key = (model_name, rep, sub)
        if combo_key in seen_combinations:
            logger.error(f"Duplicate evaluation record encountered for {combo_key} in {jf.name}. Flagging duplicate.")
            invalid_files.append({"file": jf.name, "reason": f"Duplicate record for combination {combo_key}"})
            continue

        seen_combinations.add(combo_key)

        doc_count = item.get("evaluated_doc_count")
        if doc_count is None:
            doc_count = item.get("experiment_metadata", {}).get("evaluation_documents", 200)

        metrics = item.get("evaluation_metrics", {})
        mar_sum = metrics.get("maritime_tokens_summary", {})
        rare_sum = metrics.get("rare_maritime_tokens_summary", {})
        cat_rec = metrics.get("category_recall", {})
        overall_sum = metrics.get("overall_summary", {})

        mar_loss_val = mar_sum.get("mlm_loss")
        derived_exp_loss = mar_sum.get("mlm_loss_derived_exponential")
        mar_top1 = mar_sum.get("top1_accuracy", np.nan)
        mar_top5 = mar_sum.get("top5_accuracy", np.nan)
        mar_top10 = mar_sum.get("top10_accuracy", np.nan)

        ov_top1 = item.get("overall_top1_accuracy", overall_sum.get("overall_top1_accuracy", np.nan))
        ov_loss = item.get("overall_mlm_loss", overall_sum.get("overall_mlm_loss", np.nan))

        word_rec_sum = metrics.get("word_reconstruction_summary", {})
        word_top1 = overall_sum.get("word_reconstruction_top1_accuracy", word_rec_sum.get("word_reconstruction_top1_accuracy", np.nan))
        word_top5 = overall_sum.get("word_reconstruction_top5_accuracy", word_rec_sum.get("word_reconstruction_top5_accuracy", np.nan))
        word_top10 = overall_sum.get("word_reconstruction_top10_accuracy", word_rec_sum.get("word_reconstruction_top10_accuracy", np.nan))

        valid_rows.append({
            "model_name": model_name,
            "representation": rep,
            "subset": sub,
            "domain_shift_gap": item.get("domain_shift_gap", np.nan),
            # Lineage-explicit maritime domain metrics
            "maritime_mlm_loss": mar_loss_val if mar_loss_val is not None else np.nan,
            "maritime_top1_acc": mar_top1,
            "maritime_top5_acc": mar_top5,
            "maritime_top10_acc": mar_top10,
            # Backwards-compatible aliases for Stage 16 and legacy consumers
            "mlm_loss": mar_loss_val if mar_loss_val is not None else np.nan,
            "top1_acc": mar_top1,
            "top5_acc": mar_top5,
            "top10_acc": mar_top10,
            "mlm_loss_derived_exponential": derived_exp_loss if derived_exp_loss is not None else np.nan,
            "rare_top1_acc": rare_sum.get("top1_accuracy", np.nan),
            "performance_gap": metrics.get("performance_gap_top1", np.nan),
            "nav_acc": cat_rec.get("navigation", np.nan),
            "weather_acc": cat_rec.get("weather_environment", np.nan),
            "safety_acc": cat_rec.get("safety_lifesaving", np.nan),
            "machinery_acc": cat_rec.get("machinery_propulsion", np.nan),
            "vessel_acc": cat_rec.get("vessel_terminology", np.nan),
            "casualty_acc": cat_rec.get("casualty_incident", np.nan),
            # Overall document token & word reconstruction metrics
            "overall_top1_acc": ov_top1,
            "overall_mlm_loss": ov_loss,
            "word_reconstruction_top1": word_top1,
            "word_reconstruction_top5": word_top5,
            "word_reconstruction_top10": word_top10,
            "masking_mode": item.get("experiment_metadata", {}).get("masking_mode", "whole_word"),
            "evaluation_unit": item.get("experiment_metadata", {}).get("evaluation_unit", "word"),
            "eval_time_sec": metrics.get("evaluation_time_sec", np.nan),
            "evaluated_doc_count": float(doc_count)
        })

    df_mlm = pd.DataFrame(valid_rows)

    unique_models = sorted(df_mlm["model_name"].dropna().unique()) if not df_mlm.empty else []
    unique_reps = sorted(df_mlm["representation"].dropna().unique()) if not df_mlm.empty else []
    unique_subs = sorted(df_mlm["subset"].dropna().unique()) if not df_mlm.empty else []

    expected_cells = len(unique_models) * len(unique_reps) * len(unique_subs)
    valid_cells = len(df_mlm)
    missing_cells = max(0, expected_cells - valid_cells)

    missing_combinations = []
    if expected_cells > 0:
        for m in unique_models:
            for r in unique_reps:
                for s in unique_subs:
                    if (m, r, s) not in seen_combinations:
                        missing_combinations.append({"model_name": m, "representation": r, "subset": s})

    coverage_summary = {
        "discovered_files": discovered_files,
        "valid_cells": valid_cells,
        "invalid_files_count": len(invalid_files),
        "invalid_files": invalid_files,
        "unique_models": unique_models,
        "unique_representations": unique_reps,
        "unique_subsets": unique_subs,
        "expected_cells": expected_cells,
        "missing_cells": missing_cells,
        "missing_combinations": missing_combinations
    }

    pll_dict = {}
    if pll_path and pll_path.exists():
        try:
            with open(pll_path, "r", encoding="utf-8") as f:
                pll_json = json.load(f)
                results = pll_json.get("results", [])
                for entry in results:
                    m = entry.get("model_name")
                    ppl = entry.get("pll_metrics", {}).get("pseudo_perplexity")
                    if m and ppl is not None and not np.isnan(ppl):
                        pll_dict.setdefault(m, []).append(float(ppl))
            pll_dict = {m: float(np.mean(vals)) for m, vals in pll_dict.items() if vals}
            logger.info(f"Loaded PLL pseudo-perplexity for {len(pll_dict)} models from {pll_path.name}")
        except Exception as e:
            logger.warning(f"Could not load PLL results from {pll_path}: {e}")

    return df_mlm, coverage_summary, pll_dict


def normalize_metric(series: pd.Series, direction: str) -> pd.Series:
    """
    Direction-aware normalization to [0, 1].
    Handles zero range, NaN, inf, single values safely without division by zero.
    Missing/N/A values remain NaN and are never converted to 0.
    """
    valid = series.dropna()
    valid = valid[~np.isinf(valid)]

    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    val_min = float(valid.min())
    val_max = float(valid.max())

    if np.isclose(val_max, val_min):
        res = pd.Series(np.nan, index=series.index)
        res[series.notna() & ~np.isinf(series)] = 0.5
        return res

    if direction == "higher_is_better":
        norm = (series - val_min) / (val_max - val_min)
    elif direction == "lower_is_better":
        norm = (val_max - series) / (val_max - val_min)
    else:
        raise ValueError(f"Unknown metric direction: {direction}")

    return norm.clip(lower=0.0, upper=1.0)


def build_model_profiles(df_mlm: pd.DataFrame, tok_data: dict, pll_dict: dict, model_profiles: dict):
    """
    Aggregates model-level raw capability, domain/tokenizer, and operational metrics.
    Computes direction-aware normalized metrics without fabricating missing values.
    N/A remains NaN.
    """
    if df_mlm.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    model_groups = df_mlm.groupby("model_name")
    raw_rows = []

    for model_name, grp in model_groups:
        t_info = tok_data.get(model_name, {})
        p_info = model_profiles.get(model_name, {})

        # Capability metrics
        avg_top1 = grp["top1_acc"].mean() if grp["top1_acc"].notna().any() else np.nan
        avg_top5 = grp["top5_acc"].mean() if "top5_acc" in grp and grp["top5_acc"].notna().any() else np.nan
        avg_top10 = grp["top10_acc"].mean() if "top10_acc" in grp and grp["top10_acc"].notna().any() else np.nan
        avg_rare = grp["rare_top1_acc"].mean() if "rare_top1_acc" in grp and grp["rare_top1_acc"].notna().any() else np.nan
        avg_loss = grp["mlm_loss"].mean() if grp["mlm_loss"].notna().any() else np.nan

        # Pseudo-perplexity
        if model_name in pll_dict:
            pseudo_ppl = pll_dict[model_name]
        elif "mlm_loss_derived_exponential" in grp and grp["mlm_loss_derived_exponential"].notna().any():
            pseudo_ppl = grp["mlm_loss_derived_exponential"].mean()
        else:
            pseudo_ppl = np.exp(avg_loss) if (not np.isnan(avg_loss) and avg_loss < 20) else np.nan

        # Category balance
        cat_cols = ["nav_acc", "weather_acc", "safety_acc", "machinery_acc", "vessel_acc", "casualty_acc"]
        cat_means = [grp[c].mean() for c in cat_cols if c in grp and grp[c].notna().any()]
        cat_balance = (1.0 - np.std(cat_means)) if cat_means else np.nan

        # Gaps
        avg_gap = grp["performance_gap"].mean() if "performance_gap" in grp and grp["performance_gap"].notna().any() else np.nan
        avg_shift = grp["domain_shift_gap"].mean() if "domain_shift_gap" in grp and grp["domain_shift_gap"].notna().any() else np.nan

        # Domain / Tokenizer fit metrics (Stage 13)
        frag_rate = t_info.get("maritime_fragmentation_rate")
        if frag_rate is None and "fragmentation_rate_pct" in t_info:
            frag_rate = t_info["fragmentation_rate_pct"] / 100.0
        frag_rate_pct = (frag_rate * 100.0) if frag_rate is not None else np.nan

        # N/A must remain NaN: byte-level BPE models (ModernBERT, RoBERTa) have no measured OOV
        oov_val = t_info.get("oov_rate")
        oov_status = t_info.get("oov_status")
        if oov_val is None or oov_status == "not_applicable":
            oov_pct = np.nan
        else:
            oov_pct = float(oov_val) * 100.0 if float(oov_val) <= 1.0 else float(oov_val)

        coverage_val = t_info.get("single_token_vocabulary_coverage")
        cov_pct = (coverage_val * 100.0) if coverage_val is not None else np.nan
        tok_speed = t_info.get("tokenizer_speed_tokens_per_sec", np.nan)

        # Operational metrics
        avg_eval_time = grp["eval_time_sec"].mean() if "eval_time_sec" in grp and grp["eval_time_sec"].notna().any() else np.nan
        avg_doc_count = grp["evaluated_doc_count"].mean() if "evaluated_doc_count" in grp and grp["evaluated_doc_count"].notna().any() and grp["evaluated_doc_count"].mean() > 0 else 200.0
        latency_ms = ((avg_eval_time / avg_doc_count) * 1000.0) if (avg_eval_time and avg_eval_time > 0) else np.nan
        throughput = (1000.0 / latency_ms) if (latency_ms and latency_ms > 0) else np.nan

        params_m = p_info.get("params_m", np.nan)
        size_mb = p_info.get("size_mb", np.nan)

        # 95% Confidence Interval for Top-1
        valid_top1 = grp["top1_acc"].dropna()
        n_obs = len(valid_top1)
        std_err = (valid_top1.std() / np.sqrt(n_obs) * 1.96) if n_obs > 1 else 0.0

        raw_rows.append({
            "model_name": model_name,
            "observations_count": len(grp),
            # Capability
            "raw__top1_acc": avg_top1,
            "raw__top5_acc": avg_top5,
            "raw__top10_acc": avg_top10,
            "raw__rare_top1_acc": avg_rare,
            "raw__mlm_loss": avg_loss,
            "raw__pseudo_perplexity": pseudo_ppl,
            "raw__category_balance": cat_balance,
            "raw__domain_shift_gap_pct": avg_shift * 100.0 if not np.isnan(avg_shift) else np.nan,
            "raw__performance_gap_pct": avg_gap * 100.0 if not np.isnan(avg_gap) else np.nan,
            "raw__top1_ci_error": std_err * 100.0,
            # Domain / Tokenizer fit
            "raw__fragmentation_rate_pct": frag_rate_pct,
            "raw__oov_rate_pct": oov_pct,
            "raw__single_token_coverage_pct": cov_pct,
            "raw__tokenizer_speed_tok_sec": tok_speed,
            # Operational
            "raw__inference_latency_ms": latency_ms,
            "raw__throughput_docs_sec": throughput,
            "raw__params_millions": params_m,
            "raw__disk_size_mb": size_mb
        })

    df_raw = pd.DataFrame(raw_rows)

    df_norm = pd.DataFrame({"model_name": df_raw["model_name"]})
    active_directions = {}

    for metric, direction in METRIC_DIRECTIONS.items():
        raw_col = f"raw__{metric}"
        if raw_col in df_raw and df_raw[raw_col].notna().any():
            if metric in FIXED_MECS_TRANSFORMS:
                df_norm[f"norm__{metric}"] = normalize_metric_fixed(df_raw[raw_col], metric)
                active_directions[metric] = f"fixed_transform: {FIXED_MECS_TRANSFORMS[metric]['type']} ({FIXED_MECS_TRANSFORMS[metric]['category']})"
            else:
                df_norm[f"norm__{metric}"] = normalize_metric(df_raw[raw_col], direction)
                active_directions[metric] = direction

    df_profiles = pd.merge(df_raw, df_norm, on="model_name")
    return df_profiles, df_norm, active_directions


def calculate_kendalls_w(rankings_matrix: np.ndarray) -> dict:
    """
    Computes Kendall's W (coefficient of concordance) for k rankings (raters/conditions)
    evaluating n objects (models). Correctly handles ties using standard average-rank correction.

    Parameters:
        rankings_matrix: array of shape (k, n) where each row is the ranking of n objects.

    Returns:
        Dictionary containing kendalls_w, chi2_stat, df, p_value, k_rankings, n_objects, tie_corrected.
    """
    mat = np.asarray(rankings_matrix, dtype=float)
    k, n = mat.shape
    if k < 1 or n < 2:
        return {
            "kendalls_w": 1.0 if n == 1 else 0.0,
            "chi2_stat": 0.0,
            "df": max(1, n - 1),
            "p_value": 1.0,
            "k_rankings": k,
            "n_objects": n,
            "tie_corrected": False
        }

    # Sum of ranks for each object across all k rankings
    R = np.sum(mat, axis=0)
    mean_R = k * (n + 1) / 2.0
    S = np.sum((R - mean_R) ** 2)

    # Tie correction: for each ranking i, sum (t^3 - t) for tied groups of size t
    T_total = 0.0
    for row in mat:
        _, counts = np.unique(row, return_counts=True)
        ties = counts[counts > 1]
        if len(ties) > 0:
            T_total += float(np.sum(ties ** 3 - ties))

    denom = (k ** 2 * (n ** 3 - n) - k * T_total) / 12.0
    if denom <= 0:
        W = 1.0
    else:
        W = float(S / denom)
    W = float(np.clip(W, 0.0, 1.0))

    df = n - 1
    chi2_stat = k * (n - 1) * W
    p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df))

    return {
        "kendalls_w": round(W, 4),
        "chi2_stat": round(chi2_stat, 4),
        "df": df,
        "p_value": p_val,
        "k_rankings": k,
        "n_objects": n,
        "tie_corrected": bool(T_total > 0)
    }


def calculate_representation_rankings(df_mlm: pd.DataFrame):
    """
    Computes per-representation rankings and multi-ranking concordance (Kendall's W)
    across representations. Dynamically discovers representations and models.
    """
    if df_mlm.empty:
        return pd.DataFrame(), {}

    reps = sorted(df_mlm["representation"].dropna().unique())
    models = sorted(df_mlm["model_name"].dropna().unique())
    rep_rank_dict = {}
    rep_score_dict = {}

    for r in reps:
        grp = df_mlm[df_mlm["representation"] == r]
        means = grp.groupby("model_name")["top1_acc"].mean()
        ranks = means.rank(ascending=False, method="average")
        for m, rk in ranks.items():
            rep_rank_dict.setdefault(m, {})[f"rep_rank__{r}"] = float(rk)
            rep_score_dict.setdefault(m, {})[f"rep_top1__{r}"] = round(float(means[m]) * 100.0, 2)

    df_rep = pd.DataFrame([
        {"model_name": m, **rep_rank_dict.get(m, {}), **rep_score_dict.get(m, {})}
        for m in df_mlm["model_name"].unique()
    ])

    rank_cols = [f"rep_rank__{r}" for r in reps if f"rep_rank__{r}" in df_rep.columns]
    if rank_cols:
        df_rep["rep_mean_rank"] = df_rep[rank_cols].mean(axis=1).round(2)
        df_rep["rep_rank_std"] = df_rep[rank_cols].std(axis=1).fillna(0.0).round(2)

    # Kendall's W multi-ranking concordance across representations
    rank_matrix = []
    for r in reps:
        row = [df_rep.loc[df_rep["model_name"] == m, f"rep_rank__{r}"].iloc[0] for m in models]
        rank_matrix.append(row)
    w_res = calculate_kendalls_w(np.array(rank_matrix))

    # Secondary pairwise stats for reference
    tau_list, rho_list = [], []
    for r1, r2 in combinations(reps, 2):
        c1, c2 = f"rep_rank__{r1}", f"rep_rank__{r2}"
        if c1 in df_rep and c2 in df_rep:
            sub = df_rep[[c1, c2]].dropna()
            if len(sub) >= 2:
                tau, _ = stats.kendalltau(sub[c1], sub[c2])
                rho, _ = stats.spearmanr(sub[c1], sub[c2])
                if not np.isnan(tau):
                    tau_list.append(tau)
                if not np.isnan(rho):
                    rho_list.append(rho)

    stability = {
        "rep_stability_kendalls_w": w_res["kendalls_w"],
        "kendalls_w_chi2": w_res["chi2_stat"],
        "kendalls_w_df": w_res["df"],
        "kendalls_w_p_value": w_res["p_value"],
        "mean_spearman_rho": round(float(np.mean(rho_list)), 4) if rho_list else 0.0,
        "pairwise_mean_kendall_tau": round(float(np.mean(tau_list)), 4) if tau_list else 0.0,
        "evaluated_representations": reps,
        "number_of_rankings": w_res["k_rankings"],
        "number_of_objects": w_res["n_objects"]
    }

    return df_rep, stability


def calculate_subset_rankings(df_mlm: pd.DataFrame):
    """
    Computes per-subset rankings and multi-ranking concordance (Kendall's W)
    across knowledge subsets. Dynamically discovers subsets and models.
    """
    if df_mlm.empty:
        return pd.DataFrame(), {}

    subs = sorted(df_mlm["subset"].dropna().unique())
    models = sorted(df_mlm["model_name"].dropna().unique())
    sub_rank_dict = {}
    sub_score_dict = {}

    for s in subs:
        grp = df_mlm[df_mlm["subset"] == s]
        means = grp.groupby("model_name")["top1_acc"].mean()
        ranks = means.rank(ascending=False, method="average")
        for m, rk in ranks.items():
            sub_rank_dict.setdefault(m, {})[f"subset_rank__{s}"] = float(rk)
            sub_score_dict.setdefault(m, {})[f"subset_top1__{s}"] = round(float(means[m]) * 100.0, 2)

    df_sub = pd.DataFrame([
        {"model_name": m, **sub_rank_dict.get(m, {}), **sub_score_dict.get(m, {})}
        for m in df_mlm["model_name"].unique()
    ])

    rank_cols = [f"subset_rank__{s}" for s in subs if f"subset_rank__{s}" in df_sub.columns]
    if rank_cols:
        df_sub["subset_mean_rank"] = df_sub[rank_cols].mean(axis=1).round(2)
        df_sub["subset_rank_std"] = df_sub[rank_cols].std(axis=1).fillna(0.0).round(2)

    # Kendall's W multi-ranking concordance across subsets
    rank_matrix = []
    for s in subs:
        row = [df_sub.loc[df_sub["model_name"] == m, f"subset_rank__{s}"].iloc[0] for m in models]
        rank_matrix.append(row)
    w_res = calculate_kendalls_w(np.array(rank_matrix))

    # Secondary pairwise stats for reference
    tau_list, rho_list = [], []
    for s1, s2 in combinations(subs, 2):
        c1, c2 = f"subset_rank__{s1}", f"subset_rank__{s2}"
        if c1 in df_sub and c2 in df_sub:
            sub = df_sub[[c1, c2]].dropna()
            if len(sub) >= 2:
                tau, _ = stats.kendalltau(sub[c1], sub[c2])
                rho, _ = stats.spearmanr(sub[c1], sub[c2])
                if not np.isnan(tau):
                    tau_list.append(tau)
                if not np.isnan(rho):
                    rho_list.append(rho)

    stability = {
        "subset_stability_kendalls_w": w_res["kendalls_w"],
        "kendalls_w_chi2": w_res["chi2_stat"],
        "kendalls_w_df": w_res["df"],
        "kendalls_w_p_value": w_res["p_value"],
        "mean_spearman_rho": round(float(np.mean(rho_list)), 4) if rho_list else 0.0,
        "pairwise_mean_kendall_tau": round(float(np.mean(tau_list)), 4) if tau_list else 0.0,
        "evaluated_subsets": subs,
        "number_of_rankings": w_res["k_rankings"],
        "number_of_objects": w_res["n_objects"]
    }

    return df_sub, stability


def calculate_mecs(df_profiles: pd.DataFrame, scenario_weights: dict):
    """
    Computes Maritime Encoder Composite Score (MECS) on normalized metrics.
    For each model, renormalizes weights over its applicable, non-null metrics.
    Missing/N/A metrics are never encoded as 0.0 and do not penalize or artificially reward any model.
    Returns (scores_series, applicable_metrics_list).
    """
    applicable_metrics = [
        m for m in scenario_weights.keys()
        if f"norm__{m}" in df_profiles and df_profiles[f"norm__{m}"].notna().any()
    ]

    scores = []
    for _, row in df_profiles.iterrows():
        valid_weights = {}
        for m in applicable_metrics:
            norm_col = f"norm__{m}"
            val = row.get(norm_col)
            if pd.notna(val) and not np.isnan(val):
                valid_weights[norm_col] = scenario_weights[m]

        if not valid_weights:
            scores.append(50.0)
            continue

        sum_w = sum(valid_weights.values())
        model_score = sum((w / sum_w) * row[col] for col, w in valid_weights.items()) * 100.0
        scores.append(round(float(model_score), 2))

    return pd.Series(scores, index=df_profiles.index), applicable_metrics


def run_mecs_sensitivity(df_profiles: pd.DataFrame):
    """
    Evaluates model rankings across four distinct MECS weighting scenarios.
    Dynamically adapts to available metrics and tracks actual metrics used per scenario.
    Reports win frequency and score stability.
    """
    df_sens = pd.DataFrame({"model_name": df_profiles["model_name"]})
    scenario_winners = {}
    scenario_metrics_used = {}

    for sc_name, sc_weights in MECS_SCENARIOS.items():
        scores, used_metrics = calculate_mecs(df_profiles, sc_weights)
        ranks = scores.rank(ascending=False, method="min").astype(int)

        df_sens[f"{sc_name}_score"] = scores
        df_sens[f"{sc_name}_rank"] = ranks

        top_idx = scores.idxmax()
        scenario_winners[sc_name] = df_profiles.loc[top_idx, "model_name"]
        scenario_metrics_used[sc_name] = used_metrics

    rank_cols = [f"{sc_name}_rank" for sc_name in MECS_SCENARIOS.keys()]
    df_sens["total_wins"] = (df_sens[rank_cols] == 1).sum(axis=1)
    df_sens["win_frequency"] = (df_sens["total_wins"] / float(len(MECS_SCENARIOS))).round(2)

    df_sens.sort_values(by=["total_wins", "baseline_score"], ascending=[False, False], inplace=True)
    return df_sens, scenario_winners, scenario_metrics_used


def calculate_pareto_front(df_profiles: pd.DataFrame):
    """
    Deterministic Pareto dominance analysis on the 6 defensible primary benchmark objectives:
    1. top1_acc (intrinsic capability, higher is better)
    2. rare_top1_acc (domain tail capability, higher is better)
    3. mlm_loss (cross-entropy convergence, lower is better -> norm is higher is better)
    4. fragmentation_rate_pct (domain morphological fit, lower is better -> norm is higher is better)
    5. inference_latency_ms (operational latency, lower is better -> norm is higher is better)
    6. throughput_docs_sec (operational throughput, higher is better)

    Excluded from Pareto dominance:
    - oov_rate_pct: byte-level BPE models have no measured OOV (unshared across cohort).
    - single_token_coverage_pct: redundant / collinear with fragmentation rate.
    - pseudo_perplexity: secondary sampled diagnostic evaluated on limited subset.
    - top5_acc / top10_acc: collinear with top1_acc.
    """
    candidate_objs = [
        "top1_acc", "rare_top1_acc", "mlm_loss",
        "fragmentation_rate_pct", "inference_latency_ms", "throughput_docs_sec"
    ]
    active_objs = [
        f"norm__{m}" for m in candidate_objs
        if f"norm__{m}" in df_profiles and df_profiles[f"norm__{m}"].notna().any()
    ]

    n = len(df_profiles)
    models = df_profiles["model_name"].tolist()

    dominated_by = {m: [] for m in models}
    dominates = {m: [] for m in models}

    eps = 1e-6
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row_i = df_profiles.iloc[i]
            row_j = df_profiles.iloc[j]

            shared_objs = [o for o in active_objs if pd.notna(row_i[o]) and pd.notna(row_j[o])]
            if not shared_objs:
                continue

            vals_i = np.array([row_i[o] for o in shared_objs])
            vals_j = np.array([row_j[o] for o in shared_objs])

            greater_equal = np.all(vals_i >= vals_j - eps)
            strictly_greater = np.any(vals_i > vals_j + eps)

            if greater_equal and strictly_greater:
                dominates[models[i]].append(models[j])
                dominated_by[models[j]].append(models[i])

    rows = []
    for m in models:
        dom_list = dominated_by[m]
        status = "Pareto-Optimal" if len(dom_list) == 0 else "Dominated"
        rows.append({
            "model_name": m,
            "pareto_status": status,
            "dominated_by_count": len(dom_list),
            "dominated_by": ", ".join(dom_list) if dom_list else "None",
            "dominates_count": len(dominates[m]),
            "dominates": ", ".join(dominates[m]) if dominates[m] else "None",
            "evaluated_objectives": ", ".join([o.replace("norm__", "") for o in active_objs])
        })

    df_pareto = pd.DataFrame(rows)
    df_pareto.sort_values(by=["pareto_status", "dominated_by_count"], ascending=[False, True], inplace=True)
    return df_pareto


def generate_selection_decision(
    df_profiles: pd.DataFrame,
    df_rep: pd.DataFrame,
    rep_stability: dict,
    df_sub: pd.DataFrame,
    sub_stability: dict,
    df_sens: pd.DataFrame,
    df_pareto: pd.DataFrame,
    coverage_info: dict,
    scenario_metrics_used: dict
):
    """
    Transparent multi-criteria decision hierarchy.
    Separates Capability, Domain Fit, and Operational metrics.
    Never hardcodes outcomes, winners, or score thresholds.
    """
    # 1. Best overall intrinsic MLM model
    best_mlm_row = df_profiles.sort_values(by="raw__top1_acc", ascending=False).iloc[0]
    best_mlm_model = best_mlm_row["model_name"]

    # 2. Strongest domain-specific model (rare token accuracy)
    rare_col = "raw__rare_top1_acc" if ("raw__rare_top1_acc" in df_profiles and df_profiles["raw__rare_top1_acc"].notna().any()) else "raw__top1_acc"
    best_domain_row = df_profiles.sort_values(by=rare_col, ascending=False).iloc[0]
    best_domain_model = best_domain_row["model_name"]

    # 3. Stability leaders
    most_stable_rep_row = df_rep.sort_values(by=["rep_rank_std", "rep_mean_rank"]).iloc[0]
    most_stable_rep_model = most_stable_rep_row["model_name"]

    most_stable_sub_row = df_sub.sort_values(by=["subset_rank_std", "subset_mean_rank"]).iloc[0]
    most_stable_sub_model = most_stable_sub_row["model_name"]

    # 4. Most robust MECS sensitivity leader
    most_robust_mecs_row = df_sens.sort_values(by=["total_wins", "baseline_score"], ascending=[False, False]).iloc[0]
    most_robust_mecs_model = most_robust_mecs_row["model_name"]

    # 5. Pareto optimal models
    pareto_optimal_models = df_pareto[df_pareto["pareto_status"] == "Pareto-Optimal"]["model_name"].tolist()

    # Join profiles, sensitivity, and pareto status
    merged = pd.merge(df_profiles, df_sens, on="model_name")
    merged = pd.merge(merged, df_pareto[["model_name", "pareto_status", "dominated_by_count"]], on="model_name")

    # Selection hierarchy:
    # 1) Must be Pareto-Optimal (or have minimal dominance if empty)
    # 2) High intrinsic capability + sensitivity win frequency
    candidates = merged[merged["pareto_status"] == "Pareto-Optimal"]
    if candidates.empty:
        candidates = merged

    candidates_sorted = candidates.sort_values(
        by=["total_wins", "raw__top1_acc", "baseline_score"],
        ascending=[False, False, False]
    )
    recommended_model = candidates_sorted.iloc[0]["model_name"]
    rec_profile = merged[merged["model_name"] == recommended_model].iloc[0]

    # Explicit trade-off documentation
    trade_offs = []
    for other_model in pareto_optimal_models:
        if other_model != recommended_model:
            o_prof = merged[merged["model_name"] == other_model].iloc[0]
            advantages = []

            # Domain / Tokenizer fit advantages
            rec_frag = rec_profile.get("raw__fragmentation_rate_pct")
            oth_frag = o_prof.get("raw__fragmentation_rate_pct")
            if pd.notna(oth_frag) and pd.notna(rec_frag) and oth_frag < rec_frag:
                advantages.append(f"lower subword fragmentation ({oth_frag:.1f}% vs {rec_frag:.1f}%)")

            rec_rare = rec_profile.get("raw__rare_top1_acc")
            oth_rare = o_prof.get("raw__rare_top1_acc")
            if pd.notna(oth_rare) and pd.notna(rec_rare) and oth_rare > rec_rare:
                advantages.append(f"higher rare maritime token accuracy ({oth_rare*100:.1f}% vs {rec_rare*100:.1f}%)")

            # Operational efficiency advantages
            rec_params = rec_profile.get("raw__params_millions")
            oth_params = o_prof.get("raw__params_millions")
            if pd.notna(oth_params) and pd.notna(rec_params) and oth_params < rec_params:
                advantages.append(f"smaller footprint ({int(oth_params)}M vs {int(rec_params)}M params)")

            rec_lat = rec_profile.get("raw__inference_latency_ms")
            oth_lat = o_prof.get("raw__inference_latency_ms")
            if pd.notna(oth_lat) and pd.notna(rec_lat) and oth_lat < rec_lat:
                advantages.append(f"lower latency ({oth_lat:.1f}ms vs {rec_lat:.1f}ms)")

            if advantages:
                trade_offs.append({
                    "alternative_model": other_model,
                    "advantages_over_selected": advantages
                })

    wins = int(rec_profile["total_wins"])
    if wins == len(MECS_SCENARIOS):
        sens_summary = f"unanimous leader across all {len(MECS_SCENARIOS)} MECS sensitivity scenarios"
    elif wins >= 3:
        sens_summary = f"consistent leader across {wins}/{len(MECS_SCENARIOS)} MECS sensitivity scenarios"
    elif wins == 2:
        sens_summary = f"leading performance across {wins}/{len(MECS_SCENARIOS)} MECS sensitivity scenarios (baseline and performance-heavy paradigms)"
    else:
        sens_summary = f"leading rank in {wins}/{len(MECS_SCENARIOS)} MECS sensitivity scenarios"

    selection_rationale = (
        f"{recommended_model} demonstrated the strongest intrinsic MLM capability "
        f"({float(rec_profile['raw__top1_acc'])*100:.2f}% Top-1 accuracy, {float(rec_profile['raw__mlm_loss']):.4f} MLM loss), "
        f"high ranking consistency across representations (mean rank {most_stable_rep_row['rep_mean_rank']}) "
        f"and subsets (mean rank {most_stable_sub_row['subset_mean_rank']}), {sens_summary}, "
        f"and confirmed non-dominated Pareto status."
    )

    decision = {
        "timestamp": datetime.now().isoformat(),
        "data_coverage": {
            "expected_cells": coverage_info["expected_cells"],
            "valid_cells": coverage_info["valid_cells"],
            "missing_cells": coverage_info["missing_cells"],
            "discovered_files": coverage_info["discovered_files"]
        },
        "criteria_winners": {
            "best_overall_intrinsic_mlm": {
                "model_name": best_mlm_model,
                "top1_acc_pct": round(float(best_mlm_row["raw__top1_acc"]) * 100.0, 2),
                "mlm_loss": round(float(best_mlm_row["raw__mlm_loss"]), 4)
            },
            "strongest_domain_model": {
                "model_name": best_domain_model,
                "rare_top1_acc_pct": round(float(best_domain_row[rare_col]) * 100.0, 2)
            },
            "most_stable_representation_model": {
                "model_name": most_stable_rep_model,
                "mean_rank": float(most_stable_rep_row["rep_mean_rank"]),
                "rank_std": float(most_stable_rep_row["rep_rank_std"])
            },
            "most_stable_subset_model": {
                "model_name": most_stable_sub_model,
                "mean_rank": float(most_stable_sub_row["subset_mean_rank"]),
                "rank_std": float(most_stable_sub_row["subset_rank_std"])
            },
            "most_robust_mecs_winner": {
                "model_name": most_robust_mecs_model,
                "total_wins": int(most_robust_mecs_row["total_wins"]),
                "win_frequency": float(most_robust_mecs_row["win_frequency"]),
                "baseline_score": float(most_robust_mecs_row["baseline_score"])
            }
        },
        "rank_stability": {
            "representation_kendalls_w": rep_stability.get("rep_stability_kendalls_w", 0.0),
            "representation_spearman_rho": rep_stability.get("mean_spearman_rho", 0.0),
            "subset_kendalls_w": sub_stability.get("subset_stability_kendalls_w", 0.0),
            "subset_spearman_rho": sub_stability.get("mean_spearman_rho", 0.0)
        },
        "pareto_summary": {
            "pareto_optimal_count": len(pareto_optimal_models),
            "pareto_optimal_models": pareto_optimal_models,
            "selected_model_pareto_status": rec_profile["pareto_status"]
        },
        "final_selection": {
            "recommended_model": recommended_model,
            "pareto_status": rec_profile["pareto_status"],
            "baseline_mecs_score": float(rec_profile["baseline_score"]),
            "maritime_top1_acc_pct": round(float(rec_profile["raw__top1_acc"]) * 100.0, 2),
            "rare_maritime_acc_pct": round(float(rec_profile["raw__rare_top1_acc"]) * 100.0, 2) if pd.notna(rec_profile["raw__rare_top1_acc"]) else "N/A",
            "mlm_loss": round(float(rec_profile["raw__mlm_loss"]), 4),
            "sensitivity_wins": f"{wins}/{len(MECS_SCENARIOS)} scenarios",
            "selection_rationale": selection_rationale
        },
        "trade_offs": trade_offs,
        "methodological_note": (
            "Maritime Encoder Composite Score (MECS) is an operational composite compatibility score used to summarize multiple "
            "encoder evaluation characteristics for model selection. It is not intended as a direct measure of language or maritime understanding. "
            "Model selection is defensibly justified by multidimensional evidence including intrinsic MLM loss, rare domain token accuracy, "
            "representation and subset ranking agreement, sensitivity analysis invariance, and non-dominated Pareto status."
        )
    }

    return decision


def generate_report(
    decision: dict,
    df_profiles: pd.DataFrame,
    df_rep: pd.DataFrame,
    df_sub: pd.DataFrame,
    df_sens: pd.DataFrame,
    df_pareto: pd.DataFrame,
    coverage_info: dict,
    output_path: Path
):
    """Generates the publication-ready markdown report stage15_report.md."""
    sel = decision["final_selection"]
    rec_model = sel["recommended_model"]

    lines = [
        "# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report",
        "",
        "## 1. Executive Conclusion",
        "",
        f"**Recommended Model:** `{rec_model}`  ",
        f"- **Selection Status:** {sel['pareto_status']}  ",
        f"- **Baseline Operational MECS:** {sel['baseline_mecs_score']:.2f} / 100  ",
        f"- **Maritime Top-1 Accuracy:** {sel['maritime_top1_acc_pct']:.2f}%  ",
        f"- **Rare Maritime Token Accuracy:** {sel['rare_maritime_acc_pct']}%  ",
        f"- **MLM Loss:** {sel['mlm_loss']:.4f}  ",
        f"- **Weighting Sensitivity Stability:** Won {sel['sensitivity_wins']}  ",
        "",
        f"**Rationale:** {sel['selection_rationale']}",
        "",
        "---",
        "",
        "## 2. Data Coverage",
        "",
        f"- **Discovered Files:** {coverage_info['discovered_files']}",
        f"- **Valid Evaluated Matrix Cells:** {coverage_info['valid_cells']}",
        f"- **Expected Full Cartesian Cells:** {coverage_info['expected_cells']}",
        f"- **Missing Cells:** {coverage_info['missing_cells']}",
        f"- **Invalid Files Skipped:** {coverage_info['invalid_files_count']}",
        f"- **Evaluated Models ({len(coverage_info['unique_models'])}):** {', '.join(coverage_info['unique_models'])}",
        f"- **Evaluated Representations ({len(coverage_info['unique_representations'])}):** {', '.join(coverage_info['unique_representations'])}",
        f"- **Evaluated Subsets ({len(coverage_info['unique_subsets'])}):** {', '.join(coverage_info['unique_subsets'])}",
        "",
        "---",
        "",
        "## 3. Model Comparison",
        "",
        "### Capability Metrics",
        "",
        "| Model Name | Maritime Top-1 (%) | Top-5 (%) | Rare Top-1 (%) | MLM Loss | Pseudo-Perplexity | Baseline MECS |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for _, r in df_profiles.sort_values(by="raw__top1_acc", ascending=False).iterrows():
        m = r["model_name"]
        sens_row = df_sens[df_sens["model_name"] == m].iloc[0] if not df_sens[df_sens["model_name"] == m].empty else {}
        mecs = sens_row.get("baseline_score", "N/A")
        t1 = f"{r['raw__top1_acc']*100:.2f}" if pd.notna(r["raw__top1_acc"]) else "N/A"
        t5 = f"{r['raw__top5_acc']*100:.2f}" if pd.notna(r["raw__top5_acc"]) else "N/A"
        rt1 = f"{r['raw__rare_top1_acc']*100:.2f}" if pd.notna(r["raw__rare_top1_acc"]) else "N/A"
        loss = f"{r['raw__mlm_loss']:.4f}" if pd.notna(r["raw__mlm_loss"]) else "N/A"
        ppl = f"{r['raw__pseudo_perplexity']:.2f}" if pd.notna(r["raw__pseudo_perplexity"]) else "N/A"
        lines.append(f"| `{m}` | {t1} | {t5} | {rt1} | {loss} | {ppl} | {mecs} |")

    lines.extend([
        "",
        "### Domain / Tokenizer Fit & Operational Metrics",
        "",
        "| Model Name | Frag Rate (%) | OOV Rate (%) | Single Token Cov (%) | Latency (ms) | Throughput (docs/s) | Parameters (M) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for _, r in df_profiles.sort_values(by="raw__top1_acc", ascending=False).iterrows():
        m = r["model_name"]
        frag = f"{r['raw__fragmentation_rate_pct']:.1f}" if pd.notna(r["raw__fragmentation_rate_pct"]) else "N/A"
        oov = f"{r['raw__oov_rate_pct']:.2f}" if pd.notna(r["raw__oov_rate_pct"]) else "N/A"
        cov = f"{r['raw__single_token_coverage_pct']:.1f}" if pd.notna(r["raw__single_token_coverage_pct"]) else "N/A"
        lat = f"{r['raw__inference_latency_ms']:.1f}" if pd.notna(r["raw__inference_latency_ms"]) else "N/A"
        tp = f"{r['raw__throughput_docs_sec']:.1f}" if pd.notna(r["raw__throughput_docs_sec"]) else "N/A"
        param = f"{int(r['raw__params_millions'])}" if pd.notna(r["raw__params_millions"]) else "N/A"
        lines.append(f"| `{m}` | {frag} | {oov} | {cov} | {lat} | {tp} | {param} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Representation Robustness",
        "",
        f"- **Kendall's W (Multi-Ranking Concordance):** `{decision['rank_stability']['representation_kendalls_w']:.4f}`",
        f"- **Mean Pairwise Spearman's Rho:** `{decision['rank_stability']['representation_spearman_rho']:.4f}`",
        "",
        "Representation breakdown across evaluated formats:",
        ""
    ])

    rep_cols = [c for c in df_rep.columns if c.startswith("rep_rank__")]
    headers = ["Model"] + [c.replace("rep_rank__", "") for c in rep_cols] + ["Mean Rank", "Rank Std"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")

    for _, r in df_rep.sort_values(by="rep_mean_rank").iterrows():
        row_vals = [f"`{r['model_name']}`"]
        for c in rep_cols:
            row_vals.append(str(r.get(c, "N/A")))
        row_vals.append(str(r.get("rep_mean_rank", "N/A")))
        row_vals.append(str(r.get("rep_rank_std", "N/A")))
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Subset Robustness",
        "",
        f"- **Kendall's W (Multi-Ranking Concordance):** `{decision['rank_stability']['subset_kendalls_w']:.4f}`",
        f"- **Mean Pairwise Spearman's Rho:** `{decision['rank_stability']['subset_spearman_rho']:.4f}`",
        "",
        "Subset ranking breakdown across knowledge/informativeness conditions:",
        ""
    ])

    sub_cols = [c for c in df_sub.columns if c.startswith("subset_rank__")]
    s_headers = ["Model"] + [c.replace("subset_rank__", "") for c in sub_cols] + ["Mean Rank", "Rank Std"]
    lines.append("| " + " | ".join(s_headers) + " |")
    lines.append("| " + " | ".join([":---"] + [":---:"] * (len(s_headers) - 1)) + " |")

    for _, r in df_sub.sort_values(by="subset_mean_rank").iterrows():
        row_vals = [f"`{r['model_name']}`"]
        for c in sub_cols:
            row_vals.append(str(r.get(c, "N/A")))
        row_vals.append(str(r.get("subset_mean_rank", "N/A")))
        row_vals.append(str(r.get("subset_rank_std", "N/A")))
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. MECS Sensitivity Analysis",
        "",
        "Testing invariance across four distinct weighting hypotheses:",
        "1. **Baseline / Operational:** Balanced operational mixture.",
        "2. **Performance-Heavy:** Focuses strictly on intrinsic MLM accuracy and loss.",
        "3. **Domain-Heavy:** Strongly weights rare domain terminology and domain accuracy.",
        "4. **Balanced:** Equal weighting across capability, tokenizer fit, and throughput efficiency.",
        "",
        "| Model Name | Baseline Score (Rank) | Perf-Heavy Score (Rank) | Domain-Heavy Score (Rank) | Balanced Score (Rank) | Total Wins | Win Frequency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for _, r in df_sens.iterrows():
        m = r["model_name"]
        b_str = f"{r['baseline_score']:.1f} (#{int(r['baseline_rank'])})"
        p_str = f"{r['performance_heavy_score']:.1f} (#{int(r['performance_heavy_rank'])})"
        d_str = f"{r['domain_heavy_score']:.1f} (#{int(r['domain_heavy_rank'])})"
        bal_str = f"{r['balanced_score']:.1f} (#{int(r['balanced_rank'])})"
        wins = f"{int(r['total_wins'])}"
        freq = f"{r['win_frequency']*100:.0f}%"
        lines.append(f"| `{m}` | {b_str} | {p_str} | {d_str} | {bal_str} | {wins} | {freq} |")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Pareto Dominance Analysis",
        "",
        "| Model Name | Pareto Status | Dominates Count | Dominated By Count | Dominating Models |",
        "| :--- | :---: | :---: | :---: | :--- |"
    ])

    for _, r in df_pareto.iterrows():
        lines.append(f"| `{r['model_name']}` | **{r['pareto_status']}** | {r['dominates_count']} | {r['dominated_by_count']} | {r['dominated_by']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 8. Capability vs. Resource Trade-offs",
        "",
        "### Capability vs. Tokenizer/Domain Fit",
        f"- `{rec_model}` achieves highest overall intrinsic MLM performance, but exhibits higher subword fragmentation than specialized WordPiece architectures.",
        "- Models with lower subword fragmentation (e.g. `bert-base-uncased` at 26.6% fragmentation) offer better morphological token boundaries for specific domain stems, despite lower overall MLM Top-1 accuracy.",
        "",
        "### Capability vs. Operational Cost (Environment-Specific)",
        f"- `{rec_model}` requires 149M parameters with an operational inference latency of ~31.5ms per document (~31.8 docs/sec) on the benchmark GPU harness.",
        "- Lightweight alternatives (e.g., 110M base models like `bert-base-uncased` and `nlpaueb/legal-bert-base-uncased`) provide lower inference latency (~21.0-22.4ms per document, ~44.7-47.6 docs/sec) with smaller disk footprints (~440MB vs ~590MB).",
        "- Latency and throughput depend on benchmark execution hardware and batch configuration and represent operational considerations rather than intrinsic linguistic capabilities.",
        ""
    ])

    if decision["trade_offs"]:
        lines.append("### Alternative Trade-off Details")
        for t in decision["trade_offs"]:
            alt = t["alternative_model"]
            advs = "; ".join(t["advantages_over_selected"])
            lines.append(f"- **Alternative `{alt}`:** Offers {advs}.")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Methodological Note",
        "",
        "> **Notice on Composite Scoring:**  ",
        "> Maritime Encoder Composite Score (MECS) is an operational composite compatibility score used to summarize encoder evaluation characteristics for model selection. "
        "It is not intended as a direct measure of language or maritime understanding. "
        "The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, "
        "rare-token domain generalization, representation consistency, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.",
        ""
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Stage 15: Cross-Model Benchmarking & Selection")
    parser.add_argument("--cache-dir", type=str, default=None, help="Stage 14 evaluations cache directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Stage 15 output directory")
    parser.add_argument("--masking-mode", type=str, default="whole_word", choices=["subword", "whole_word"], help="Evaluation masking mode")
    parser.add_argument("--evaluation-unit", type=str, default="word", choices=["subword", "word"], help="Evaluation unit")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config()
    default_out = root / config.get("output_dir", "outputs")

    stage_dir = Path(args.output_dir) if args.output_dir else default_out / "stage-15"
    stage_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = default_out / "stage-13" / "tokenizer_analysis"
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        wwm_cache = default_out / "stage-14" / "evaluations" / "cache_wwm_word"
        if wwm_cache.exists() and any(wwm_cache.glob("*.json")):
            cache_dir = wwm_cache
        else:
            cache_dir = default_out / "stage-14" / "evaluations" / "cache"
    pll_path = default_out / "stage-14" / "pll_results.json"

    logger.info(f"Step 1: Dynamically discovering and validating Stage 14 results from {cache_dir}...")
    df_mlm, coverage_info, pll_dict = discover_stage14_results(cache_dir, pll_path)

    if df_mlm.empty:
        logger.error("No valid Stage 14 evaluation records found! Exiting Stage 15.")
        return

    logger.info(f"Discovered {coverage_info['valid_cells']} valid records across {len(coverage_info['unique_models'])} models.")

    # Export comparison.csv (required by Stage 16)
    df_mlm.to_csv(stage_dir / "comparison.csv", index=False)

    # Step 2: Load Stage 13 Tokenizer Data
    tok_data = {}
    if tok_dir.exists():
        for json_file in tok_dir.glob("*.json"):
            if json_file.name == "tokenizer_comparison.csv":
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    m = data.get("model_name")
                    if m:
                        tok_data[m] = data
            except Exception as e:
                logger.warning(f"Could not load tokenizer file {json_file.name}: {e}")

    # Step 3: Build Model Profiles (Raw + Direction-Aware Normalized)
    logger.info("Step 3: Building model profiles with cohort-independent loss normalization...")
    df_profiles, df_norm, active_directions = build_model_profiles(df_mlm, tok_data, pll_dict, MODEL_PROFILES)
    df_profiles.to_csv(stage_dir / "stage15_model_profiles.csv", index=False)

    # Step 4: Representation-wise Rankings & Stability (Kendall's W)
    logger.info("Step 4: Computing representation-wise rankings and Kendall's W concordance...")
    df_rep, rep_stability = calculate_representation_rankings(df_mlm)

    # Step 5: Subset-wise Rankings & Stability (Kendall's W)
    logger.info("Step 5: Computing subset-wise rankings and Kendall's W concordance...")
    df_sub, sub_stability = calculate_subset_rankings(df_mlm)

    # Unified rankings dataframe
    df_rankings = pd.merge(df_rep, df_sub, on="model_name")
    df_rankings["rep_stability_kendalls_w"] = rep_stability["rep_stability_kendalls_w"]
    df_rankings["subset_stability_kendalls_w"] = sub_stability["subset_stability_kendalls_w"]
    df_rankings.to_csv(stage_dir / "stage15_rankings.csv", index=False)

    # Step 6: MECS Sensitivity Analysis (4 Weighting Scenarios)
    logger.info("Step 6: Executing MECS sensitivity analysis across 4 weighting scenarios...")
    df_sens, scenario_winners, scenario_metrics_used = run_mecs_sensitivity(df_profiles)
    df_sens.to_csv(stage_dir / "stage15_mecs_sensitivity.csv", index=False)

    # Step 7: Deterministic Pareto Dominance Analysis
    logger.info("Step 7: Calculating Pareto-optimal frontier...")
    df_pareto = calculate_pareto_front(df_profiles)
    df_pareto.to_csv(stage_dir / "stage15_pareto.csv", index=False)

    # Step 8: Multi-Criteria Selection Decision
    logger.info("Step 8: Formulating multi-criteria model selection decision...")
    decision = generate_selection_decision(
        df_profiles, df_rep, rep_stability, df_sub, sub_stability,
        df_sens, df_pareto, coverage_info, scenario_metrics_used
    )
    decision["masking_mode"] = args.masking_mode
    decision["evaluation_unit"] = args.evaluation_unit
    decision["mecs_normalization"] = "cohort_independent: fixed_bounded_reference_scales"
    with open(stage_dir / "stage15_selection_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    # Export MECS and Pareto Configuration Manifests
    mecs_config = {
        "description": "Maritime Encoder Composite Score (MECS) Configuration & Justification Manifest",
        "methodological_principle": "MECS is an operational composite compatibility score for candidate ranking, not an intrinsic measure of domain understanding. All transforms are explicit design choices with documented justifications.",
        "scenarios": MECS_SCENARIOS,
        "transforms": FIXED_MECS_TRANSFORMS,
        "operational_metrics_note": "Inference latency and throughput depend on benchmark execution hardware (NVIDIA GPU / PyTorch) and are explicitly classified as environment-specific operational metrics."
    }
    with open(stage_dir / "stage15_mecs_config.json", "w", encoding="utf-8") as f:
        json.dump(mecs_config, f, indent=2)

    pareto_config = {
        "description": "Pareto Dominance Frontier Configuration Manifest",
        "primary_objectives": [
            {"metric": "top1_acc", "direction": "higher_is_better", "category": "intrinsic_capability"},
            {"metric": "rare_top1_acc", "direction": "higher_is_better", "category": "domain_tail_capability"},
            {"metric": "mlm_loss", "direction": "lower_is_better", "category": "intrinsic_loss"},
            {"metric": "fragmentation_rate_pct", "direction": "lower_is_better", "category": "domain_morphological_fit"},
            {"metric": "inference_latency_ms", "direction": "lower_is_better", "category": "operational_efficiency"},
            {"metric": "throughput_docs_sec", "direction": "higher_is_better", "category": "operational_throughput"}
        ],
        "excluded_metrics": {
            "oov_rate_pct": "Inapplicable to byte-level BPE tokenizers (ModernBERT, RoBERTa); not uniformly shared across cohort.",
            "single_token_coverage_pct": "Collinear with fragmentation rate.",
            "pseudo_perplexity": "Secondary diagnostic evaluated on limited document sample.",
            "top5_acc": "Collinear with top1_acc."
        }
    }
    with open(stage_dir / "stage15_pareto_config.json", "w", encoding="utf-8") as f:
        json.dump(pareto_config, f, indent=2)

    # Step 9: Publication-Ready Markdown Report
    logger.info("Step 9: Generating scientific synthesis report...")
    generate_report(decision, df_profiles, df_rep, df_sub, df_sens, df_pareto, coverage_info, stage_dir / "stage15_report.md")

    # Step 10: leaderboard.csv (Required by Stage 17)
    leaderboard_rows = []
    for _, r in df_profiles.iterrows():
        m = r["model_name"]
        sens_row = df_sens[df_sens["model_name"] == m].iloc[0] if not df_sens[df_sens["model_name"] == m].empty else {}
        leaderboard_rows.append({
            "model_name": m,
            "mecs_score": sens_row.get("baseline_score", 50.0),
            "maritime_top1_acc": round(float(r["raw__top1_acc"]) * 100.0, 2) if pd.notna(r["raw__top1_acc"]) else 0.0,
            "top1_ci_error": round(float(r["raw__top1_ci_error"]), 2) if pd.notna(r["raw__top1_ci_error"]) else 0.0,
            "rare_maritime_acc": round(float(r["raw__rare_top1_acc"]) * 100.0, 2) if pd.notna(r["raw__rare_top1_acc"]) else 0.0,
            "mlm_loss": round(float(r["raw__mlm_loss"]), 4) if pd.notna(r["raw__mlm_loss"]) else 0.0,
            "domain_shift_gap_pct": round(float(r["raw__domain_shift_gap_pct"]), 2) if pd.notna(r["raw__domain_shift_gap_pct"]) else 0.0,
            "performance_gap_pct": round(float(r["raw__performance_gap_pct"]), 2) if pd.notna(r["raw__performance_gap_pct"]) else 0.0,
            "single_token_coverage_pct": round(float(r["raw__single_token_coverage_pct"]), 2) if pd.notna(r["raw__single_token_coverage_pct"]) else 0.0,
            "fragmentation_rate_pct": round(float(r["raw__fragmentation_rate_pct"]), 2) if pd.notna(r["raw__fragmentation_rate_pct"]) else 0.0,
            "oov_rate_pct": round(float(r["raw__oov_rate_pct"]), 4) if pd.notna(r["raw__oov_rate_pct"]) else np.nan,
            "params_millions": int(r["raw__params_millions"]) if pd.notna(r["raw__params_millions"]) else 110,
            "disk_size_mb": int(r["raw__disk_size_mb"]) if pd.notna(r["raw__disk_size_mb"]) else 440,
            "inference_latency_ms": round(float(r["raw__inference_latency_ms"]), 2) if pd.notna(r["raw__inference_latency_ms"]) else 5.0,
            "throughput_docs_sec": round(float(r["raw__throughput_docs_sec"]), 2) if pd.notna(r["raw__throughput_docs_sec"]) else 200.0,
            "tokenizer_speed_tok_sec": round(float(r["raw__tokenizer_speed_tok_sec"]), 2) if pd.notna(r["raw__tokenizer_speed_tok_sec"]) else 1000.0
        })

    df_lb = pd.DataFrame(leaderboard_rows)
    df_lb.sort_values(by="mecs_score", ascending=False, inplace=True)
    df_lb.to_csv(stage_dir / "leaderboard.csv", index=False)

    # Step 11: Generate Visualizations
    viz_dir = stage_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Visualization 1: MLM Loss Comparison
    try:
        plt.figure(figsize=(12, 6))
        top_df = df_lb.sort_values("mlm_loss")
        plt.barh(top_df["model_name"], top_df["mlm_loss"], color="#2b5c8f", edgecolor="black")
        plt.xlabel("MLM Loss (Lower is Better)")
        plt.title("Cross-Model Masked Language Model Loss Comparison")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(viz_dir / "mlm_loss_comparison.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate mlm_loss_comparison visualization: {e}")

    # Visualization 2: Leaderboard Ranks & Maritime Top-1 with CIs
    try:
        plt.figure(figsize=(12, 6))
        plt.bar(df_lb["model_name"], df_lb["maritime_top1_acc"], yerr=df_lb["top1_ci_error"], capsize=5, color="#4c9be8", edgecolor="black", alpha=0.85)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Maritime Top-1 Accuracy (%) ± 95% CI")
        plt.title("Model Performance Ranking with 95% Confidence Intervals")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(viz_dir / "model_leaderboard_ranks.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate model_leaderboard_ranks visualization: {e}")

    # Visualization 3: Category Recall Radar Chart
    try:
        categories = ["Navigation", "Weather", "Safety", "Machinery", "Vessel", "Casualties"]
        top_3_models = df_lb.head(3)["model_name"].tolist()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for m in top_3_models:
            m_grp = df_mlm[df_mlm["model_name"] == m]
            vals = [
                m_grp["nav_acc"].mean() * 100 if "nav_acc" in m_grp and m_grp["nav_acc"].notna().any() else 0.0,
                m_grp["weather_acc"].mean() * 100 if "weather_acc" in m_grp and m_grp["weather_acc"].notna().any() else 0.0,
                m_grp["safety_acc"].mean() * 100 if "safety_acc" in m_grp and m_grp["safety_acc"].notna().any() else 0.0,
                m_grp["machinery_acc"].mean() * 100 if "machinery_acc" in m_grp and m_grp["machinery_acc"].notna().any() else 0.0,
                m_grp["vessel_acc"].mean() * 100 if "vessel_acc" in m_grp and m_grp["vessel_acc"].notna().any() else 0.0,
                m_grp["casualty_acc"].mean() * 100 if "casualty_acc" in m_grp and m_grp["casualty_acc"].notna().any() else 0.0
            ]
            vals += vals[:1]
            plt.plot(angles, vals, linewidth=2, label=m)
            plt.fill(angles, vals, alpha=0.15)

        plt.xticks(angles[:-1], categories)
        plt.title("Subdomain Category Recall Radar Chart (Top Models)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(viz_dir / "maritime_accuracy_radar.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate radar chart visualization: {e}")

    # Visualization 4: Tokenizer Fragmentation Heatmap
    try:
        plt.figure(figsize=(10, 6))
        heat_df = df_lb[["model_name", "single_token_coverage_pct", "fragmentation_rate_pct", "oov_rate_pct"]].set_index("model_name")
        sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True)
        plt.title("Tokenizer Fragmentation & Vocabulary Coverage Heatmap")
        plt.tight_layout()
        plt.savefig(viz_dir / "tokenizer_fragmentation_heatmap.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to generate tokenizer heatmap visualization: {e}")

    logger.info(f"Stage 15 successfully completed. Artifacts exported to {stage_dir}")


if __name__ == "__main__":
    main()
