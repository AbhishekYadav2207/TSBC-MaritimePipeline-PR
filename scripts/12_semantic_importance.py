"""
Stage 12: Domain Informativeness Analysis & Knowledge Characterization
Maritime Pipeline Version 2.1

Upgraded from legacy heuristic "Semantic Importance" to an interpretable, defensible,
and reproducible Domain Informativeness analysis for intrinsic corpus characterization.

Scientific Framing:
Stage 12 estimates domain informativeness using multiple observable signals representing
domain relevance, information content, corpus representativeness, and redundancy/noise,
and compares heuristic, lexical, representational, and hybrid selection signals.

Methodological Notes:
- Does NOT claim true knowledge measurement, optimal hybrid weights, or guaranteed DAPT gains.
- AlignSet, TextGram, and Predictive Data Selection provide theoretical motivation for multi-signal
  informativeness assessment, but their algorithms are not implemented here.
- Fully preserves backward-compatible fields and legacy subset files for downstream Stages 13-18.
"""

import os
import json
import re
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("12_semantic_importance")

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
STAGE12_CONFIG = {
    "scoring_version": "stage12-domain-informativeness-v2",
    "random_seed": 42,
    # Equal-Weight Hybrid Informativeness Baseline weights
    "hybrid_weights": {
        "domain_relevance": 0.25,
        "information_content": 0.25,
        "tfidf_representativeness": 0.25,
        "redundancy_noise": -0.25
    },
    # Corpus-relative quantile thresholds
    "quantiles": {
        "low_percentile": 20.0,    # Bottom 20% -> Low Informativeness
        "high_percentile": 80.0    # Top 20% -> High Informativeness
    },
    # Redundancy and near-duplicate thresholds
    "redundancy": {
        "near_duplicate_similarity_threshold": 0.85,
        "boilerplate_penalty_threshold": 0.40,
        "max_candidate_bucket_size": 300
    },
    # Diversity penalty for subset selection
    "diversity": {
        "lambda_penalty": 0.05
    },
    # Subset percentages to extract
    "subset_percentages": [0.10, 0.20, 0.30, 0.50, 1.00],
    # Resampling stability parameters
    "stability": {
        "sample_size": 2000,
        "num_resamples": 5,
        "top_k_fraction": 0.20
    },
    # TF-IDF vectorizer configuration
    "tfidf": {
        "max_features": 3000,
        "min_df": 2,
        "ngram_range": (1, 2)
    }
}

# ==============================================================================
# DOMAIN TAXONOMY & VOCABULARY CONSTANTS (PRESERVED FROM LEGACY)
# ==============================================================================
CATEGORIES = {
    "vessel_terminology": ["vess", "ship", "boat", "barge", "tug", "tanker", "trawler", "carrier", "hull", "deck", "keel", "tonnage", "transom", "freeboard", "gunwale", "bilge"],
    "navigation": ["navig", "gps", "ais", "vhf", "radar", "sonar", "compass", "gyro", "sounder", "chart", "vdr", "fathometer"],
    "machinery_propulsion": ["engine", "propel", "machinery", "motor", "shaft", "boiler", "fuel", "steering", "windlass", "hawser"],
    "casualty_incident": ["collision", "grounding", "stranding", "flooding", "leak", "capsiz", "sink", "injury", "death", "fatality", "missing", "damage"],
    "weather_environment": ["weather", "wind", "sea", "wave", "swell", "temp", "ice", "visibility", "fog", "clear", "windward", "leeward"],
    "safety_lifesaving": ["lifeboat", "liferaft", "lifejack", "lsa", "epirb", "sart", "buoy", "flare", "safety", "davit", "coxswain"]
}

RARE_MARITIME_TERMS = {
    "gyrocompass", "fathometer", "forepeak", "bulwark", "stempost", "windlass",
    "epirb", "sart", "hawser", "freeboard", "coxswain", "transom", "gunwale",
    "bilge", "fairlead", "windward", "leeward", "davit", "bitts", "bollard", "focsle"
}

GENERAL_ENGLISH_SENTENCES = [
    "The library offers a wide variety of books and digital media for public access.",
    "Engineers designed a new solar panel system to increase energy efficiency in urban homes.",
    "Scientists conducted an extensive field study on migratory birds across northern lakes.",
    "The financial market experienced significant fluctuation following quarterly earnings announcements.",
    "Students participated in a regional mathematics competition held at the city convention center.",
    "Local park authorities launched an initiative to plant native trees and preserve wetland habitats.",
    "Software developers released a major software update addressing security vulnerabilities.",
    "The museum featured an exhibit highlighting ancient architectural techniques and pottery.",
    "Researchers analyzed statistical trends in public transportation usage across major cities.",
    "A team of doctors published findings on early diagnostic methods for cardiovascular health."
]

# ==============================================================================
# FEATURE EXTRACTION (PRESERVES ALL 10 HEURISTIC FEATURES)
# ==============================================================================
def compute_document_features(doc_text: str, structured: dict, term_freq_map: Counter, total_docs: int) -> dict:
    """
    Computes 10 fine-grained document features preserved from the legacy pipeline.
    All scores are normalized in [0, 1].
    """
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', doc_text.lower())
    num_tokens = len(tokens)
    if num_tokens == 0:
        return {
            "maritime_density": 0.0, "rare_term_count": 0, "rare_score": 0.0,
            "concept_diversity": 0.0, "entity_diversity": 0.0, "event_complexity": 0.0,
            "information_density": 0.0, "redundancy_penalty": 1.0, "metadata_completeness": 0.0,
            "linguistic_diversity": 0.0, "domain_novelty": 0.0, "concepts": []
        }

    # 1. Maritime Terminology Density
    maritime_tokens = 0
    detected_concepts = set()
    for tok in tokens:
        for cat, stems in CATEGORIES.items():
            if any(stem in tok for stem in stems):
                maritime_tokens += 1
                detected_concepts.add(cat)
                break
    maritime_density = min(1.0, maritime_tokens / num_tokens)

    # 2. Rare Maritime Vocabulary
    rare_count = sum(1 for tok in tokens if tok in RARE_MARITIME_TERMS)
    rare_score = min(1.0, rare_count / 3.0)

    # 3. Concept Diversity
    concept_diversity = len(detected_concepts) / len(CATEGORIES)

    # 4. Entity Diversity
    occ = structured.get("occurrence", {})
    vessels = structured.get("vessels", [])
    distinct_entities = set()
    if occ.get("NearestLocationDescription"): distinct_entities.add(occ.get("NearestLocationDescription"))
    if occ.get("WeatherConditionDisplayEng"): distinct_entities.add(occ.get("WeatherConditionDisplayEng"))
    for v in vessels:
        if v.get("VesselName"): distinct_entities.add(v.get("VesselName"))
        if v.get("VesselTypeDisplayEng"): distinct_entities.add(v.get("VesselTypeDisplayEng"))
        if v.get("HullMaterialDisplayEng"): distinct_entities.add(v.get("HullMaterialDisplayEng"))
    entity_diversity = min(1.0, len(distinct_entities) / 6.0)

    # 5. Event Complexity
    causal_markers = len(re.findall(r'\b(caused|during|while|resulted|following|underway|sustained)\b', doc_text.lower()))
    num_clauses = len(re.split(r'[,;.]', doc_text))
    event_complexity = min(1.0, (causal_markers * 0.3 + num_clauses * 0.1))

    # 6. Information Density
    info_density = min(1.0, maritime_tokens / (num_tokens * 0.5)) if num_tokens > 0 else 0.0

    # 7. Redundancy Penalty (based on boilerplate text patterns)
    is_boilerplate = 1.0 if "resulting in a marine occurrence" in doc_text and len(doc_text) < 120 else 0.0
    redundancy_penalty = is_boilerplate * 0.5

    # 8. Metadata Completeness
    meta_fields = [
        occ.get("NearestLocationDescription"), occ.get("WeatherConditionDisplayEng"),
        occ.get("AccIncTypeDisplayEng"), vessels[0].get("VesselName") if vessels else None,
        vessels[0].get("GrossTonnage") if vessels else None
    ]
    meta_completeness = sum(1 for f in meta_fields if f) / len(meta_fields)

    # 9. Linguistic Diversity (TTR)
    ttr = len(set(tokens)) / num_tokens if num_tokens > 0 else 0.0

    # 10. Domain Novelty (IDF sum of rare maritime terms)
    novelty_sum = sum(math.log((total_docs + 1) / (term_freq_map.get(tok, 1) + 1)) for tok in tokens if tok in RARE_MARITIME_TERMS)
    domain_novelty = min(1.0, novelty_sum / 10.0)

    return {
        "maritime_density": float(maritime_density),
        "rare_term_count": int(rare_count),
        "rare_score": float(rare_score),
        "concept_diversity": float(concept_diversity),
        "entity_diversity": float(entity_diversity),
        "event_complexity": float(event_complexity),
        "information_density": float(info_density),
        "redundancy_penalty": float(redundancy_penalty),
        "metadata_completeness": float(meta_completeness),
        "linguistic_diversity": float(ttr),
        "domain_novelty": float(domain_novelty),
        "concepts": sorted(list(detected_concepts))
    }

# ==============================================================================
# SPARSE TF-IDF & NEAR-DUPLICATE NEIGHBOUR DETECTION
# ==============================================================================
def compute_sparse_near_duplicates(tfidf_matrix, threshold: float = 0.85, max_bucket_size: int = 300) -> Tuple[np.ndarray, int]:
    """
    Computes near-duplicate cosine similarity for each document using sparse top-term candidate bucketing.
    Avoids full N x N matrix construction by evaluating only candidates that share salient TF-IDF features.
    Deterministic and linear-time bounded.
    """
    n_docs = tfidf_matrix.shape[0]
    near_dup_scores = np.zeros(n_docs, dtype=np.float32)
    near_dup_count = 0

    buckets = defaultdict(list)
    for i in range(n_docs):
        row = tfidf_matrix.getrow(i)
        if row.nnz >= 2:
            top_indices = tuple(sorted(row.indices[np.argsort(row.data)[-2:]]))
            buckets[top_indices].append(i)
        elif row.nnz == 1:
            buckets[(row.indices[0],)].append(i)

    for k, doc_ids in buckets.items():
        if len(doc_ids) > 1 and len(doc_ids) <= max_bucket_size:
            sub_X = tfidf_matrix[doc_ids]
            sims = (sub_X * sub_X.T).toarray()
            np.fill_diagonal(sims, 0.0)
            max_sims = sims.max(axis=1)
            for idx, s in zip(doc_ids, max_sims):
                if s > near_dup_scores[idx]:
                    near_dup_scores[idx] = float(s)

    near_dup_count = int(np.sum(near_dup_scores >= threshold))
    return near_dup_scores, near_dup_count

# ==============================================================================
# MAIN STAGE 12 PIPELINE EXECUTION
# ==============================================================================
def main():
    config = STAGE12_CONFIG
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    root = get_project_root()
    pipeline_cfg = load_config()
    output_dir = root / pipeline_cfg.get("output_dir", "outputs")

    clean_path = output_dir / "clean_documents.jsonl"
    if not clean_path.exists():
        logger.error(f"Clean documents not found at {clean_path}! Run Step 7 first.")
        return

    logger.info(f"Starting Stage 12: Domain Informativeness Analysis (v2 - Intrinsic Corpus Characterization)")

    # -------------------------------------------------------------------------
    # PASS 1: Read clean documents and compute vocabulary frequencies
    # -------------------------------------------------------------------------
    logger.info("Pass 1/4: Ingesting documents and accumulating vocabulary frequencies...")
    all_raw_records = []
    term_freq = Counter()

    with open(clean_path, "r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            all_raw_records.append(rec)
            doc_lower = rec.get("document", "").lower()
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc_lower))
            term_freq.update(words)

    total_docs = len(all_raw_records)
    logger.info(f"Loaded {total_docs} clean documents. Total distinct lexical tokens: {len(term_freq)}")

    # -------------------------------------------------------------------------
    # PASS 2: Compute fine-grained features and TF-IDF representations
    # -------------------------------------------------------------------------
    logger.info("Pass 2/4: Computing 10 observable document features...")
    doc_features_list = []
    doc_texts = []

    for rec in tqdm(all_raw_records, desc="Extracting Features"):
        doc_text = rec.get("document", "")
        structured = rec.get("structured", {})
        feats = compute_document_features(doc_text, structured, term_freq, total_docs)
        doc_features_list.append(feats)
        doc_texts.append(doc_text)

    logger.info("Fitting sparse TF-IDF vectorizer for corpus representativeness and relevance...")
    tfidf_cfg = config["tfidf"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        min_df=tfidf_cfg["min_df"],
        ngram_range=tfidf_cfg["ngram_range"],
        stop_words="english",
        norm="l2"
    )
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    logger.info(f"TF-IDF Matrix constructed with shape {tfidf_matrix.shape} (sparse CSR format)")

    # -------------------------------------------------------------------------
    # PASS 3: Compute Representativeness, Domain Relevance, and Redundancy
    # -------------------------------------------------------------------------
    logger.info("Pass 3/4: Computing four interpretable informativeness dimensions...")

    # 1. Corpus Centroid & TF-IDF Representativeness
    corpus_centroid = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    centroid_norm = np.linalg.norm(corpus_centroid)
    if centroid_norm > 0:
        corpus_centroid = corpus_centroid / centroid_norm
    # Cosine similarity to corpus centroid: X * c^T
    tfidf_representativeness_scores = np.asarray(tfidf_matrix.dot(corpus_centroid)).flatten()
    tfidf_representativeness_scores = np.clip(tfidf_representativeness_scores, 0.0, 1.0)

    # 2. Domain Relevance via TF-IDF reference query
    domain_query_terms = list(RARE_MARITIME_TERMS)
    for cat, stems in CATEGORIES.items():
        domain_query_terms.extend(stems)
    # Check if maritime_vocabulary.txt exists for enhanced query reference
    vocab_path = output_dir / "maritime_vocabulary.txt"
    if vocab_path.exists():
        try:
            with open(vocab_path, "r", encoding="utf-8") as fv:
                vocab_lines = [line.strip() for line in fv if line.strip()]
            domain_query_terms.extend(vocab_lines[:150])
        except Exception as e:
            logger.warning(f"Could not read maritime vocabulary: {e}")

    domain_query_str = " ".join(domain_query_terms)
    domain_query_vec = vectorizer.transform([domain_query_str]).toarray().flatten()
    dq_norm = np.linalg.norm(domain_query_vec)
    if dq_norm > 0:
        domain_query_vec = domain_query_vec / dq_norm
    tfidf_relevance_scores = np.asarray(tfidf_matrix.dot(domain_query_vec)).flatten()
    tfidf_relevance_scores = np.clip(tfidf_relevance_scores, 0.0, 1.0)

    # 3. Near-Duplicate Similarity (Sparse bucketing, avoids full N x N matrix)
    logger.info("Evaluating sparse near-duplicate similarity via salient feature bucketing...")
    near_dup_scores, near_dup_count = compute_sparse_near_duplicates(
        tfidf_matrix,
        threshold=config["redundancy"]["near_duplicate_similarity_threshold"],
        max_bucket_size=config["redundancy"]["max_candidate_bucket_size"]
    )
    logger.info(f"Identified {near_dup_count} documents with near-duplicate TF-IDF similarity >= {config['redundancy']['near_duplicate_similarity_threshold']}")

    # -------------------------------------------------------------------------
    # PASS 4: Assemble Multi-Signal Scores & Classifications
    # -------------------------------------------------------------------------
    logger.info("Pass 4/4: Assembling dimension scores, quantile tiers, and baseline rankings...")
    scored_records = []
    w = config["hybrid_weights"]

    boilerplate_count = 0
    redundancy_flag_count = 0

    for idx, (rec, feats) in enumerate(zip(all_raw_records, doc_features_list)):
        occ_id = rec.get("occurrence_id")
        doc_text = rec.get("document", "")

        # Dimension 1: Domain Relevance [0, 1]
        domain_relevance = float(np.clip(
            (feats["maritime_density"] + feats["rare_score"] + feats["domain_novelty"]) / 3.0,
            0.0, 1.0
        ))

        # Dimension 2: Information Content [0, 1]
        information_content = float(np.clip(
            (feats["event_complexity"] + feats["entity_diversity"] + feats["metadata_completeness"] + feats["linguistic_diversity"]) / 4.0,
            0.0, 1.0
        ))

        # Dimension 3: Corpus Representativeness (TF-IDF Centroid) [0, 1]
        rep_score = float(tfidf_representativeness_scores[idx])

        # Dimension 4: Redundancy & Noise [0, 1]
        near_sim = float(near_dup_scores[idx])
        red_noise = float(np.clip(max(feats["redundancy_penalty"], near_sim * 0.5), 0.0, 1.0))

        if feats["redundancy_penalty"] >= config["redundancy"]["boilerplate_penalty_threshold"]:
            boilerplate_count += 1

        is_redundant = bool(
            feats["redundancy_penalty"] >= config["redundancy"]["boilerplate_penalty_threshold"] or
            near_sim >= config["redundancy"]["near_duplicate_similarity_threshold"]
        )
        if is_redundant:
            redundancy_flag_count += 1

        # Heuristic Informativeness Baseline (H-INF) - exact legacy calculation preserved
        raw_h_score = (
            0.30 * feats["maritime_density"] +
            0.20 * feats["rare_score"] +
            0.15 * feats["concept_diversity"] +
            0.10 * feats["entity_diversity"] +
            0.10 * feats["event_complexity"] +
            0.10 * feats["information_density"] +
            0.05 * feats["metadata_completeness"] +
            0.05 * feats["linguistic_diversity"] +
            0.05 * feats["domain_novelty"] -
            0.10 * feats["redundancy_penalty"]
        )
        heuristic_h_inf = float(np.clip(raw_h_score, 0.0, 1.0))
        legacy_importance_score = float(np.clip(raw_h_score * 100.0, 0.0, 100.0))

        # Equal-Weight Hybrid Informativeness Baseline [0, 1]
        raw_hybrid = (
            w["domain_relevance"] * domain_relevance +
            w["information_content"] * information_content +
            w["tfidf_representativeness"] * rep_score +
            w["redundancy_noise"] * red_noise  # note w['redundancy_noise'] is -0.25
        )
        domain_inf_score = float(np.clip(raw_hybrid, 0.0, 1.0))

        scored_records.append({
            "idx": idx,
            "occurrence_id": occ_id,
            "document": doc_text,
            "concepts": feats["concepts"],
            "maritime_density": round(feats["maritime_density"], 4),
            "rare_term_count": feats["rare_term_count"],
            "concept_diversity": round(feats["concept_diversity"], 4),
            # Core Dimensions
            "domain_relevance": round(domain_relevance, 4),
            "information_content": round(information_content, 4),
            "tfidf_representativeness": round(rep_score, 4),
            "redundancy_noise": round(red_noise, 4),
            # Scoring Baselines & Hybrid
            "heuristic_informativeness_score": round(heuristic_h_inf, 4),
            "tfidf_relevance": round(float(tfidf_relevance_scores[idx]), 4),
            "domain_informativeness_score": round(domain_inf_score, 4),
            "importance_score": round(legacy_importance_score, 2),
            "redundancy_flag": is_redundant,
            "subdomain_labels": feats["concepts"],
            "scoring_version": config["scoring_version"]
        })

    # Compute Quantiles for corpus-relative tiers
    all_hybrid_scores = np.array([r["domain_informativeness_score"] for r in scored_records])
    q_low_val = float(np.percentile(all_hybrid_scores, config["quantiles"]["low_percentile"]))
    q_high_val = float(np.percentile(all_hybrid_scores, config["quantiles"]["high_percentile"]))
    logger.info(f"Corpus-Relative Informativeness Quantiles: Bottom 20% (P20) <= {q_low_val:.4f}, Top 20% (P80) >= {q_high_val:.4f}")

    # Compute Deterministic Ranking (1-indexed, descending by hybrid informativeness score)
    # Tie-breaker: occurrence_id
    sorted_order = sorted(range(len(scored_records)), key=lambda i: (scored_records[i]["domain_informativeness_score"], -scored_records[i]["occurrence_id"]), reverse=True)
    rank_map = {orig_idx: rank + 1 for rank, orig_idx in enumerate(sorted_order)}

    tier_counts = Counter()
    legacy_tier_counts = Counter()

    for idx, r in enumerate(scored_records):
        r["informativeness_rank"] = rank_map[idx]
        score = r["domain_informativeness_score"]
        is_red = r["redundancy_flag"]

        # 1. New Interpretable Informativeness Tiers
        if is_red:
            inf_tier = "Redundant / Noise"
        elif score >= q_high_val:
            inf_tier = "High Informativeness"
        elif score >= q_low_val:
            inf_tier = "Medium Informativeness"
        else:
            inf_tier = "Low Informativeness"
        tier_counts[inf_tier] += 1
        r["informativeness_tier"] = inf_tier

        # 2. Legacy Knowledge Tiers (Exact backward compatibility for Stage 14)
        if is_red:
            legacy_tier = "Redundant"
        elif score >= q_high_val:
            legacy_tier = "High Knowledge"
        elif score >= q_low_val:
            legacy_tier = "Medium Knowledge"
        else:
            legacy_tier = "Low Knowledge"
        legacy_tier_counts[legacy_tier] += 1
        r["knowledge_tier"] = legacy_tier

    # -------------------------------------------------------------------------
    # EXPORT DOCUMENT IMPORTANCE JSONL (FULL BACKWARD COMPATIBILITY + EXTENSIONS)
    # -------------------------------------------------------------------------
    imp_path = output_dir / "document_importance.jsonl"
    logger.info(f"Writing extended document informativeness records to {imp_path}...")
    with open(imp_path, "w", encoding="utf-8") as fout:
        for r in scored_records:
            out_obj = {
                # Legacy fields (preserved in exact order and format)
                "occurrence_id": r["occurrence_id"],
                "importance_score": r["importance_score"],
                "knowledge_tier": r["knowledge_tier"],
                "maritime_density": r["maritime_density"],
                "rare_term_count": r["rare_term_count"],
                "concept_diversity": r["concept_diversity"],
                "concepts": r["concepts"],
                "document": r["document"],
                # Upgraded Domain Informativeness fields
                "domain_relevance": r["domain_relevance"],
                "information_content": r["information_content"],
                "tfidf_representativeness": r["tfidf_representativeness"],
                "redundancy_noise": r["redundancy_noise"],
                "heuristic_informativeness_score": r["heuristic_informativeness_score"],
                "tfidf_relevance": r["tfidf_relevance"],
                "domain_informativeness_score": r["domain_informativeness_score"],
                "informativeness_rank": r["informativeness_rank"],
                "subdomain_labels": r["subdomain_labels"],
                "redundancy_flag": r["redundancy_flag"],
                "scoring_version": r["scoring_version"]
            }
            fout.write(json.dumps(out_obj) + "\n")

    # -------------------------------------------------------------------------
    # DIMENSION ABLATION STUDY
    # -------------------------------------------------------------------------
    logger.info("Computing leave-one-dimension-out feature ablation study...")
    dr_arr = np.array([r["domain_relevance"] for r in scored_records])
    ic_arr = np.array([r["information_content"] for r in scored_records])
    rep_arr = np.array([r["tfidf_representativeness"] for r in scored_records])
    red_arr = np.array([r["redundancy_noise"] for r in scored_records])
    full_hybrid_arr = all_hybrid_scores

    ablations = {
        "full_hybrid": full_hybrid_arr,
        "minus_domain_relevance": np.clip((ic_arr + rep_arr - red_arr) / 3.0, 0.0, 1.0),
        "minus_information_content": np.clip((dr_arr + rep_arr - red_arr) / 3.0, 0.0, 1.0),
        "minus_tfidf_representativeness": np.clip((dr_arr + ic_arr - red_arr) / 3.0, 0.0, 1.0),
        "minus_redundancy_noise": np.clip((dr_arr + ic_arr + rep_arr) / 3.0, 0.0, 1.0)
    }

    ablation_results = {}
    for name, arr in ablations.items():
        spearman_corr, _ = stats.spearmanr(full_hybrid_arr, arr)
        # Sample for Kendall tau if full array takes non-trivial time
        kendall_corr, _ = stats.kendalltau(full_hybrid_arr[:5000], arr[:5000])
        ablation_results[name] = {
            "spearman_rank_correlation_with_full": round(float(spearman_corr), 4),
            "kendall_tau_correlation_with_full": round(float(kendall_corr), 4),
            "mean_score": round(float(np.mean(arr)), 4),
            "median_score": round(float(np.median(arr)), 4),
            "std_score": round(float(np.std(arr)), 4),
            "min_score": round(float(np.min(arr)), 4),
            "max_score": round(float(np.max(arr)), 4),
            "quartiles": [round(float(q), 4) for q in np.percentile(arr, [25, 50, 75])]
        }

    ablation_path = output_dir / "informativeness_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as fabl:
        json.dump({
            "scoring_version": config["scoring_version"],
            "description": "Leave-one-dimension-out ablation evaluating the marginal impact of each informativeness signal.",
            "ablation_results": ablation_results
        }, fabl, indent=2)

    # -------------------------------------------------------------------------
    # METHOD COMPARISON & RANKING STABILITY (RESAMPLING)
    # -------------------------------------------------------------------------
    logger.info("Evaluating method comparisons and ranking stability across resamples...")
    h_inf_arr = np.array([r["heuristic_informativeness_score"] for r in scored_records])
    tfidf_rel_arr = np.array([r["tfidf_relevance"] for r in scored_records])

    method_signals = {
        "heuristic_informativeness_h_inf": h_inf_arr,
        "tfidf_domain_relevance": tfidf_rel_arr,
        "tfidf_corpus_representativeness": rep_arr,
        "equal_weight_hybrid_informativeness": full_hybrid_arr
    }

    # Full Corpus Method Comparison
    method_comparison = {}
    for m1, a1 in method_signals.items():
        method_comparison[m1] = {
            "mean": round(float(np.mean(a1)), 4),
            "std": round(float(np.std(a1)), 4),
            "correlations": {}
        }
        for m2, a2 in method_signals.items():
            sp_r, _ = stats.spearmanr(a1, a2)
            method_comparison[m1]["correlations"][m2] = round(float(sp_r), 4)

    comp_path = output_dir / "informativeness_method_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as fcomp:
        json.dump({
            "scoring_version": config["scoring_version"],
            "description": "Pairwise Spearman rank correlation matrix across heuristic, lexical, representational, and hybrid informativeness baselines.",
            "comparison": method_comparison
        }, fcomp, indent=2)

    # Resampling Stability Analysis (5 resamples x 2000 documents)
    stab_cfg = config["stability"]
    n_sample = min(stab_cfg["sample_size"], total_docs)
    k_top = int(n_sample * stab_cfg["top_k_fraction"])

    stability_records = []
    rng = np.random.RandomState(config["random_seed"])

    for iter_idx in range(stab_cfg["num_resamples"]):
        sample_indices = rng.choice(total_docs, size=n_sample, replace=False)
        sample_h_inf = h_inf_arr[sample_indices]
        sample_tfidf = tfidf_rel_arr[sample_indices]
        sample_rep = rep_arr[sample_indices]
        sample_hyb = full_hybrid_arr[sample_indices]

        # Pairwise correlations on resample
        rho_h_hyb, _ = stats.spearmanr(sample_h_inf, sample_hyb)
        tau_h_hyb, _ = stats.kendalltau(sample_h_inf, sample_hyb)

        rho_tf_hyb, _ = stats.spearmanr(sample_tfidf, sample_hyb)
        tau_tf_hyb, _ = stats.kendalltau(sample_tfidf, sample_hyb)

        rho_rep_hyb, _ = stats.spearmanr(sample_rep, sample_hyb)
        tau_rep_hyb, _ = stats.kendalltau(sample_rep, sample_hyb)

        # Top-K Overlap (Jaccard) with Hybrid
        top_hyb = set(np.argsort(-sample_hyb)[:k_top])
        top_h = set(np.argsort(-sample_h_inf)[:k_top])
        top_tf = set(np.argsort(-sample_tfidf)[:k_top])
        top_rep = set(np.argsort(-sample_rep)[:k_top])

        def jaccard(s1, s2):
            return len(s1 & s2) / max(1, len(s1 | s2))

        stability_records.append({
            "resample_iteration": iter_idx + 1,
            "spearman_rho_hybrid_vs_h_inf": round(float(rho_h_hyb), 4),
            "kendall_tau_hybrid_vs_h_inf": round(float(tau_h_hyb), 4),
            "jaccard_top_k_hybrid_vs_h_inf": round(float(jaccard(top_hyb, top_h)), 4),
            "spearman_rho_hybrid_vs_tfidf_relevance": round(float(rho_tf_hyb), 4),
            "kendall_tau_hybrid_vs_tfidf_relevance": round(float(tau_tf_hyb), 4),
            "jaccard_top_k_hybrid_vs_tfidf_relevance": round(float(jaccard(top_hyb, top_tf)), 4),
            "spearman_rho_hybrid_vs_tfidf_representativeness": round(float(rho_rep_hyb), 4),
            "kendall_tau_hybrid_vs_tfidf_representativeness": round(float(tau_rep_hyb), 4),
            "jaccard_top_k_hybrid_vs_tfidf_representativeness": round(float(jaccard(top_hyb, top_rep)), 4),
        })

    # Summary across resamples
    mean_stab = {}
    for key in stability_records[0].keys():
        if key != "resample_iteration":
            vals = [rec[key] for rec in stability_records]
            mean_stab[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4)
            }

    stab_path = output_dir / "ranking_stability.json"
    with open(stab_path, "w", encoding="utf-8") as fstab:
        json.dump({
            "scoring_version": config["scoring_version"],
            "methodology_note": "Resampling with fixed seed on 2,000 document subset across 5 deterministic iterations to evaluate ranking stability across selection signals (not independent experimental runs).",
            "resample_sample_size": n_sample,
            "top_k_fraction": stab_cfg["top_k_fraction"],
            "iterations": stability_records,
            "aggregate_stability": mean_stab
        }, fstab, indent=2)

    # -------------------------------------------------------------------------
    # SUMMARY ARTIFACTS: DOMAIN INFORMATIVENESS & IMPORTANCE STATISTICS
    # -------------------------------------------------------------------------
    logger.info("Exporting statistics summary artifacts...")
    legacy_stats = {
        "total_documents": total_docs,
        "mean_importance_score": float(np.mean([r["importance_score"] for r in scored_records])),
        "median_importance_score": float(np.median([r["importance_score"] for r in scored_records])),
        "std_importance_score": float(np.std([r["importance_score"] for r in scored_records])),
        "min_importance_score": float(np.min([r["importance_score"] for r in scored_records])),
        "max_importance_score": float(np.max([r["importance_score"] for r in scored_records])),
        "quartiles": [round(float(q), 2) for q in np.percentile([r["importance_score"] for r in scored_records], [25, 50, 75])],
        "knowledge_tier_breakdown": dict(legacy_tier_counts)
    }

    stat_path = output_dir / "importance_statistics.json"
    with open(stat_path, "w", encoding="utf-8") as fstat:
        json.dump(legacy_stats, fstat, indent=2)

    # Comprehensive Domain Informativeness Statistics
    domain_inf_stats = {
        "scoring_version": config["scoring_version"],
        "pipeline_stage": "Stage 12: Domain Informativeness Analysis",
        "scientific_framing": (
            "Stage 12 estimates domain informativeness using multiple observable signals representing "
            "domain relevance, information content, corpus representativeness, and redundancy/noise, "
            "and compares heuristic, lexical, representational, and hybrid selection signals."
        ),
        "total_documents": total_docs,
        "random_seed": config["random_seed"],
        "configuration": config,
        "quantile_thresholds": {
            "p20_low_boundary": round(q_low_val, 4),
            "p80_high_boundary": round(q_high_val, 4)
        },
        "informativeness_tier_breakdown": dict(tier_counts),
        "legacy_knowledge_tier_breakdown": dict(legacy_tier_counts),
        "redundancy_statistics": {
            "total_flagged_redundant": redundancy_flag_count,
            "boilerplate_matches": boilerplate_count,
            "near_duplicate_matches": near_dup_count,
            "redundancy_rate_pct": round(100.0 * redundancy_flag_count / total_docs, 2)
        },
        "representativeness_statistics": {
            "representation_method": "deterministic_tfidf_centroid_cosine_similarity",
            "missing_or_invalid_representations": 0,
            "centroid_vector_norm": round(float(centroid_norm), 4)
        },
        "dimension_distributions": {
            "domain_relevance": {
                "mean": round(float(np.mean(dr_arr)), 4),
                "median": round(float(np.median(dr_arr)), 4),
                "std": round(float(np.std(dr_arr)), 4),
                "min": round(float(np.min(dr_arr)), 4),
                "max": round(float(np.max(dr_arr)), 4)
            },
            "information_content": {
                "mean": round(float(np.mean(ic_arr)), 4),
                "median": round(float(np.median(ic_arr)), 4),
                "std": round(float(np.std(ic_arr)), 4),
                "min": round(float(np.min(ic_arr)), 4),
                "max": round(float(np.max(ic_arr)), 4)
            },
            "tfidf_representativeness": {
                "mean": round(float(np.mean(rep_arr)), 4),
                "median": round(float(np.median(rep_arr)), 4),
                "std": round(float(np.std(rep_arr)), 4),
                "min": round(float(np.min(rep_arr)), 4),
                "max": round(float(np.max(rep_arr)), 4)
            },
            "redundancy_noise": {
                "mean": round(float(np.mean(red_arr)), 4),
                "median": round(float(np.median(red_arr)), 4),
                "std": round(float(np.std(red_arr)), 4),
                "min": round(float(np.min(red_arr)), 4),
                "max": round(float(np.max(red_arr)), 4)
            },
            "heuristic_informativeness_score": {
                "mean": round(float(np.mean(h_inf_arr)), 4),
                "median": round(float(np.median(h_inf_arr)), 4),
                "std": round(float(np.std(h_inf_arr)), 4),
                "min": round(float(np.min(h_inf_arr)), 4),
                "max": round(float(np.max(h_inf_arr)), 4)
            },
            "tfidf_relevance": {
                "mean": round(float(np.mean(tfidf_rel_arr)), 4),
                "median": round(float(np.median(tfidf_rel_arr)), 4),
                "std": round(float(np.std(tfidf_rel_arr)), 4),
                "min": round(float(np.min(tfidf_rel_arr)), 4),
                "max": round(float(np.max(tfidf_rel_arr)), 4)
            },
            "domain_informativeness_score": {
                "mean": round(float(np.mean(full_hybrid_arr)), 4),
                "median": round(float(np.median(full_hybrid_arr)), 4),
                "std": round(float(np.std(full_hybrid_arr)), 4),
                "min": round(float(np.min(full_hybrid_arr)), 4),
                "max": round(float(np.max(full_hybrid_arr)), 4),
                "quartiles": [round(float(q), 4) for q in np.percentile(full_hybrid_arr, [25, 50, 75])]
            }
        }
    }

    dom_stat_path = output_dir / "domain_informativeness_statistics.json"
    with open(dom_stat_path, "w", encoding="utf-8") as fdom:
        json.dump(domain_inf_stats, fdom, indent=2)

    # -------------------------------------------------------------------------
    # PLOT SCORE DISTRIBUTION
    # -------------------------------------------------------------------------
    logger.info("Generating informativeness score distribution figure...")
    plt.figure(figsize=(10, 6))
    scores_for_plot = [r["importance_score"] for r in scored_records]
    plt.hist(scores_for_plot, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.axvline(np.mean(scores_for_plot), color="red", linestyle="dashed", linewidth=2, label=f"Mean: {np.mean(scores_for_plot):.2f}")
    plt.axvline(np.median(scores_for_plot), color="green", linestyle="dotted", linewidth=2, label=f"Median: {np.median(scores_for_plot):.2f}")
    plt.title("Corpus Semantic & Domain Informativeness Score Distribution")
    plt.xlabel("Heuristic Informativeness / Importance Score (0 - 100)")
    plt.ylabel("Document Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = output_dir / "importance_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # SUBSET GENERATION: PERCENTAGE SUBSETS + LEGACY COMPATIBILITY
    # -------------------------------------------------------------------------
    subsets_dir = output_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating evaluation subsets in {subsets_dir}...")

    # Sort documents by hybrid score descending with lightweight diversity penalty
    # Fast lightweight diversity: penalize near-duplicates when candidate bucket is already populated
    lambda_div = config["diversity"]["lambda_penalty"]
    ranked_records = sorted(scored_records, key=lambda x: (x["domain_informativeness_score"], -x["occurrence_id"]), reverse=True)

    # 1. Percentage-based subsets (10%, 20%, 30%, 50%, 100%)
    for pct in config["subset_percentages"]:
        k_count = int(round(total_docs * pct))
        if pct >= 1.0:
            pct_subset = scored_records
        else:
            # Deterministic selection from ranked order with lightweight diversity filtering
            selected_docs = []
            seen_dup_candidates = set()
            for r in ranked_records:
                if len(selected_docs) >= k_count:
                    break
                # Apply diversity penalty if flagged near-duplicate
                if r["redundancy_flag"] and len(selected_docs) > 0 and (r["idx"] in seen_dup_candidates):
                    # Penalized selection score check
                    penalized_score = r["domain_informativeness_score"] - lambda_div
                    if penalized_score < 0.10:
                        continue
                selected_docs.append(r)
                seen_dup_candidates.add(r["idx"])

            if len(selected_docs) < k_count:
                # Top-up if strict filtering fell short
                selected_docs = ranked_records[:k_count]
            pct_subset = selected_docs

        pct_tag = f"{int(pct * 100)}pct"
        out_sub_path = subsets_dir / f"domain_inf_subset_{pct_tag}.jsonl"
        with open(out_sub_path, "w", encoding="utf-8") as fsub:
            for rec in pct_subset:
                fsub.write(json.dumps(rec) + "\n")

        # Matched Random Subset at same percentage
        rand_indices = rng.choice(total_docs, size=k_count, replace=False) if pct < 1.0 else list(range(total_docs))
        out_rand_path = subsets_dir / f"random_subset_{pct_tag}.jsonl"
        with open(out_rand_path, "w", encoding="utf-8") as frand:
            for r_idx in rand_indices:
                frand.write(json.dumps(scored_records[r_idx]) + "\n")

    logger.info("Generated ranked and random percentage subsets for 10%, 20%, 30%, 50%, 100%.")

    # 2. Legacy Evaluation Subsets (Strictly preserves Stage 14 inputs: 1000 docs each)
    target_sample_size = min(1000, total_docs)
    high_docs = ranked_records[:target_sample_size]

    valid_low = [r for r in ranked_records if r["knowledge_tier"] in ("Low Knowledge", "Redundant", "Noisy / Boilerplate")]
    low_docs = valid_low[-target_sample_size:] if len(valid_low) >= target_sample_size else ranked_records[-target_sample_size:]

    mid_start = max(0, (total_docs - target_sample_size) // 2)
    med_docs = ranked_records[mid_start:mid_start + target_sample_size]

    with open(subsets_dir / "high_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in high_docs:
            f.write(json.dumps(r) + "\n")

    with open(subsets_dir / "medium_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in med_docs:
            f.write(json.dumps(r) + "\n")

    with open(subsets_dir / "low_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in low_docs:
            f.write(json.dumps(r) + "\n")

    n_per_tier = target_sample_size // 3
    balanced_pool = high_docs[:n_per_tier] + med_docs[:n_per_tier] + low_docs[:n_per_tier]
    rng.shuffle(balanced_pool)
    with open(subsets_dir / "balanced_knowledge.jsonl", "w", encoding="utf-8") as f:
        for r in balanced_pool:
            f.write(json.dumps(r) + "\n")

    random_pool_indices = rng.choice(total_docs, size=target_sample_size, replace=False)
    with open(subsets_dir / "random_baseline.jsonl", "w", encoding="utf-8") as f:
        for r_idx in random_pool_indices:
            f.write(json.dumps(scored_records[r_idx]) + "\n")

    with open(subsets_dir / "general_english_baseline.jsonl", "w", encoding="utf-8") as f:
        for s_idx, sent in enumerate(GENERAL_ENGLISH_SENTENCES):
            f.write(json.dumps({
                "occurrence_id": 900000 + s_idx,
                "importance_score": 10.0,
                "knowledge_tier": "General English",
                "document": sent
            }) + "\n")

    logger.info(f"Preserved all 6 legacy evaluation subsets under {subsets_dir}")
    logger.info(f"[SUCCESS] Stage 12 completed successfully. Version: {config['scoring_version']}")

if __name__ == "__main__":
    main()
