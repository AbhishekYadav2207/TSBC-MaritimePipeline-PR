import os
import json
import time
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("13_tokenizer_analysis")

# Canonical Representative Subset (7 Distinct Tokenizer Families, aligned with Stage 14 & 15)
TARGET_MODELS = [
    "bert-base-uncased",                                            # Standard WordPiece (30,522) - Baseline
    "dmis-lab/biobert-base-cased-v1.2",                            # Bio/Clinical WordPiece (28,996 Cased)
    "nlpaueb/legal-bert-base-uncased",                              # Legal WordPiece (30,522)
    "allenai/scibert_scivocab_uncased",                             # SciVocab WordPiece (31,090)
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",# PubMed Domain WordPiece (30,522)
    "roberta-base",                                                 # Standard Byte-Level BPE (50,265)
    "answerdotai/ModernBERT-base"                                   # Modern Extended BPE (50,280)
]

# Canonical Rare Maritime Terminology list (standardized across Stages 12 and 14)
RARE_MARITIME_TERMS = [
    "bilge", "bitts", "bollard", "bulwark", "coxswain", "davit", "epirb", "fairlead",
    "fathometer", "forepeak", "freeboard", "gunwale", "gyrocompass", "hawser",
    "leeward", "sart", "stempost", "transom", "windlass", "windward"
]

SCIENTIFIC_DISCLAIMER = (
    "Scientific Note: This analysis measures empirical tokenizer properties and statistical associations "
    "(Spearman rank correlation) between document informativeness scores and subword tokenizer behaviors. "
    "Correlation does not imply causation; domain informativeness does not cause tokenizer behavior or "
    "downstream masked language modeling performance. Domain informativeness scores reflect algorithmic outputs "
    "from Stage 12 heuristic/hybrid scoring, not externally validated ground truth."
)

def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def analyze_vocabulary_category(tokenizer, terms: list, category_name: str) -> dict:
    """
    Analyzes subword fragmentation and piece counts for a specific vocabulary category.
    """
    vocab_splits = []
    piece_counts = []
    single_token_count = 0

    for term in terms:
        tokens = tokenizer.tokenize(term)
        num_pieces = len(tokens)
        piece_counts.append(num_pieces)
        if num_pieces == 1:
            single_token_count += 1
        vocab_splits.append({
            "term": term,
            "tokens": tokens,
            "num_pieces": num_pieces
        })

    vocab_splits.sort(key=lambda x: x["num_pieces"], reverse=True)
    total_terms = len(terms) if terms else 1

    single_token_coverage = single_token_count / total_terms
    frag_rate = (total_terms - single_token_count) / total_terms
    avg_pieces = float(np.mean(piece_counts)) if piece_counts else 0.0
    median_pieces = float(np.median(piece_counts)) if piece_counts else 0.0
    p95_pieces = float(np.percentile(piece_counts, 95)) if piece_counts else 0.0
    max_pieces = int(np.max(piece_counts)) if piece_counts else 0

    return {
        "category": category_name,
        "total_terms": len(terms),
        "single_token_count": single_token_count,
        "single_token_coverage": round(single_token_coverage, 4),
        "fragmentation_rate": round(frag_rate, 4),
        "avg_pieces_per_term": round(avg_pieces, 3),
        "median_pieces_per_term": round(median_pieces, 1),
        "p95_pieces_per_term": round(p95_pieces, 2),
        "max_pieces_per_term": max_pieces,
        "worst_fragmented_terms": vocab_splits[:15],
        "splits": vocab_splits
    }

def compute_spearman(x: list, y: list) -> dict:
    """
    Computes Spearman rank correlation with sample size reporting and zero-variance guard.
    """
    n = len(x)
    if n < 3:
        return {"rho": None, "p_value": None, "sample_size": n, "status": "insufficient_observations"}
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return {"rho": None, "p_value": None, "sample_size": n, "status": "insufficient_variance"}

    res = spearmanr(x_arr, y_arr)
    rho = float(res.statistic) if not np.isnan(res.statistic) else None
    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else None

    return {
        "rho": round(rho, 4) if rho is not None else None,
        "p_value": round(p_val, 6) if p_val is not None else None,
        "sample_size": n,
        "status": "computed" if rho is not None else "nan_encountered"
    }

def analyze_tokenizer(
    model_name: str,
    vocab_categories: dict,
    sampled_records: list
) -> dict:
    logger.info(f"Analyzing tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer '{model_name}': {e}. Skipping.")
        return None

    # Detect byte-level BPE tokenizers where [UNK] is not semantically applicable
    is_byte_bpe = any(k in model_name.lower() for k in ["roberta", "modernbert", "gpt", "llama", "byte"])

    # 1. Vocabulary Analysis across Categories (General, Maritime, Rare Maritime)
    cat_results = {}
    for cat_name, terms in vocab_categories.items():
        cat_results[cat_name] = analyze_vocabulary_category(tokenizer, terms, cat_name)

    maritime_res = cat_results.get("maritime_terminology", {})
    total_maritime_terms = maritime_res.get("total_terms", 0)
    single_token_count = maritime_res.get("single_token_count", 0)
    single_token_coverage = maritime_res.get("single_token_coverage", 0.0)
    maritime_frag_rate = maritime_res.get("fragmentation_rate", 0.0)
    avg_pieces = maritime_res.get("avg_pieces_per_term", 0.0)
    median_pieces = maritime_res.get("median_pieces_per_term", 0.0)
    p95_pieces = maritime_res.get("p95_pieces_per_term", 0.0)
    max_pieces = maritime_res.get("max_pieces_per_term", 0)
    worst_fragmented_terms = maritime_res.get("worst_fragmented_terms", [])
    maritime_splits = maritime_res.get("splits", [])

    # 2. Document-Level Profiling & Informativeness Join
    total_raw_words = 0
    total_subword_tokens = 0
    total_unk_tokens = 0
    seq_length_dist = {"under_128": 0, "under_256": 0, "under_512": 0, "over_512": 0}

    doc_profiles = []
    t0 = time.time()

    unk_tok = getattr(tokenizer, "unk_token", "[UNK]") if not is_byte_bpe else None

    for r in sampled_records:
        doc_text = r["document"]
        raw_words = doc_text.split()
        num_raw_words = len(raw_words)
        if num_raw_words == 0:
            continue

        tokens = tokenizer.tokenize(doc_text)
        num_tokens = len(tokens)
        total_raw_words += num_raw_words
        total_subword_tokens += num_tokens

        # Subword fertility for document
        fertility = num_tokens / num_raw_words

        # Word-level fragmentation and coverage for document
        split_words_count = 0
        for w in raw_words:
            w_pieces = len(tokenizer.tokenize(w))
            if w_pieces > 1:
                split_words_count += 1

        doc_frag = split_words_count / num_raw_words
        doc_cov = (num_raw_words - split_words_count) / num_raw_words

        # OOV token count
        doc_unk = 0
        if unk_tok and unk_tok in tokens:
            doc_unk = tokens.count(unk_tok)
            total_unk_tokens += doc_unk

        # Sequence length bucketing
        if num_tokens <= 128: seq_length_dist["under_128"] += 1
        elif num_tokens <= 256: seq_length_dist["under_256"] += 1
        elif num_tokens <= 512: seq_length_dist["under_512"] += 1
        else: seq_length_dist["over_512"] += 1

        doc_profiles.append({
            "occurrence_id": r.get("occurrence_id"),
            "knowledge_tier": r.get("knowledge_tier", "Unknown"),
            "domain_informativeness_score": float(r.get("domain_informativeness_score", 0.0)),
            "raw_words": num_raw_words,
            "subword_tokens": num_tokens,
            "fertility": fertility,
            "word_fragmentation": doc_frag,
            "word_coverage": doc_cov,
            "unk_count": doc_unk
        })

    tok_elapsed = time.time() - t0
    tokenizer_speed = total_subword_tokens / tok_elapsed if tok_elapsed > 0 else 0.0
    overall_fertility = total_subword_tokens / total_raw_words if total_raw_words > 0 else 0.0

    if is_byte_bpe:
        oov_rate = None
        oov_status = "not_applicable"
    else:
        oov_rate = float(total_unk_tokens / total_subword_tokens) if total_subword_tokens > 0 else 0.0
        oov_status = "measured"

    # 3. Informativeness-Stratified Tokenizer Analysis
    tier_groups = defaultdict(list)
    for p in doc_profiles:
        tier_groups[p["knowledge_tier"]].append(p)

    stratified_results = {}
    for tier_name, p_list in tier_groups.items():
        t_words = sum(p["raw_words"] for p in p_list)
        t_tokens = sum(p["subword_tokens"] for p in p_list)
        t_fert = t_tokens / t_words if t_words > 0 else 0.0
        t_frag = float(np.mean([p["word_fragmentation"] for p in p_list])) if p_list else 0.0
        t_cov = float(np.mean([p["word_coverage"] for p in p_list])) if p_list else 0.0
        t_seq_lens = [p["subword_tokens"] for p in p_list]
        t_mean_seq = float(np.mean(t_seq_lens)) if t_seq_lens else 0.0

        t_dist = {"under_128": 0, "under_256": 0, "under_512": 0, "over_512": 0}
        for sl in t_seq_lens:
            if sl <= 128: t_dist["under_128"] += 1
            elif sl <= 256: t_dist["under_256"] += 1
            elif sl <= 512: t_dist["under_512"] += 1
            else: t_dist["over_512"] += 1

        if is_byte_bpe:
            t_oov = None
            t_oov_status = "not_applicable"
        else:
            t_unks = sum(p["unk_count"] for p in p_list)
            t_oov = float(t_unks / t_tokens) if t_tokens > 0 else 0.0
            t_oov_status = "measured"

        stratified_results[tier_name] = {
            "tier": tier_name,
            "document_count": len(p_list),
            "total_raw_words": t_words,
            "total_subword_tokens": t_tokens,
            "mean_fertility": round(t_fert, 4),
            "mean_sequence_length": round(t_mean_seq, 2),
            "mean_document_fragmentation": round(t_frag, 4),
            "mean_document_coverage": round(t_cov, 4),
            "oov_rate": round(t_oov, 6) if t_oov is not None else None,
            "oov_status": t_oov_status,
            "sequence_length_distribution": t_dist
        }

    # 4. Correlation Analysis (Spearman rho)
    inf_scores = [p["domain_informativeness_score"] for p in doc_profiles]
    fertilities = [p["fertility"] for p in doc_profiles]
    frags = [p["word_fragmentation"] for p in doc_profiles]
    covs = [p["word_coverage"] for p in doc_profiles]

    correlations = {
        "informativeness_vs_fertility": compute_spearman(inf_scores, fertilities),
        "informativeness_vs_fragmentation": compute_spearman(inf_scores, frags),
        "informativeness_vs_coverage": compute_spearman(inf_scores, covs),
        "scientific_interpretation": SCIENTIFIC_DISCLAIMER
    }

    # 5. Deterministic Term Examples
    maritime_sorted = sorted(maritime_splits, key=lambda x: x["term"])
    general_sorted = sorted(cat_results.get("general_vocabulary", {}).get("splits", []), key=lambda x: x["term"])
    rare_sorted = sorted(cat_results.get("rare_maritime_terminology", {}).get("splits", []), key=lambda x: x["term"])

    term_examples = {
        "general_maritime_examples": [
            {"term": item["term"], "tokens": item["tokens"], "num_pieces": item["num_pieces"]}
            for item in maritime_sorted[:12]
        ],
        "rare_maritime_examples": [
            {"term": item["term"], "tokens": item["tokens"], "num_pieces": item["num_pieces"]}
            for item in rare_sorted
        ],
        "general_vocabulary_examples": [
            {"term": item["term"], "tokens": item["tokens"], "num_pieces": item["num_pieces"]}
            for item in general_sorted[:12]
        ]
    }

    # 6. Model Compatibility Summary Record
    mean_seq_len = float(np.mean([p["subword_tokens"] for p in doc_profiles])) if doc_profiles else 0.0
    compat_summary = {
        "model": model_name,
        "clean_model_name": clean_model_filename(model_name),
        "domain_token_coverage": round(single_token_coverage, 4),
        "domain_fragmentation": round(maritime_frag_rate, 4),
        "domain_fertility": round(overall_fertility, 4),
        "domain_unknown_rate": round(oov_rate, 6) if oov_rate is not None else None,
        "oov_status": oov_status,
        "mean_sequence_length": round(mean_seq_len, 2),
        "tokenization_speed": round(tokenizer_speed, 2),
        "stratified_by_informativeness_tier": {
            tier: {
                "document_count": info["document_count"],
                "mean_fertility": info["mean_fertility"],
                "mean_sequence_length": info["mean_sequence_length"],
                "mean_document_fragmentation": info["mean_document_fragmentation"],
                "mean_document_coverage": info["mean_document_coverage"]
            }
            for tier, info in stratified_results.items()
        }
    }

    # Complete per-model report preserving ALL legacy keys + new analyses
    return {
        # Legacy preserved keys
        "model_name": model_name,
        "clean_model_name": clean_model_filename(model_name),
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer),
        "sampled_documents": len(doc_profiles),
        "total_raw_words_analyzed": total_raw_words,
        "total_subword_tokens_analyzed": total_subword_tokens,
        "average_subwords_per_word": overall_fertility,
        "maritime_fragmentation_rate": maritime_frag_rate,
        "single_token_vocabulary_coverage": single_token_coverage,
        "single_token_count": single_token_count,
        "total_maritime_terms": total_maritime_terms,
        "avg_pieces_per_term": avg_pieces,
        "median_pieces_per_term": median_pieces,
        "p95_pieces_per_term": p95_pieces,
        "max_pieces_per_term": max_pieces,
        "oov_rate": oov_rate,
        "oov_status": oov_status,
        "tokenizer_speed_tokens_per_sec": tokenizer_speed,
        "sequence_length_distribution": seq_length_dist,
        "worst_fragmented_terms": worst_fragmented_terms,
        "maritime_vocabulary_splits": maritime_splits[:50],

        # Extended Stage 12 Domain Informativeness Integration keys
        "vocabulary_categories": {
            cat_name: {
                "category": res["category"],
                "total_terms": res["total_terms"],
                "single_token_count": res["single_token_count"],
                "single_token_coverage": res["single_token_coverage"],
                "fragmentation_rate": res["fragmentation_rate"],
                "avg_pieces_per_term": res["avg_pieces_per_term"],
                "median_pieces_per_term": res["median_pieces_per_term"],
                "p95_pieces_per_term": res["p95_pieces_per_term"],
                "max_pieces_per_term": res["max_pieces_per_term"],
                "worst_fragmented_terms": res["worst_fragmented_terms"]
            }
            for cat_name, res in cat_results.items()
        },
        "informativeness_stratification": stratified_results,
        "document_correlations": correlations,
        "deterministic_term_examples": term_examples,
        "compatibility_summary": compat_summary,
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER
    }

def main():
    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")

    stage_dir = output_dir / "stage-13"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = stage_dir / "tokenizer_analysis"
    tok_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Vocabulary Categories
    # A. Maritime vocabulary from Stage 10
    vocab_path = output_dir / "stage-10" / "maritime_vocabulary.txt"
    maritime_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            maritime_terms = [line.strip() for line in fv if line.strip()]
    logger.info(f"Loaded {len(maritime_terms)} maritime vocabulary terms from {vocab_path}")

    # B. General vocabulary from Stage 12 General English baseline artifact
    gen_eng_path = output_dir / "stage-12" / "subsets" / "general_english_baseline.jsonl"
    general_terms = []
    if gen_eng_path.exists():
        words_set = set()
        with open(gen_eng_path, "r", encoding="utf-8") as fg:
            for line in fg:
                doc = json.loads(line).get("document", "")
                words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
                words_set.update(words)
        general_terms = sorted(list(words_set))
    else:
        # Fallback standard general terms if subset artifact is absent
        general_terms = [
            "library", "variety", "books", "digital", "media", "public", "access",
            "engineers", "solar", "panel", "system", "increase", "energy", "efficiency",
            "homes", "scientists", "extensive", "field", "study", "migratory", "birds"
        ]
    logger.info(f"Loaded {len(general_terms)} general vocabulary terms")

    # C. Rare maritime vocabulary (standardized canonical list)
    rare_terms = sorted(list(RARE_MARITIME_TERMS))
    logger.info(f"Loaded {len(rare_terms)} rare maritime terms")

    vocab_categories = {
        "maritime_terminology": maritime_terms,
        "general_vocabulary": general_terms,
        "rare_maritime_terminology": rare_terms
    }

    # 2. Ingest Stage 12 document_importance.jsonl (Authoritative Stage 12 Source)
    stage12_path = output_dir / "stage-12" / "document_importance.jsonl"
    if not stage12_path.exists():
        logger.error(f"Stage 12 document importance file not found at {stage12_path}! Run Stage 12 first.")
        return

    logger.info(f"Ingesting authoritative Stage 12 document informativeness records from {stage12_path}...")
    all_valid_records = []
    with open(stage12_path, "r", encoding="utf-8") as f12:
        for idx, line in enumerate(f12):
            rec = json.loads(line)
            doc_text = rec.get("document", "")
            if doc_text and doc_text.strip():
                all_valid_records.append({
                    "occurrence_id": rec.get("occurrence_id"),
                    "document": doc_text,
                    "domain_informativeness_score": float(rec.get("domain_informativeness_score", 0.0)),
                    "domain_relevance": float(rec.get("domain_relevance", 0.0)),
                    "information_content": float(rec.get("information_content", 0.0)),
                    "tfidf_representativeness": float(rec.get("tfidf_representativeness", 0.0)),
                    "redundancy_noise": float(rec.get("redundancy_noise", 0.0)),
                    "heuristic_informativeness_score": float(rec.get("heuristic_informativeness_score", 0.0)),
                    "tfidf_relevance": float(rec.get("tfidf_relevance", 0.0)),
                    "informativeness_rank": rec.get("informativeness_rank"),
                    "knowledge_tier": rec.get("knowledge_tier", "Unknown"),
                    "subdomain_labels": rec.get("subdomain_labels", []),
                    "redundancy_flag": rec.get("redundancy_flag", False)
                })

    total_available_docs = len(all_valid_records)
    target_sample_size = min(1500, total_available_docs)
    sampled_records = all_valid_records[:target_sample_size]

    tier_distribution = dict(Counter(r["knowledge_tier"] for r in sampled_records))
    logger.info(
        f"Joined {total_available_docs} valid Stage 12 records. "
        f"Sampled {len(sampled_records)} documents with observed tier distribution: {tier_distribution}"
    )

    # 3. Analyze Tokenizers across Canonical Models
    summary_reports = []
    for model_name in TARGET_MODELS:
        report = analyze_tokenizer(model_name, vocab_categories, sampled_records)
        if report:
            summary_reports.append(report)
            clean_name = report["clean_model_name"]
            with open(tok_dir / f"{clean_name}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

    if not summary_reports:
        logger.error("No tokenizers evaluated successfully!")
        return

    # 4. Copy BERT baseline to outputs/stage-13/tokenizer_analysis.json for backwards compatibility
    bert_report = next((r for r in summary_reports if r["model_name"] == "bert-base-uncased"), summary_reports[0])
    with open(stage_dir / "tokenizer_analysis.json", "w", encoding="utf-8") as f:
        json.dump(bert_report, f, indent=2)

    # 5. Generate Tokenizer Comparison CSV (preserving existing columns)
    csv_rows = []
    for r in summary_reports:
        oov_disp = round(r["oov_rate"] * 100, 4) if r["oov_rate"] is not None else "N/A"
        csv_rows.append({
            "model_name": r["model_name"],
            "vocab_size": r["vocab_size"],
            "subwords_per_word_fertility": round(r["average_subwords_per_word"], 4),
            "single_token_coverage_pct": round(r["single_token_vocabulary_coverage"] * 100, 2),
            "fragmentation_rate_pct": round(r["maritime_fragmentation_rate"] * 100, 2),
            "oov_rate_pct": oov_disp,
            "oov_status": r["oov_status"],
            "avg_pieces_per_term": round(r["avg_pieces_per_term"], 2),
            "median_pieces": r["median_pieces_per_term"],
            "p95_pieces": r["p95_pieces_per_term"],
            "max_pieces": r["max_pieces_per_term"],
            "tokenizer_speed_tok_sec": round(r["tokenizer_speed_tokens_per_sec"], 2)
        })

    df = pd.DataFrame(csv_rows)
    df.sort_values(by="single_token_coverage_pct", ascending=False, inplace=True)
    df.to_csv(tok_dir / "tokenizer_comparison.csv", index=False)

    # 6. Generate Dedicated outputs/stage-13/tokenizer_stage12_analysis.json
    stage12_analysis_artifact = {
        "metadata": {
            "title": "Stage 12 Domain Informativeness ↔ Stage 13 Tokenizer Compatibility Analysis",
            "source_stage_12_file": "outputs/stage-12/document_importance.jsonl",
            "data_source_description": (
                "Authoritative Stage 12 source of document informativeness scores, ranks, and tiers. "
                "Scores reflect Stage 12 heuristic/hybrid methodology outputs."
            ),
            "evaluated_models_count": len(summary_reports),
            "total_available_documents": total_available_docs,
            "sampled_documents_count": len(sampled_records),
            "observed_knowledge_tier_distribution": tier_distribution,
            "vocabulary_categories_analyzed": list(vocab_categories.keys()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "scientific_interpretation": {
            "core_question": (
                "How compatible is each pretrained tokenizer with specialized maritime vocabulary, "
                "and does tokenizer behavior vary across documents with different levels of domain informativeness?"
            ),
            "disclaimer": SCIENTIFIC_DISCLAIMER
        },
        "vocabulary_categories_summary": {
            cat_name: [
                {
                    "model": r["model_name"],
                    "single_token_coverage": r["vocabulary_categories"][cat_name]["single_token_coverage"],
                    "fragmentation_rate": r["vocabulary_categories"][cat_name]["fragmentation_rate"],
                    "avg_pieces_per_term": r["vocabulary_categories"][cat_name]["avg_pieces_per_term"],
                    "max_pieces_per_term": r["vocabulary_categories"][cat_name]["max_pieces_per_term"]
                }
                for r in summary_reports
            ]
            for cat_name in vocab_categories
        },
        "informativeness_stratification_summary": {
            tier: [
                {
                    "model": r["model_name"],
                    "document_count": r["informativeness_stratification"].get(tier, {}).get("document_count", 0),
                    "mean_fertility": r["informativeness_stratification"].get(tier, {}).get("mean_fertility"),
                    "mean_sequence_length": r["informativeness_stratification"].get(tier, {}).get("mean_sequence_length"),
                    "mean_document_fragmentation": r["informativeness_stratification"].get(tier, {}).get("mean_document_fragmentation"),
                    "mean_document_coverage": r["informativeness_stratification"].get(tier, {}).get("mean_document_coverage"),
                    "oov_rate": r["informativeness_stratification"].get(tier, {}).get("oov_rate"),
                    "oov_status": r["informativeness_stratification"].get(tier, {}).get("oov_status")
                }
                for r in summary_reports
                if tier in r["informativeness_stratification"]
            ]
            for tier in sorted(tier_distribution.keys())
        },
        "document_level_correlations": [
            {
                "model": r["model_name"],
                "fertility_correlation": r["document_correlations"]["informativeness_vs_fertility"],
                "fragmentation_correlation": r["document_correlations"]["informativeness_vs_fragmentation"],
                "coverage_correlation": r["document_correlations"]["informativeness_vs_coverage"]
            }
            for r in summary_reports
        ],
        "deterministic_term_examples": {
            r["clean_model_name"]: r["deterministic_term_examples"]
            for r in summary_reports
        },
        "model_compatibility_summaries": [
            r["compatibility_summary"] for r in summary_reports
        ]
    }

    stage12_analysis_path = stage_dir / "tokenizer_stage12_analysis.json"
    with open(stage12_analysis_path, "w", encoding="utf-8") as f:
        json.dump(stage12_analysis_artifact, f, indent=2)

    logger.info(
        f"Stage 13 successfully completed. "
        f"Evaluated {len(summary_reports)} tokenizers across {len(sampled_records)} joined Stage 12 documents. "
        f"Saved artifacts to {stage_dir} and {tok_dir}"
    )

if __name__ == "__main__":
    main()
