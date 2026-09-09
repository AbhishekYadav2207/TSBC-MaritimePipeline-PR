import os
import sys
import json
import math
import random
import time
import re
import hashlib
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pipeline_utils import setup_logging, load_config, get_project_root

logger = setup_logging("14_mlm_evaluation")

def stable_seed(*parts) -> int:
    """
    Computes a deterministic, platform-independent integer seed using SHA-256.
    Avoids Python's randomized hash() across processes for reproducible benchmarking.
    """
    key = "||".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000

# Default Representative Fallback Models (7 Distinct Tokenizer Archetypes)
FALLBACK_TARGET_MODELS = [
    "bert-base-uncased",                                            # Standard WordPiece (30,522)
    "dmis-lab/biobert-base-cased-v1.2",                            # Bio/Clinical WordPiece (28,996 Cased)
    "nlpaueb/legal-bert-base-uncased",                              # Legal WordPiece (30,522)
    "allenai/scibert_scivocab_uncased",                             # SciVocab WordPiece (31,090)
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",# PubMed Domain WordPiece (30,522)
    "roberta-base",                                                 # Standard Byte-Level BPE (50,265)
    "answerdotai/ModernBERT-base"                                   # Modern Extended BPE (50,280)
]

CATEGORIES = {
    "vessel_terminology": ["vess", "ship", "boat", "barge", "tug", "tanker", "trawler", "carrier", "hull", "deck", "keel", "tonnage", "transom", "freeboard", "gunwale", "bilge"],
    "navigation": ["navig", "gps", "ais", "vhf", "radar", "sonar", "compass", "gyro", "sounder", "chart", "vdr", "fathometer"],
    "machinery_propulsion": ["engine", "propel", "machinery", "motor", "shaft", "boiler", "fuel", "steering", "windlass", "hawser"],
    "casualty_incident": ["collision", "grounding", "stranding", "flooding", "leak", "capsiz", "sink", "injury", "death", "fatality", "missing", "damage"],
    "weather_environment": ["weather", "wind", "sea", "wave", "swell", "temp", "ice", "visibility", "fog", "clear", "windward", "leeward"],
    "safety_lifesaving": ["lifeboat", "liferaft", "lifejack", "lsa", "epirb", "sart", "buoy", "flare", "safety", "davit", "coxswain"]
}

RARE_MARITIME_TERMS = [
    "gyrocompass", "fathometer", "forepeak", "bulwark", "stempost", "windlass",
    "epirb", "sart", "hawser", "freeboard", "coxswain", "transom", "gunwale",
    "bilge", "fairlead", "windward", "leeward", "davit", "bitts", "bollard"
]

def load_selected_models(stage13_path: Path, fallback_models: list) -> list:
    """
    Dynamically loads authoritative models selected by Stage 13.
    Validates that exactly 7 models are provided to guarantee experiment integrity.
    Falls back to documented representative defaults if Stage 13 artifact is missing or invalid.
    """
    if stage13_path.exists():
        try:
            with open(stage13_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models = data.get("selected_models", [])
            if isinstance(models, list) and len(models) == 7:
                logger.info(f"Successfully loaded exactly {len(models)} authoritative models from Stage 13: {stage13_path}")
                return models
            else:
                found_cnt = len(models) if isinstance(models, list) else "invalid format"
                logger.warning(
                    f"Stage 13 artifact at {stage13_path} did not provide exactly 7 models (found {found_cnt}). "
                    f"Falling back to default 7 representative models."
                )
        except Exception as e:
            logger.warning(f"Failed to read Stage 13 artifact ({e}). Using fallback models.")
    else:
        logger.warning(f"Stage 13 artifact not found at {stage13_path}. Falling back to default {len(fallback_models)} models.")
    
    return list(fallback_models)

def clean_model_filename(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")

def get_term_category(term: str) -> str:
    term_lower = term.lower()
    for cat, stems in CATEGORIES.items():
        if any(stem in term_lower for stem in stems):
            return cat
    return "vessel_terminology"

def build_vocabulary_token_sets(tokenizer, vocab_terms: list):
    """
    Constructs token ID sets for maritime vocabulary, rare terms, and categories.
    Used for evaluation metrics tracking and fallback token matching.
    """
    maritime_token_ids = set()
    category_token_ids = {cat: set() for cat in CATEGORIES}
    rare_token_ids = set()

    for term in vocab_terms:
        sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(term))
        maritime_token_ids.update(sub_ids)
        cat = get_term_category(term)
        category_token_ids[cat].update(sub_ids)

    for r_term in RARE_MARITIME_TERMS:
        r_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r_term))
        rare_token_ids.update(r_ids)
        maritime_token_ids.update(r_ids)
        category_token_ids["navigation"].update(r_ids)

    return maritime_token_ids, rare_token_ids, category_token_ids

def classify_token_positions(raw_text: str, input_ids: list, offsets: list, special_tokens_mask: list,
                              vocab_terms_set: set, rare_terms_set: set,
                              maritime_token_ids: set, rare_token_ids: set,
                              category_token_ids: dict = None):
    """
    Classifies token positions into rare maritime, other maritime, and general.
    Uses tokenizer offset mappings for robust subword-span matching when available.
    Also maps token positions to maritime categories.
    Falls back safely to vocabulary token IDs when offsets are unavailable.
    """
    seq_len = len(input_ids)
    eligible_positions = []
    for i in range(seq_len):
        if special_tokens_mask[i]:
            continue
        if offsets is not None and offsets[i] == [0, 0] and i != 0:
            # Padding token
            continue
        eligible_positions.append(i)

    rare_positions = set()
    maritime_positions = set()
    category_positions = {cat: set() for cat in CATEGORIES}

    if offsets is not None and len(raw_text) > 0:
        # Robust Span Matching using Offset Mapping (prevents subword false-positives)
        for term in rare_terms_set:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", raw_text, re.IGNORECASE):
                c_start, c_end = m.span()
                for idx in eligible_positions:
                    t_start, t_end = offsets[idx]
                    if t_start < c_end and t_end > c_start:
                        rare_positions.add(idx)
                        category_positions["navigation"].add(idx)

        for term in vocab_terms_set:
            cat = get_term_category(term)
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", raw_text, re.IGNORECASE):
                c_start, c_end = m.span()
                for idx in eligible_positions:
                    t_start, t_end = offsets[idx]
                    if t_start < c_end and t_end > c_start:
                        if idx not in rare_positions:
                            maritime_positions.add(idx)
                        category_positions[cat].add(idx)
    else:
        # Safe Fallback: Token ID membership
        for idx in eligible_positions:
            t_id = input_ids[idx]
            if t_id in rare_token_ids:
                rare_positions.add(idx)
                category_positions["navigation"].add(idx)
            elif t_id in maritime_token_ids:
                maritime_positions.add(idx)
                if category_token_ids:
                    for cat, cat_ids in category_token_ids.items():
                        if t_id in cat_ids:
                            category_positions[cat].add(idx)

    general_positions = [p for p in eligible_positions if p not in rare_positions and p not in maritime_positions]

    return eligible_positions, rare_positions, maritime_positions, general_positions, category_positions

def create_random_mask(eligible_positions: list, rng: random.Random, mask_budget: int):
    """Conventional Random-15% masking baseline."""
    if not eligible_positions or mask_budget <= 0:
        return []
    if len(eligible_positions) <= mask_budget:
        return list(eligible_positions)
    return rng.sample(eligible_positions, mask_budget)

def create_domain_aware_mask(eligible_positions: list, rare_positions: set, maritime_positions: set,
                             general_positions: list, rng: random.Random, mask_budget: int):
    """
    Lightweight domain-aware selective masking policy.
    Total budget is strictly maintained at approximately 15% of eligible tokens.
    Priority:
      1. Rare maritime tokens (highest)
      2. Domain maritime vocabulary tokens (next)
      3. General tokens (random fill to meet total budget)

    Important Scientific Note:
      Because priority is given to rare and maritime domain tokens, documents that are
      densely packed with maritime terminology may fill the entire 15% budget with domain
      tokens alone, leaving general_mask_fraction = 0.0. This is an expected and intentional
      property of task-guided selective masking (Train No Evil principle). The masking
      diagnostics record rare_maritime_mask_fraction, maritime_mask_fraction, and
      general_mask_fraction so this selective distribution is fully auditable.
    """
    if not eligible_positions or mask_budget <= 0:
        return []

    selected = []

    # Priority 1: Rare maritime tokens
    rare_list = list(rare_positions)
    if len(rare_list) <= mask_budget:
        selected.extend(rare_list)
    else:
        selected.extend(rng.sample(rare_list, mask_budget))

    # Priority 2: Maritime vocabulary tokens
    rem_budget = mask_budget - len(selected)
    if rem_budget > 0:
        mar_list = list(maritime_positions)
        if len(mar_list) <= rem_budget:
            selected.extend(mar_list)
        else:
            selected.extend(rng.sample(mar_list, rem_budget))

    # Priority 3: General tokens random fill
    rem_budget = mask_budget - len(selected)
    if rem_budget > 0:
        if len(general_positions) <= rem_budget:
            selected.extend(general_positions)
        else:
            selected.extend(rng.sample(general_positions, rem_budget))

    return selected

def evaluate_model_on_docs(model, tokenizer, docs: list, vocab_terms: list, device: torch.device,
                           masking_strategy: str = "random_15", seed: int = 42,
                           max_docs: int = 200, max_length: int = 256, batch_size: int = 16) -> dict:
    """
    Evaluates a pretrained MLM model on a collection of documents under either
    'random_15' (control baseline) or 'domain_aware_15' (focused experiment).
    Uses position-level span classification for evaluation metrics to prevent subword false-positives.
    """
    if not docs:
        return {}

    rng = random.Random(seed)
    torch_rng = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    torch_rng.manual_seed(seed)

    maritime_token_ids, rare_token_ids, category_token_ids = build_vocabulary_token_sets(tokenizer, vocab_terms)
    vocab_terms_set = set(vocab_terms)
    rare_terms_set = set(RARE_MARITIME_TERMS)

    general_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    maritime_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    rare_stats = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    cat_stats = {cat: {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0} for cat in CATEGORIES}

    # Diagnostics for domain-aware masking audit
    diag_eligible = 0
    diag_masked = 0
    diag_rare_masked = 0
    diag_maritime_masked = 0
    diag_general_masked = 0

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    eval_docs = docs[:max_docs]

    t_start = time.time()

    # Check if fast tokenizer offset mapping is supported
    supports_offsets = getattr(tokenizer, "is_fast", False)

    with torch.no_grad():
        for i in range(0, len(eval_docs), batch_size):
            batch_texts = eval_docs[i:i+batch_size]

            encoding_kwargs = {
                "padding": True,
                "truncation": True,
                "max_length": max_length,
                "return_tensors": "pt"
            }
            if supports_offsets:
                encoding_kwargs["return_offsets_mapping"] = True

            try:
                inputs = tokenizer(batch_texts, **encoding_kwargs)
            except Exception as e:
                # Fallback without offset mapping if fast tokenizer raised error
                encoding_kwargs.pop("return_offsets_mapping", None)
                inputs = tokenizer(batch_texts, **encoding_kwargs)
                supports_offsets = False

            offsets_tensor = inputs.pop("offset_mapping", None)

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            labels = input_ids.clone()
            masked_input_ids = input_ids.clone()
            masked_indices = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

            batch_size_actual = input_ids.shape[0]
            batch_rare_pos = []
            batch_mar_pos = []
            batch_cat_pos = []

            for b in range(batch_size_actual):
                seq_ids = input_ids[b].cpu().tolist()
                raw_text = batch_texts[b]
                offsets = offsets_tensor[b].cpu().tolist() if offsets_tensor is not None else None
                sp_mask = tokenizer.get_special_tokens_mask(seq_ids, already_has_special_tokens=True)

                eligible_positions, rare_pos, mar_pos, gen_pos, cat_pos = classify_token_positions(
                    raw_text, seq_ids, offsets, sp_mask,
                    vocab_terms_set, rare_terms_set,
                    maritime_token_ids, rare_token_ids, category_token_ids
                )
                batch_rare_pos.append(rare_pos)
                batch_mar_pos.append(mar_pos)
                batch_cat_pos.append(cat_pos)

                n_eligible = len(eligible_positions)
                diag_eligible += n_eligible
                budget = max(1, round(0.15 * n_eligible)) if n_eligible > 0 else 0

                if masking_strategy == "domain_aware_15":
                    selected = create_domain_aware_mask(eligible_positions, rare_pos, mar_pos, gen_pos, rng, budget)
                else:
                    selected = create_random_mask(eligible_positions, rng, budget)

                for pos in selected:
                    masked_indices[b, pos] = True
                    masked_input_ids[b, pos] = mask_token_id

                    # Record diagnostics based on position-level classification
                    diag_masked += 1
                    if pos in rare_pos:
                        diag_rare_masked += 1
                    elif pos in mar_pos:
                        diag_maritime_masked += 1
                    else:
                        diag_general_masked += 1

            labels[~masked_indices] = -100

            try:
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            except Exception as e:
                logger.warning(f"Error during model forward pass: {e}")
                continue

            for b in range(batch_size_actual):
                mask_positions = torch.where(masked_indices[b])[0]
                rare_pos_set = batch_rare_pos[b]
                mar_pos_set = batch_mar_pos[b]
                cat_pos_dict = batch_cat_pos[b]

                for pos_tensor in mask_positions:
                    pos = pos_tensor.item()
                    target_id = labels[b, pos].item()
                    token_logits = logits[b, pos]

                    token_loss = criterion(token_logits.unsqueeze(0), torch.tensor([target_id], device=device)).item()
                    top_k_indices = torch.topk(token_logits, 10).indices.tolist()

                    is_top1 = 1 if target_id == top_k_indices[0] else 0
                    is_top5 = 1 if target_id in top_k_indices[:5] else 0
                    is_top10 = 1 if target_id in top_k_indices[:10] else 0

                    # Position-level classification avoids subword false-positives
                    is_rare = pos in rare_pos_set
                    is_maritime = is_rare or (pos in mar_pos_set)

                    if is_rare:
                        rare_stats["loss"] += token_loss
                        rare_stats["top1"] += is_top1
                        rare_stats["top5"] += is_top5
                        rare_stats["top10"] += is_top10
                        rare_stats["count"] += 1

                    if is_maritime:
                        maritime_stats["loss"] += token_loss
                        maritime_stats["top1"] += is_top1
                        maritime_stats["top5"] += is_top5
                        maritime_stats["top10"] += is_top10
                        maritime_stats["count"] += 1

                        for cat, cat_pos_indices in cat_pos_dict.items():
                            if pos in cat_pos_indices:
                                cat_stats[cat]["loss"] += token_loss
                                cat_stats[cat]["top1"] += is_top1
                                cat_stats[cat]["top5"] += is_top5
                                cat_stats[cat]["top10"] += is_top10
                                cat_stats[cat]["count"] += 1
                    else:
                        general_stats["loss"] += token_loss
                        general_stats["top1"] += is_top1
                        general_stats["top5"] += is_top5
                        general_stats["top10"] += is_top10
                        general_stats["count"] += 1

    eval_time = time.time() - t_start

    def summarize(st):
        cnt = max(st["count"], 1)
        avg_loss = st["loss"] / cnt
        loss_exp = math.exp(avg_loss) if avg_loss < 20 else 99999.0
        return {
            "masked_sample_count": st["count"],
            "mlm_loss": float(avg_loss),
            "mlm_loss_derived_exponential": float(loss_exp),
            "top1_accuracy": float(st["top1"] / cnt),
            "top5_accuracy": float(st["top5"] / cnt),
            "top10_accuracy": float(st["top10"] / cnt)
        }

    gen_summary = summarize(general_stats)
    mar_summary = summarize(maritime_stats)
    rare_summary = summarize(rare_stats)
    cat_summaries = {cat: summarize(st) for cat, st in cat_stats.items()}

    performance_gap = gen_summary["top1_accuracy"] - mar_summary["top1_accuracy"]

    # Overall masked accuracy (combining general + maritime)
    tot_eval_count = gen_stats_cnt = general_stats["count"] + maritime_stats["count"]
    if tot_eval_count > 0:
        overall_top1 = (general_stats["top1"] + maritime_stats["top1"]) / tot_eval_count
        overall_loss = (general_stats["loss"] + maritime_stats["loss"]) / tot_eval_count
    else:
        overall_top1 = gen_summary["top1_accuracy"]
        overall_loss = gen_summary["mlm_loss"]

    actual_mask_rate = float(diag_masked / diag_eligible) if diag_eligible > 0 else 0.0
    safe_masked = max(diag_masked, 1)

    result = {
        "evaluated_documents": len(eval_docs),
        "evaluation_time_sec": float(eval_time),
        "masking_strategy": masking_strategy,
        "overall_summary": {
            "total_masked_tokens": diag_masked,
            "overall_top1_accuracy": float(overall_top1),
            "overall_mlm_loss": float(overall_loss)
        },
        "general_tokens_summary": gen_summary,
        "maritime_tokens_summary": mar_summary,
        "rare_maritime_tokens_summary": rare_summary,
        "category_recall": {cat: round(st["top1_accuracy"], 4) for cat, st in cat_summaries.items()},
        "category_breakdown": cat_summaries,
        "performance_gap_top1": float(performance_gap)
    }

    if masking_strategy == "domain_aware_15":
        result["masking_diagnostics"] = {
            "total_eligible_tokens": diag_eligible,
            "total_masked_tokens": diag_masked,
            "actual_mask_rate": round(actual_mask_rate, 4),
            "rare_maritime_tokens_masked": diag_rare_masked,
            "maritime_tokens_masked": diag_maritime_masked,
            "general_tokens_masked": diag_general_masked,
            "rare_maritime_mask_fraction": round(float(diag_rare_masked / safe_masked), 4),
            "maritime_mask_fraction": round(float(diag_maritime_masked / safe_masked), 4),
            "general_mask_fraction": round(float(diag_general_masked / safe_masked), 4)
        }

    return result

def evaluate_sampled_pll(model, tokenizer, docs: list, vocab_terms: list, device: torch.device,
                         max_docs: int = 10, max_seq_len: int = 128, max_positions_per_doc: int = 32,
                         seed: int = 42) -> dict:
    """
    Computes Sampled Pseudo-Log-Likelihood (Salazar et al., ACL 2020) over a bounded,
    deterministic sample of documents and target positions.
    Evaluates one-token-at-a-time MLM scoring to compute likelihood and pseudo-perplexity.
    Uses position-level span classification to distinguish domain vs general tokens.
    """
    if not docs:
        return {}

    rng = random.Random(seed)
    maritime_token_ids, rare_token_ids, _ = build_vocabulary_token_sets(tokenizer, vocab_terms)
    vocab_terms_set = set(vocab_terms)
    rare_terms_set = set(RARE_MARITIME_TERMS)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    eval_docs = docs[:max_docs]
    t_start = time.time()

    all_token_log_probs = []
    maritime_log_probs = []
    general_log_probs = []
    doc_pll_scores = []

    supports_offsets = getattr(tokenizer, "is_fast", False)

    with torch.no_grad():
        for doc_text in eval_docs:
            encoding_kwargs = {
                "truncation": True,
                "max_length": max_seq_len,
                "return_tensors": "pt"
            }
            if supports_offsets:
                encoding_kwargs["return_offsets_mapping"] = True

            try:
                encoded = tokenizer(doc_text, **encoding_kwargs)
            except Exception:
                encoding_kwargs.pop("return_offsets_mapping", None)
                encoded = tokenizer(doc_text, **encoding_kwargs)

            offsets_tensor = encoded.pop("offset_mapping", None)
            offsets_list = offsets_tensor[0].cpu().tolist() if offsets_tensor is not None else None

            seq = encoded["input_ids"][0]
            seq_len = len(seq)
            if seq_len <= 2:
                continue

            sp_mask = tokenizer.get_special_tokens_mask(seq.tolist(), already_has_special_tokens=True)
            eligible, rare_pos, mar_pos, gen_pos, _ = classify_token_positions(
                doc_text, seq.tolist(), offsets_list, sp_mask,
                vocab_terms_set, rare_terms_set,
                maritime_token_ids, rare_token_ids
            )

            if not eligible:
                continue

            # Deterministic bounded selection of target positions
            if len(eligible) <= max_positions_per_doc:
                target_positions = list(eligible)
            else:
                target_positions = sorted(rng.sample(eligible, max_positions_per_doc))

            # Batch single-token masked sequences for computational efficiency
            batch_inputs = []
            target_ids = []
            for pos in target_positions:
                inp = seq.clone()
                inp[pos] = mask_token_id
                batch_inputs.append(inp)
                target_ids.append(seq[pos].item())

            # Forward pass in chunks to avoid GPU/CPU memory spikes
            chunk_size = 16
            doc_lps = []
            for c in range(0, len(batch_inputs), chunk_size):
                chunk_tensors = torch.stack(batch_inputs[c:c+chunk_size]).to(device)
                chunk_attn = torch.ones_like(chunk_tensors).to(device)
                outputs = model(input_ids=chunk_tensors, attention_mask=chunk_attn)
                chunk_logits = outputs.logits # [B, L, V]
                chunk_log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)

                for b_idx in range(chunk_tensors.shape[0]):
                    pos_in_seq = target_positions[c + b_idx]
                    target_token = target_ids[c + b_idx]
                    lp = chunk_log_probs[b_idx, pos_in_seq, target_token].item()
                    doc_lps.append(lp)
                    all_token_log_probs.append(lp)

                    # Position-level domain classification via span matching
                    is_domain_token = (pos_in_seq in rare_pos) or (pos_in_seq in mar_pos)
                    if is_domain_token:
                        maritime_log_probs.append(lp)
                    else:
                        general_log_probs.append(lp)

            if doc_lps:
                doc_pll_scores.append(sum(doc_lps))

    eval_time = time.time() - t_start

    tot_tokens = len(all_token_log_probs)
    doc_pll_sum = float(sum(doc_pll_scores)) if doc_pll_scores else 0.0
    mean_token_pll = float(sum(all_token_log_probs) / tot_tokens) if tot_tokens > 0 else 0.0
    pseudo_ppl = float(math.exp(-mean_token_pll)) if tot_tokens > 0 and -mean_token_pll < 20 else 99999.0

    mar_cnt = len(maritime_log_probs)
    mar_mean = float(sum(maritime_log_probs) / mar_cnt) if mar_cnt > 0 else 0.0

    gen_cnt = len(general_log_probs)
    gen_mean = float(sum(general_log_probs) / gen_cnt) if gen_cnt > 0 else 0.0

    return {
        "metric_name": "Sampled Pseudo-Log-Likelihood (Sampled PLL)",
        "document_pll": doc_pll_sum,
        "mean_token_pll": mean_token_pll,
        "pseudo_perplexity": pseudo_ppl,
        "domain_token_pll": {
            "sum": float(sum(maritime_log_probs)),
            "mean": mar_mean,
            "count": mar_cnt
        },
        "general_token_pll": {
            "sum": float(sum(general_log_probs)),
            "mean": gen_mean,
            "count": gen_cnt
        },
        "number_of_documents": len(eval_docs),
        "number_of_scored_tokens": tot_tokens,
        "sampling_seed": seed,
        "max_length": max_seq_len,
        "max_positions_per_document": max_positions_per_doc,
        "evaluation_time_sec": float(eval_time)
    }

def screen_and_select_configurations(random_15_records: list, num_select: int = 4) -> tuple:
    """
    Analyzes the 25 representation x subset cells from the 175 Random-15 results.
    Ranks cells by mean_top1 (descending) and mean_mlm_loss (ascending).
    Selects exactly 4 diverse configurations balancing strong performance,
    representation diversity, and subset diversity.
    """
    from collections import defaultdict

    cells = defaultdict(lambda: {"top1": [], "loss": [], "mar_top1": [], "rare_top1": []})
    for d in random_15_records:
        rep = d["representation"]
        sub = d["subset"]
        m = d["evaluation_metrics"]
        overall = m.get("overall_summary", {})
        gen = m.get("general_tokens_summary", {})
        mar = m.get("maritime_tokens_summary", {})
        rare = m.get("rare_maritime_tokens_summary", {})

        top1 = overall.get("overall_top1_accuracy", gen.get("top1_accuracy", 0.0))
        loss = overall.get("overall_mlm_loss", gen.get("mlm_loss", 0.0))

        cells[(rep, sub)]["top1"].append(top1)
        cells[(rep, sub)]["loss"].append(loss)
        cells[(rep, sub)]["mar_top1"].append(mar.get("top1_accuracy", 0.0))
        cells[(rep, sub)]["rare_top1"].append(rare.get("top1_accuracy", 0.0))

    ranked_cells = []
    for (rep, sub), v in cells.items():
        n = len(v["top1"])
        mean_top1 = sum(v["top1"]) / n if n > 0 else 0.0
        mean_loss = sum(v["loss"]) / n if n > 0 else 0.0
        mean_mar_top1 = sum(v["mar_top1"]) / n if n > 0 else 0.0
        mean_rare_top1 = sum(v["rare_top1"]) / n if n > 0 else 0.0

        ranked_cells.append({
            "representation": rep,
            "subset": sub,
            "mean_top1": round(mean_top1, 4),
            "mean_mlm_loss": round(mean_loss, 4),
            "mean_maritime_top1": round(mean_mar_top1, 4),
            "mean_rare_maritime_accuracy": round(mean_rare_top1, 4)
        })

    # Primary: mean_top1 descending, Secondary: mean_mlm_loss ascending
    ranked_cells.sort(key=lambda x: (-x["mean_top1"], x["mean_mlm_loss"]))
    for idx, c in enumerate(ranked_cells, 1):
        c["rank"] = idx

    # Diversity-Preserving Selection of exactly 4 cells
    selected = []
    used_reps = set()
    used_subs = set()

    # Pass 1: Select top cells with both distinct representation AND distinct subset
    for c in ranked_cells:
        if len(selected) == num_select:
            break
        if c["representation"] not in used_reps and c["subset"] not in used_subs:
            c_copy = dict(c)
            c_copy["selection_rationale"] = (
                f"Rank {c['rank']} overall in Random-15 screening. Demonstrates top MLM performance with "
                f"distinct representation '{c['representation']}' and distinct subset '{c['subset']}'."
            )
            c_copy["selection_method"] = "Deterministic multi-attribute diversity optimization (v2.1)"
            selected.append(c_copy)
            used_reps.add(c["representation"])
            used_subs.add(c["subset"])

    # Pass 2: Select with distinct representation if needed
    if len(selected) < num_select:
        for c in ranked_cells:
            if len(selected) == num_select:
                break
            if c["representation"] not in used_reps:
                c_copy = dict(c)
                c_copy["selection_rationale"] = (
                    f"Rank {c['rank']} overall in Random-15 screening. Top performing candidate for distinct representation '{c['representation']}'."
                )
                c_copy["selection_method"] = "Deterministic multi-attribute diversity optimization (v2.1)"
                selected.append(c_copy)
                used_reps.add(c["representation"])

    # Pass 3: Fill remaining slots with highest remaining
    if len(selected) < num_select:
        selected_keys = {(s["representation"], s["subset"]) for s in selected}
        for c in ranked_cells:
            if len(selected) == num_select:
                break
            if (c["representation"], c["subset"]) not in selected_keys:
                c_copy = dict(c)
                c_copy["selection_rationale"] = f"Rank {c['rank']} overall in Random-15 screening. Next highest performing cell."
                c_copy["selection_method"] = "Deterministic multi-attribute diversity optimization (v2.1)"
                selected.append(c_copy)
                selected_keys.add((c["representation"], c["subset"]))

    return ranked_cells, selected

def run_smoke_test(stage13_path: Path, output_dir: Path):
    """
    Lightweight smoke test for Stage 14:
    Verifies model loading, data loading, random masking, domain-aware masking,
    MLM forward pass, sampled PLL, and selection logic on a minimal sample.
    """
    logger.info("=== Starting Stage 14 Lightweight Smoke Test ===")
    models = load_selected_models(stage13_path, FALLBACK_TARGET_MODELS)
    test_model_name = models[0]
    logger.info(f"Smoke test using model: {test_model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Smoke test device: {device}")

    # 1. Test model and tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(test_model_name)
    model = AutoModelForMaskedLM.from_pretrained(test_model_name)
    model.to(device)
    model.eval()
    logger.info("[OK] Model and tokenizer loaded successfully.")

    # 2. Test data loading
    vocab_path = output_dir / "stage-10" / "maritime_vocabulary.txt"
    vocab_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            vocab_terms = [l.strip() for l in fv if l.strip()]
    logger.info(f"[OK] Loaded {len(vocab_terms)} vocabulary terms.")

    reps_dir = output_dir / "stage-11" / "corpus_representations"
    rep_path = reps_dir / "narrative.jsonl"
    sample_docs = []
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            for line in f:
                sample_docs.append(json.loads(line)["document"])
                if len(sample_docs) >= 2:
                    break
    # Include a test sentence containing rare terms to verify span matching under smoke testing
    sample_docs.append(
        "The vessel used gyrocompass and fathometer navigation while EPIRB and freeboard were inspected by the coxswain at the windlass."
    )
    logger.info(f"[OK] Loaded {len(sample_docs)} sample documents for smoke testing.")

    # 3. Test Random-15 evaluation
    eval_rand = evaluate_model_on_docs(
        model, tokenizer, sample_docs, vocab_terms, device,
        masking_strategy="random_15", max_docs=len(sample_docs), max_length=64, batch_size=2
    )
    assert "overall_summary" in eval_rand, "Random-15 summary missing"
    logger.info(f"[OK] Random-15 evaluation passed: Top1={eval_rand['overall_summary']['overall_top1_accuracy']:.4f}, Loss={eval_rand['overall_summary']['overall_mlm_loss']:.4f}")

    # 4. Test Domain-Aware-15 evaluation
    eval_domain = evaluate_model_on_docs(
        model, tokenizer, sample_docs, vocab_terms, device,
        masking_strategy="domain_aware_15", max_docs=len(sample_docs), max_length=64, batch_size=2
    )
    assert "masking_diagnostics" in eval_domain, "Domain-Aware diagnostics missing"
    diag = eval_domain["masking_diagnostics"]
    logger.info(f"[OK] Domain-Aware-15 evaluation passed: MaskRate={diag['actual_mask_rate']:.4f}, RareMasked={diag['rare_maritime_tokens_masked']}, MaritimeMasked={diag['maritime_tokens_masked']}")

    # 5. Test Sampled PLL
    eval_pll = evaluate_sampled_pll(
        model, tokenizer, sample_docs, vocab_terms, device,
        max_docs=len(sample_docs), max_seq_len=64, max_positions_per_doc=8, seed=42
    )
    assert eval_pll["number_of_scored_tokens"] > 0, "PLL scored 0 tokens"
    assert not math.isnan(eval_pll["mean_token_pll"]), "PLL returned NaN"
    logger.info(f"[OK] Sampled PLL passed: ScoredTokens={eval_pll['number_of_scored_tokens']}, MeanPLL={eval_pll['mean_token_pll']:.4f}, PseudoPPL={eval_pll['pseudo_perplexity']:.4f}")

    # 6. Test Selection Logic on synthetic records
    mock_records = []
    reps = ["narrative", "key_value", "template", "json", "mixed"]
    subs = ["high_knowledge", "medium_knowledge", "low_knowledge", "balanced_knowledge", "random_baseline"]
    for r_idx, r in enumerate(reps):
        for s_idx, s in enumerate(subs):
            mock_records.append({
                "representation": r,
                "subset": s,
                "evaluation_metrics": {
                    "overall_summary": {
                        "overall_top1_accuracy": 0.3 + (r_idx * 0.08) + (s_idx * 0.02),
                        "overall_mlm_loss": 4.0 - (r_idx * 0.2)
                    },
                    "maritime_tokens_summary": {"top1_accuracy": 0.2 + (r_idx * 0.05)},
                    "rare_maritime_tokens_summary": {"top1_accuracy": 0.1 + (s_idx * 0.03)}
                }
            })
    ranked, selected = screen_and_select_configurations(mock_records, num_select=4)
    assert len(selected) == 4, f"Expected 4 selected cells, got {len(selected)}"
    assert len(ranked) == 25, f"Expected 25 ranked cells, got {len(ranked)}"
    logger.info(f"[OK] Selection logic passed: selected 4 diverse configurations from {len(ranked)} cells.")

    logger.info("=== [SUCCESS] All Stage 14 Smoke Tests Completed Successfully! ===")
    return True

def main():
    parser = argparse.ArgumentParser(description="Stage 14: Masked Language Model Benchmarking & Analysis")
    parser.add_argument("--fresh", action="store_true", help="Force fresh recomputation of all 175 Random-15 evaluations, ignoring old cache.")
    parser.add_argument("--smoke-test", action="store_true", help="Run lightweight smoke test on a minimal sample without executing full benchmark.")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config()
    output_dir = root / config.get("output_dir", "outputs")

    stage_dir = output_dir / "stage-14"
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage13_path = output_dir / "stage-13" / "selected_models.json"

    if args.smoke_test:
        run_smoke_test(stage13_path, output_dir)
        return

    # Load Authoritative Models from Stage 13
    target_models = load_selected_models(stage13_path, FALLBACK_TARGET_MODELS)

    vocab_path = output_dir / "stage-10" / "maritime_vocabulary.txt"
    vocab_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            vocab_terms = [line.strip() for line in fv if line.strip()]

    # Matrix Dimensions: 5 representations x 5 subsets
    representations = ["narrative", "key_value", "template", "json", "mixed"]
    subsets = ["high_knowledge", "medium_knowledge", "low_knowledge", "balanced_knowledge", "random_baseline"]

    reps_dir = output_dir / "stage-11" / "corpus_representations"
    subsets_dir = output_dir / "stage-12" / "subsets"

    # Load General English Baseline Subset
    gen_eng_docs = []
    gen_eng_path = subsets_dir / "general_english_baseline.jsonl"
    if gen_eng_path.exists():
        with open(gen_eng_path, "r", encoding="utf-8") as f:
            gen_eng_docs = [json.loads(l)["document"] for l in f]

    eval_out_dir = stage_dir / "evaluations"
    cache_random_dir = eval_out_dir / "cache"
    cache_domain_dir = cache_random_dir / "domain_aware_15"
    cache_pll_dir = cache_random_dir / "pll"

    cache_random_dir.mkdir(parents=True, exist_ok=True)
    cache_domain_dir.mkdir(parents=True, exist_ok=True)
    cache_pll_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    total_random_runs = len(target_models) * len(representations) * len(subsets)
    logger.info(f"Broad Random-15 evaluations: {total_random_runs} runs ({len(target_models)} models x {len(representations)} reps x {len(subsets)} subsets)")

    if args.fresh:
        logger.info("Executing fresh benchmark: existing Random-15 cache will be recomputed.")
    else:
        logger.info("Preserving existing Random-15 cache entries where available.")

    # =========================================================================
    # PHASE 1: Broad Random-15 Evaluation (175 runs)
    # =========================================================================
    random_15_records = []
    run_count = 0

    for model_name in target_models:
        clean_model = clean_model_filename(model_name)
        logger.info(f"Loading model & tokenizer: {model_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model.to(device)
            model.eval()
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model '{model_name}': {e}. Skipping model.")
            continue

        # Evaluate General English Baseline once for Domain Shift calculation
        gen_eng_eval = evaluate_model_on_docs(
            model, tokenizer, gen_eng_docs, vocab_terms, device,
            masking_strategy="random_15", seed=42, max_docs=200, max_length=256, batch_size=16
        )
        gen_eng_top1 = gen_eng_eval.get("general_tokens_summary", {}).get("top1_accuracy", 0.85)

        eval_record = None
        for rep in representations:
            rep_path = reps_dir / f"{rep}.jsonl"
            if not rep_path.exists():
                continue
            with open(rep_path, "r", encoding="utf-8") as f:
                rep_records = [json.loads(l) for l in f]

            for sub in subsets:
                cache_key = f"{clean_model}__{rep}__{sub}.json"
                cache_path = cache_random_dir / cache_key

                # Check cache if not running fresh
                if not args.fresh and cache_path.exists():
                    run_count += 1
                    with open(cache_path, "r", encoding="utf-8") as f_c:
                        eval_record = json.load(f_c)
                    random_15_records.append(eval_record)
                    continue

                sub_path = subsets_dir / f"{sub}.jsonl"
                sub_occ_ids = set()
                if sub_path.exists():
                    with open(sub_path, "r", encoding="utf-8") as f:
                        sub_occ_ids = {json.loads(l)["occurrence_id"] for l in f}

                if not sub_occ_ids:
                    logger.warning(f"Subset '{sub}' contains 0 occurrence IDs. Skipping.")
                    continue

                target_docs = [
                    rec["document"]
                    for rec in rep_records
                    if rec.get("occurrence_id") in sub_occ_ids
                ][:200]

                if not target_docs:
                    logger.warning(f"Representation '{rep}' has 0 matching documents for subset '{sub}'. Skipping.")
                    continue

                # Deterministic reproducible seed per configuration
                cell_seed = stable_seed(model_name, rep, sub, "random_15")

                logger.info(f"[{run_count + 1}/{total_random_runs}] Evaluating {clean_model} | Rep: {rep} | Subset: {sub} | Docs: {len(target_docs)}")

                eval_res = evaluate_model_on_docs(
                    model, tokenizer, target_docs, vocab_terms, device,
                    masking_strategy="random_15", seed=cell_seed, max_docs=200, max_length=256, batch_size=16
                )

                maritime_top1 = eval_res.get("maritime_tokens_summary", {}).get("top1_accuracy", 0.0)
                domain_shift_gap = float(gen_eng_top1 - maritime_top1)

                eval_record = {
                    "model_name": model_name,
                    "clean_model_name": clean_model,
                    "representation": rep,
                    "subset": sub,
                    "evaluated_doc_count": len(target_docs),
                    "general_english_baseline_top1": float(gen_eng_top1),
                    "domain_shift_gap": domain_shift_gap,
                    "experiment_metadata": {
                        "masking_strategy": "random_15",
                        "mask_rate": 0.15,
                        "max_length": 256,
                        "evaluation_documents": len(target_docs),
                        "seed": cell_seed
                    },
                    "evaluation_metrics": eval_res
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(eval_record, f, indent=2)

                run_count += 1
                random_15_records.append(eval_record)
                logger.info(f"[{run_count}/{total_random_runs}] Completed: {clean_model} | Rep: {rep} | Subset: {sub} | Top1: {maritime_top1:.4f}")

        if eval_record is not None:
            with open(eval_out_dir / f"{clean_model}.json", "w", encoding="utf-8") as f:
                json.dump(eval_record, f, indent=2)

    # Copy BERT baseline to bert_mlm_evaluation.json for backward compatibility
    bert_clean = clean_model_filename("bert-base-uncased")
    bert_cache = list(cache_random_dir.glob(f"{bert_clean}__*.json"))
    if bert_cache:
        with open(bert_cache[0], "r", encoding="utf-8") as f_in, open(stage_dir / "bert_mlm_evaluation.json", "w", encoding="utf-8") as f_out:
            json.dump(json.load(f_in), f_out, indent=2)

    logger.info(f"Phase 1 complete: {len(random_15_records)} Random-15 evaluations available.")

    # =========================================================================
    # PHASE 2: Screen 25 Cells & Select Exactly 4 Diverse Configurations
    # =========================================================================
    logger.info("Phase 2: Screening 25 cells across 7 models to select exactly 4 strong, diverse configurations...")
    all_ranked_cells, selected_cells = screen_and_select_configurations(random_15_records, num_select=4)

    selection_artifact = {
        "selection_method": "Multi-attribute diversity optimization (v2.1): Primary ranking by mean Top-1 (descending) & mean MLM loss (ascending)",
        "screening_basis": "Stage 14 Random-15 MLM benchmark results aggregated across all 7 models",
        "total_cells_evaluated": len(all_ranked_cells),
        "selected_cell_count": len(selected_cells),
        "selected_configurations": selected_cells,
        "all_ranked_cells": all_ranked_cells
    }

    with open(stage_dir / "pll_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection_artifact, f, indent=2)

    logger.info(f"Phase 2 complete: Selected {len(selected_cells)} focused configurations:")
    for s in selected_cells:
        logger.info(f"  * Rep: {s['representation']:10s} | Subset: {s['subset']:20s} | Rank: {s['rank']:2d} | Top1: {s['mean_top1']:.4f}")

    # =========================================================================
    # PHASE 3: Focused Domain-Aware 15% Masking Evaluation (28 runs)
    # =========================================================================
    focused_domain_runs = len(selected_cells) * len(target_models)
    logger.info(f"Phase 3: Starting focused Domain-Aware 15% evaluations ({focused_domain_runs} runs: {len(selected_cells)} cells x {len(target_models)} models)...")

    focused_domain_results = []
    masking_comparisons = []
    domain_run_count = 0

    for s_cell in selected_cells:
        rep = s_cell["representation"]
        sub = s_cell["subset"]

        rep_path = reps_dir / f"{rep}.jsonl"
        with open(rep_path, "r", encoding="utf-8") as f:
            rep_records = [json.loads(l) for l in f]

        sub_path = subsets_dir / f"{sub}.jsonl"
        with open(sub_path, "r", encoding="utf-8") as f:
            sub_occ_ids = {json.loads(l)["occurrence_id"] for l in f}

        target_docs = [
            rec["document"]
            for rec in rep_records
            if rec.get("occurrence_id") in sub_occ_ids
        ][:200]

        for model_name in target_models:
            clean_model = clean_model_filename(model_name)
            cache_key = f"{clean_model}__{rep}__{sub}.json"
            cache_path = cache_domain_dir / cache_key

            eval_domain_rec = None
            if not args.fresh and cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f_c:
                    eval_domain_rec = json.load(f_c)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForMaskedLM.from_pretrained(model_name)
                    model.to(device)
                    model.eval()
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name} for Domain-Aware evaluation: {e}")
                    continue

                cell_seed = stable_seed(model_name, rep, sub, "domain_aware_15")
                domain_res = evaluate_model_on_docs(
                    model, tokenizer, target_docs, vocab_terms, device,
                    masking_strategy="domain_aware_15", seed=cell_seed, max_docs=200, max_length=256, batch_size=16
                )

                eval_domain_rec = {
                    "model_name": model_name,
                    "clean_model_name": clean_model,
                    "representation": rep,
                    "subset": sub,
                    "evaluated_doc_count": len(target_docs),
                    "masking_condition": "domain_aware_15",
                    "experiment_metadata": {
                        "masking_strategy": "domain_aware_15",
                        "mask_rate": 0.15,
                        "priority": [
                            "rare_maritime",
                            "maritime_vocabulary",
                            "random_general_fill"
                        ],
                        "max_length": 256,
                        "evaluation_documents": len(target_docs),
                        "seed": cell_seed
                    },
                    "evaluation_metrics": domain_res
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(eval_domain_rec, f, indent=2)

            focused_domain_results.append(eval_domain_rec)
            domain_run_count += 1

            # Retrieve matching Random-15 record for side-by-side comparison
            rand_cache_path = cache_random_dir / cache_key
            rand_metrics = {}
            if rand_cache_path.exists():
                with open(rand_cache_path, "r", encoding="utf-8") as f_r:
                    rand_metrics = json.load(f_r).get("evaluation_metrics", {})

            dom_metrics = eval_domain_rec.get("evaluation_metrics", {})
            r_mar = rand_metrics.get("maritime_tokens_summary", {})
            d_mar = dom_metrics.get("maritime_tokens_summary", {})
            r_rare = rand_metrics.get("rare_maritime_tokens_summary", {})
            d_rare = dom_metrics.get("rare_maritime_tokens_summary", {})

            comparison_entry = {
                "model_name": model_name,
                "representation": rep,
                "subset": sub,
                "random_15": {
                    "mlm_loss": r_mar.get("mlm_loss", 0.0),
                    "top1": r_mar.get("top1_accuracy", 0.0),
                    "top5": r_mar.get("top5_accuracy", 0.0),
                    "top10": r_mar.get("top10_accuracy", 0.0),
                    "rare_top1": r_rare.get("top1_accuracy", 0.0)
                },
                "domain_aware_15": {
                    "mlm_loss": d_mar.get("mlm_loss", 0.0),
                    "top1": d_mar.get("top1_accuracy", 0.0),
                    "top5": d_mar.get("top5_accuracy", 0.0),
                    "top10": d_mar.get("top10_accuracy", 0.0),
                    "rare_top1": d_rare.get("top1_accuracy", 0.0),
                    "masking_diagnostics": dom_metrics.get("masking_diagnostics", {})
                },
                "delta_domain_minus_random": {
                    "top1_delta": round(d_mar.get("top1_accuracy", 0.0) - r_mar.get("top1_accuracy", 0.0), 4),
                    "rare_top1_delta": round(d_rare.get("top1_accuracy", 0.0) - r_rare.get("top1_accuracy", 0.0), 4),
                    "loss_delta": round(d_mar.get("mlm_loss", 0.0) - r_mar.get("mlm_loss", 0.0), 4)
                }
            }
            masking_comparisons.append(comparison_entry)

    with open(stage_dir / "focused_domain_aware_results.json", "w", encoding="utf-8") as f:
        json.dump(focused_domain_results, f, indent=2)

    with open(stage_dir / "masking_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment_description": "Controlled comparison of Random-15 vs Domain-Aware-15 on screened high-priority configurations. Note: Domain-Aware-15 prioritizes domain tokens within the 15% budget; in dense maritime text, general_mask_fraction may approach 0, concentrating evaluation on domain terminology.",
            "evaluated_configurations_count": len(masking_comparisons),
            "comparisons": masking_comparisons
        }, f, indent=2)

    logger.info(f"Phase 3 complete: Saved {len(focused_domain_results)} domain-aware evaluations and comparison artifact.")

    # =========================================================================
    # PHASE 4: Focused Sampled Pseudo-Log-Likelihood (PLL) Evaluation (28 runs)
    # =========================================================================
    focused_pll_runs = len(selected_cells) * len(target_models)
    logger.info(f"Phase 4: Starting focused Sampled PLL evaluations ({focused_pll_runs} runs: {len(selected_cells)} cells x {len(target_models)} models)...")

    pll_results = []
    pll_run_count = 0

    for s_cell in selected_cells:
        rep = s_cell["representation"]
        sub = s_cell["subset"]

        rep_path = reps_dir / f"{rep}.jsonl"
        with open(rep_path, "r", encoding="utf-8") as f:
            rep_records = [json.loads(l) for l in f]

        sub_path = subsets_dir / f"{sub}.jsonl"
        with open(sub_path, "r", encoding="utf-8") as f:
            sub_occ_ids = {json.loads(l)["occurrence_id"] for l in f}

        target_docs = [
            rec["document"]
            for rec in rep_records
            if rec.get("occurrence_id") in sub_occ_ids
        ][:10]

        for model_name in target_models:
            clean_model = clean_model_filename(model_name)
            cache_key = f"{clean_model}__{rep}__{sub}.json"
            cache_path = cache_pll_dir / cache_key

            eval_pll_rec = None
            if not args.fresh and cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f_c:
                    eval_pll_rec = json.load(f_c)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForMaskedLM.from_pretrained(model_name)
                    model.to(device)
                    model.eval()
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name} for PLL evaluation: {e}")
                    continue

                cell_seed = stable_seed(model_name, rep, sub, "pll")
                pll_res = evaluate_sampled_pll(
                    model, tokenizer, target_docs, vocab_terms, device,
                    max_docs=10, max_seq_len=128, max_positions_per_doc=32, seed=cell_seed
                )

                eval_pll_rec = {
                    "model_name": model_name,
                    "clean_model_name": clean_model,
                    "representation": rep,
                    "subset": sub,
                    "evaluated_doc_count": len(target_docs),
                    "pll_metrics": pll_res
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(eval_pll_rec, f, indent=2)

            pll_results.append(eval_pll_rec)
            pll_run_count += 1

    with open(stage_dir / "pll_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment_description": "Sampled Pseudo-Log-Likelihood (Sampled PLL) scoring (Salazar et al., ACL 2020) across focused screened configurations using exact single-token masking per sampled position",
            "evaluated_configurations_count": len(pll_results),
            "results": pll_results
        }, f, indent=2)

    logger.info(f"Phase 4 complete: Saved {len(pll_results)} Sampled PLL evaluation records to {stage_dir / 'pll_results.json'}")

    logger.info("=========================================================================")
    logger.info("Stage 14 MLM Evaluation & Analysis Completed Successfully.")
    logger.info(f"  * Broad Random-15 evaluations: {len(random_15_records)}")
    logger.info(f"  * Screened Configurations Selected: {len(selected_cells)}")
    logger.info(f"  * Focused Domain-Aware-15 evaluations: {len(focused_domain_results)}")
    logger.info(f"  * Focused Sampled-PLL evaluations: {len(pll_results)}")
    logger.info("=========================================================================")

if __name__ == "__main__":
    main()
