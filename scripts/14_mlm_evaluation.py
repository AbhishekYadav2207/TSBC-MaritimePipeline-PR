import os
import sys
import json
import math
import random
import time
import re
import hashlib
import argparse
from collections import defaultdict
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

# Explicit, configuration-driven permanent benchmark model exclusions
EXCLUDED_MODELS = {
    "microsoft/deberta-v3-base": {
        "reason": "Excluded after benchmark compatibility anomaly: zero MLM Top-1/Top-5/Top-10 across the evaluated standard cells and English diagnostic."
    }
}

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


def compute_cache_key(model_name: str, rep: str, sub: str, masking_mode: str = "subword", evaluation_unit: str = "subword") -> str:
    """
    Computes a unique, collision-proof cache key incorporating model identity,
    representation, subset partition, masking strategy, and evaluation scoring unit.
    """
    clean_model = model_name.replace("/", "_").replace("-", "_")
    if masking_mode in ("wwm_word", "word") or (masking_mode == "whole_word" and evaluation_unit == "word"):
        sub_dir = "cache_wwm_word"
    elif masking_mode in ("wwm_subword", "whole_word"):
        sub_dir = "cache_wwm_subword"
    else:
        sub_dir = "cache"
    return f"{sub_dir}/{clean_model}__{rep}__{sub}.json"


def load_selected_models(stage13_path: Path, fallback_models: list = None) -> list:
    """
    Dynamically loads authoritative models selected by Stage 13.
    Accepts list or dict manifest, ensuring non-empty cohort.
    Filters out models in EXCLUDED_MODELS.
    Falls back to documented defaults if Stage 13 artifact is missing or invalid.
    """
    if stage13_path.exists():
        try:
            with open(stage13_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_models = data
            elif isinstance(data, dict):
                raw_models = data.get("selected_models", [])
            else:
                raw_models = []

            # Filter out permanently excluded models
            models = [m for m in raw_models if m not in EXCLUDED_MODELS]
            excluded_found = [m for m in raw_models if m in EXCLUDED_MODELS]
            if excluded_found:
                logger.warning(f"Filtered out {len(excluded_found)} excluded models from Stage 13 cohort: {excluded_found}")

            if isinstance(models, list) and len(models) > 0:
                logger.info(f"Successfully loaded {len(models)} authoritative active models from Stage 13: {stage13_path}")
                return models
            else:
                found_cnt = len(models) if isinstance(models, list) else "invalid format"
                logger.warning(
                    f"Stage 13 artifact at {stage13_path} did not provide valid model list (found {found_cnt}). "
                    f"Falling back to default representative models."
                )
        except Exception as e:
            logger.warning(f"Failed to read Stage 13 artifact ({e}). Using fallback models.")
    else:
        logger.warning(f"Stage 13 artifact not found at {stage13_path}. Falling back to default {len(fallback_models)} models.")
    
    return [m for m in fallback_models if m not in EXCLUDED_MODELS]

DEFAULT_REPRESENTATIONS = ["narrative", "key_value", "template", "json", "mixed"]
DEFAULT_SUBSETS = ["high_knowledge", "medium_knowledge", "low_knowledge", "balanced_knowledge", "random_baseline"]

def discover_representations(reps_dir: Path, fallback_reps: list) -> list:
    """Dynamically discovers representation files from Stage 11."""
    if reps_dir.exists():
        found = [p.stem for p in sorted(reps_dir.glob("*.jsonl"))]
        if found:
            logger.info(f"Discovered {len(found)} representations from {reps_dir}: {found}")
            return found
    return list(fallback_reps)

def discover_subsets(subsets_dir: Path, fallback_subs: list) -> list:
    """
    Dynamically discovers canonical benchmark subset files from Stage 12.
    Filters strictly to canonical stratification subsets, excluding auxiliary
    general_english_baseline and percentage ratio scaling subsets.
    """
    if subsets_dir.exists():
        canonical_set = set(fallback_subs)
        found = [
            p.stem for p in sorted(subsets_dir.glob("*.jsonl"))
            if p.stem in canonical_set and p.stem != "general_english_baseline"
        ]
        if found:
            logger.info(f"Discovered {len(found)} canonical benchmark subsets from {subsets_dir}: {found}")
            return found
    return list(fallback_subs)

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

def classify_token_positions(
    raw_text: str,
    input_ids: list,
    offsets: list,
    special_tokens_mask: list,
    vocab_terms_set: set,
    rare_terms_set: set,
    maritime_token_ids: set,
    rare_token_ids: set,
    category_token_ids: dict = None
):
    """
    Experiment 2:
    Legacy token-ID based classification from the original Stage 14.

    A token is classified as:
      - Rare maritime  -> token ID is in rare_token_ids
      - Maritime       -> token ID is in maritime_token_ids
      - General        -> neither

    No character-offset or span matching is used.
    """
    if category_token_ids is None or not isinstance(category_token_ids, dict):
        raise TypeError(
            f"category_token_ids must be a non-null dictionary mapping category names to token ID sets, got {type(category_token_ids).__name__}"
        )

    eligible_positions = []

    for i, token_id in enumerate(input_ids):

        # Exclude special tokens
        if special_tokens_mask[i]:
            continue

        # Exclude padding
        if offsets is not None and offsets[i] == [0, 0]:
            continue

        eligible_positions.append(i)

    # Legacy token-ID classification
    rare_positions = set()
    maritime_positions = set()

    category_positions = {
        cat: set()
        for cat in CATEGORIES
    }

    for pos in eligible_positions:

        token_id = input_ids[pos]

        # Same logic as old Stage 14:
        # is_rare = target_id in rare_token_ids
        if token_id in rare_token_ids:
            rare_positions.add(pos)

        # Same logic as old Stage 14:
        # is_maritime = target_id in maritime_token_ids
        if token_id in maritime_token_ids:
            maritime_positions.add(pos)

            # Legacy category classification
            for cat, cat_ids in category_token_ids.items():
                if token_id in cat_ids:
                    category_positions[cat].add(pos)

    # General = eligible tokens that are not maritime
    general_positions = [
        pos
        for pos in eligible_positions
        if pos not in maritime_positions
    ]

    return (
        eligible_positions,
        rare_positions,
        maritime_positions,
        general_positions,
        category_positions
    )

def create_random_mask(eligible_positions: list, rng: random.Random, mask_budget: int):
    """Conventional Random-15% subword masking baseline."""
    if not eligible_positions or mask_budget <= 0:
        return []
    if len(eligible_positions) <= mask_budget:
        return list(eligible_positions)
    return rng.sample(eligible_positions, mask_budget)

def create_domain_aware_mask(eligible_positions: list, rare_positions: set, maritime_positions: set,
                             general_positions: list, rng: random.Random, mask_budget: int):
    """
    Lightweight domain-aware selective masking policy (subword level).
    Total budget is strictly maintained at approximately 15% of eligible tokens.
    Priority:
      1. Rare maritime tokens (highest)
      2. Domain maritime vocabulary tokens (next)
      3. General tokens (random fill to meet total budget)
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

def extract_word_groups(word_ids: list, eligible_positions: list) -> tuple:
    """
    Groups eligible token positions by intact whole-word boundaries.
    word_ids: list mapping token position -> word integer id (or None for special tokens)
    Returns:
      word_to_positions: dict mapping wid -> list of token positions [p1, p2, ...]
      position_to_word: dict mapping pos -> wid
    """
    word_to_positions = defaultdict(list)
    position_to_word = {}
    eligible_set = set(eligible_positions)
    for pos, wid in enumerate(word_ids):
        if wid is not None and pos in eligible_set:
            word_to_positions[wid].append(pos)
            position_to_word[pos] = wid
    return dict(word_to_positions), position_to_word

def create_whole_word_random_mask(word_to_positions: dict, rng: random.Random, mask_budget: int) -> tuple:
    """
    Selects whole words until the total number of subword pieces reaches mask_budget.
    Guarantees all subword pieces of a selected word are masked together (zero sibling leakage).
    """
    if not word_to_positions or mask_budget <= 0:
        return [], set()

    words = list(word_to_positions.keys())
    rng.shuffle(words)

    selected_positions = []
    selected_words = set()
    total_tokens = 0

    for wid in words:
        positions = word_to_positions[wid]
        w_len = len(positions)
        if total_tokens == 0 or total_tokens + w_len <= mask_budget or (total_tokens < mask_budget and (total_tokens + w_len - mask_budget <= mask_budget - total_tokens)):
            selected_positions.extend(positions)
            selected_words.add(wid)
            total_tokens += w_len
            if total_tokens >= mask_budget:
                break

    return selected_positions, selected_words

def create_whole_word_domain_aware_mask(
    word_to_positions: dict, rare_positions: set, maritime_positions: set,
    general_positions: list, rng: random.Random, mask_budget: int
) -> tuple:
    """
    Selects whole words prioritizing rare maritime terms, then maritime vocabulary,
    then general words, up to mask_budget. All pieces of selected words are masked together.
    """
    if not word_to_positions or mask_budget <= 0:
        return [], set()

    rare_words = []
    maritime_words = []
    general_words = []

    for wid, positions in word_to_positions.items():
        if any(p in rare_positions for p in positions):
            rare_words.append(wid)
        elif any(p in maritime_positions for p in positions):
            maritime_words.append(wid)
        else:
            general_words.append(wid)

    rng.shuffle(rare_words)
    rng.shuffle(maritime_words)
    rng.shuffle(general_words)

    selected_positions = []
    selected_words = set()
    total_tokens = 0

    for pool in [rare_words, maritime_words, general_words]:
        for wid in pool:
            if total_tokens >= mask_budget:
                break
            positions = word_to_positions[wid]
            w_len = len(positions)
            if total_tokens == 0 or total_tokens + w_len <= mask_budget or (total_tokens < mask_budget and (total_tokens + w_len - mask_budget <= mask_budget - total_tokens)):
                selected_positions.extend(positions)
                selected_words.add(wid)
                total_tokens += w_len
        if total_tokens >= mask_budget:
            break

    return selected_positions, selected_words

def evaluate_model_on_docs(model, tokenizer, docs: list, vocab_terms: list, device: torch.device,
                           masking_strategy: str = "random_15", seed: int = 42,
                           max_docs: int = 200, max_length: int = 256, batch_size: int = 16,
                           masking_mode: str = "subword") -> dict:
    """
    Evaluates a pretrained MLM model on a collection of documents.
    Supports three scientifically verified evaluation modes (A01 robustness):
      - MODE 1: 'subword' -> subword masking + subword evaluation (baseline default)
      - MODE 2: 'wwm_subword' -> whole-word masking + subword diagnostic evaluation
      - MODE 3: 'wwm_word' -> whole-word masking + strict word reconstruction evaluation
    """
    if not docs:
        return {}

    is_wwm = masking_mode in ("wwm_subword", "wwm_word")
    eval_unit = "word" if masking_mode == "wwm_word" else "subword"

    if is_wwm and not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer {tokenizer.__class__.__name__} is not a fast tokenizer supporting word_ids(). "
            f"Whole-Word Masking (WWM) requires verified word-boundary mapping. Silent fallback to subword masking is strictly disallowed."
        )

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

    # Strict whole-word reconstruction tracking (for Modes 2 & 3)
    word_eval_stats = {
        "total_words": 0,
        "correct_words_top1": 0,
        "correct_words_top5": 0,
        "correct_words_top10": 0,
        "maritime_total_words": 0,
        "maritime_correct_words_top1": 0,
        "maritime_correct_words_top5": 0,
        "maritime_correct_words_top10": 0,
        "rare_total_words": 0,
        "rare_correct_words_top1": 0,
        "rare_correct_words_top5": 0,
        "rare_correct_words_top10": 0,
        "general_total_words": 0,
        "general_correct_words_top1": 0,
        "general_correct_words_top5": 0,
        "general_correct_words_top10": 0
    }

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
            batch_selected_words = []
            batch_word_to_positions = []

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

                if is_wwm:
                    try:
                        wids = inputs.word_ids(b)
                    except Exception as e:
                        raise RuntimeError(f"Tokenizer failed to provide word_ids() for Whole-Word Masking: {e}. Silent fallback is disallowed.")
                    if wids is None:
                        raise RuntimeError("Tokenizer returned None for word_ids(). Cannot construct Whole-Word Mask without verified word boundaries. Silent fallback is disallowed.")

                    word_to_positions, position_to_word = extract_word_groups(wids, eligible_positions)
                    batch_word_to_positions.append(word_to_positions)

                    if masking_strategy == "domain_aware_15":
                        selected, sel_words = create_whole_word_domain_aware_mask(
                            word_to_positions, rare_pos, mar_pos, gen_pos, rng, budget
                        )
                    else:
                        selected, sel_words = create_whole_word_random_mask(word_to_positions, rng, budget)
                    batch_selected_words.append(sel_words)
                else:
                    batch_word_to_positions.append({})
                    batch_selected_words.append(set())
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

                # Store per-position top-k predictions for word-level reconstruction evaluation
                b_pos_pred_top1 = {}
                b_pos_pred_top5 = {}
                b_pos_pred_top10 = {}

                for pos_tensor in mask_positions:
                    pos = pos_tensor.item()
                    target_id = labels[b, pos].item()
                    token_logits = logits[b, pos]

                    token_loss = criterion(token_logits.unsqueeze(0), torch.tensor([target_id], device=device)).item()
                    top_k_indices = torch.topk(token_logits, 10).indices.tolist()

                    is_top1 = 1 if target_id == top_k_indices[0] else 0
                    is_top5 = 1 if target_id in top_k_indices[:5] else 0
                    is_top10 = 1 if target_id in top_k_indices[:10] else 0

                    b_pos_pred_top1[pos] = (target_id == top_k_indices[0])
                    b_pos_pred_top5[pos] = (target_id in top_k_indices[:5])
                    b_pos_pred_top10[pos] = (target_id in top_k_indices[:10])

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

                # Strict whole-word reconstruction evaluation
                if is_wwm:
                    sel_words_set = batch_selected_words[b]
                    w2p = batch_word_to_positions[b]
                    for wid in sel_words_set:
                        w_positions = w2p.get(wid, [])
                        if not w_positions:
                            continue
                        # A word is reconstructed correctly iff ALL subword pieces are correctly predicted
                        word_correct_top1 = all(b_pos_pred_top1.get(p, False) for p in w_positions)
                        word_correct_top5 = all(b_pos_pred_top5.get(p, False) for p in w_positions)
                        word_correct_top10 = all(b_pos_pred_top10.get(p, False) for p in w_positions)

                        w_is_rare = any(p in rare_pos_set for p in w_positions)
                        w_is_maritime = w_is_rare or any(p in mar_pos_set for p in w_positions)

                        word_eval_stats["total_words"] += 1
                        if word_correct_top1:
                            word_eval_stats["correct_words_top1"] += 1
                        if word_correct_top5:
                            word_eval_stats["correct_words_top5"] += 1
                        if word_correct_top10:
                            word_eval_stats["correct_words_top10"] += 1

                        if w_is_rare:
                            word_eval_stats["rare_total_words"] += 1
                            if word_correct_top1:
                                word_eval_stats["rare_correct_words_top1"] += 1
                            if word_correct_top5:
                                word_eval_stats["rare_correct_words_top5"] += 1
                            if word_correct_top10:
                                word_eval_stats["rare_correct_words_top10"] += 1

                        if w_is_maritime:
                            word_eval_stats["maritime_total_words"] += 1
                            if word_correct_top1:
                                word_eval_stats["maritime_correct_words_top1"] += 1
                            if word_correct_top5:
                                word_eval_stats["maritime_correct_words_top5"] += 1
                            if word_correct_top10:
                                word_eval_stats["maritime_correct_words_top10"] += 1
                        else:
                            word_eval_stats["general_total_words"] += 1
                            if word_correct_top1:
                                word_eval_stats["general_correct_words_top1"] += 1
                            if word_correct_top5:
                                word_eval_stats["general_correct_words_top5"] += 1
                            if word_correct_top10:
                                word_eval_stats["general_correct_words_top10"] += 1

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
            "subword_top1_accuracy": float(st["top1"] / cnt),
            "top5_accuracy": float(st["top5"] / cnt),
            "top10_accuracy": float(st["top10"] / cnt)
        }

    gen_summary = summarize(general_stats)
    mar_summary = summarize(maritime_stats)
    rare_summary = summarize(rare_stats)
    cat_summaries = {cat: summarize(st) for cat, st in cat_stats.items()}

    performance_gap = gen_summary["subword_top1_accuracy"] - mar_summary["subword_top1_accuracy"]

    # Overall masked subword accuracy (combining general + maritime)
    tot_eval_count = general_stats["count"] + maritime_stats["count"]
    if tot_eval_count > 0:
        overall_subword_top1 = (general_stats["top1"] + maritime_stats["top1"]) / tot_eval_count
        overall_loss = (general_stats["loss"] + maritime_stats["loss"]) / tot_eval_count
    else:
        overall_subword_top1 = gen_summary["subword_top1_accuracy"]
        overall_loss = gen_summary["mlm_loss"]

    actual_mask_rate = float(diag_masked / diag_eligible) if diag_eligible > 0 else 0.0
    safe_masked = max(diag_masked, 1)

    # Word reconstruction accuracy calculations for WWM
    word_recon_top1 = None
    word_recon_top5 = None
    word_recon_top10 = None
    mar_word_recon_top1 = None
    mar_word_recon_top5 = None
    mar_word_recon_top10 = None
    rare_word_recon_top1 = None
    rare_word_recon_top5 = None
    rare_word_recon_top10 = None
    gen_word_recon_top1 = None
    gen_word_recon_top5 = None
    gen_word_recon_top10 = None

    if is_wwm and word_eval_stats["total_words"] > 0:
        tot_w = max(word_eval_stats["total_words"], 1)
        word_recon_top1 = float(word_eval_stats["correct_words_top1"] / tot_w)
        word_recon_top5 = float(word_eval_stats["correct_words_top5"] / tot_w)
        word_recon_top10 = float(word_eval_stats["correct_words_top10"] / tot_w)

        mar_w = max(word_eval_stats["maritime_total_words"], 1)
        mar_word_recon_top1 = float(word_eval_stats["maritime_correct_words_top1"] / mar_w)
        mar_word_recon_top5 = float(word_eval_stats["maritime_correct_words_top5"] / mar_w)
        mar_word_recon_top10 = float(word_eval_stats["maritime_correct_words_top10"] / mar_w)

        rare_w = max(word_eval_stats["rare_total_words"], 1)
        rare_word_recon_top1 = float(word_eval_stats["rare_correct_words_top1"] / rare_w)
        rare_word_recon_top5 = float(word_eval_stats["rare_correct_words_top5"] / rare_w)
        rare_word_recon_top10 = float(word_eval_stats["rare_correct_words_top10"] / rare_w)

        gen_w = max(word_eval_stats["general_total_words"], 1)
        gen_word_recon_top1 = float(word_eval_stats["general_correct_words_top1"] / gen_w)
        gen_word_recon_top5 = float(word_eval_stats["general_correct_words_top5"] / gen_w)
        gen_word_recon_top10 = float(word_eval_stats["general_correct_words_top10"] / gen_w)

    # In Mode 3 ('wwm_word'), primary top1/5/10 is strict word reconstruction
    if masking_mode == "wwm_word":
        primary_overall_top1 = word_recon_top1 if word_recon_top1 is not None else overall_subword_top1
        mar_summary["top1_accuracy"] = mar_word_recon_top1 if mar_word_recon_top1 is not None else mar_summary["top1_accuracy"]
        mar_summary["top5_accuracy"] = mar_word_recon_top5 if mar_word_recon_top5 is not None else mar_summary["top5_accuracy"]
        mar_summary["top10_accuracy"] = mar_word_recon_top10 if mar_word_recon_top10 is not None else mar_summary["top10_accuracy"]

        gen_summary["top1_accuracy"] = gen_word_recon_top1 if gen_word_recon_top1 is not None else gen_summary["top1_accuracy"]
        gen_summary["top5_accuracy"] = gen_word_recon_top5 if gen_word_recon_top5 is not None else gen_summary["top5_accuracy"]
        gen_summary["top10_accuracy"] = gen_word_recon_top10 if gen_word_recon_top10 is not None else gen_summary["top10_accuracy"]

        rare_summary["top1_accuracy"] = rare_word_recon_top1 if rare_word_recon_top1 is not None else rare_summary["top1_accuracy"]
        rare_summary["top5_accuracy"] = rare_word_recon_top5 if rare_word_recon_top5 is not None else rare_summary["top5_accuracy"]
        rare_summary["top10_accuracy"] = rare_word_recon_top10 if rare_word_recon_top10 is not None else rare_summary["top10_accuracy"]
    else:
        primary_overall_top1 = overall_subword_top1

    # Attach explicit unambiguous metrics to summaries
    mar_summary["word_reconstruction_accuracy"] = mar_word_recon_top1
    gen_summary["word_reconstruction_accuracy"] = gen_word_recon_top1
    rare_summary["word_reconstruction_accuracy"] = rare_word_recon_top1

    result = {
        "evaluated_documents": len(eval_docs),
        "evaluation_time_sec": float(eval_time),
        "masking_strategy": masking_strategy,
        "masking_mode": "whole_word" if is_wwm else "subword",
        "evaluation_unit": eval_unit,
        "overall_summary": {
            "total_masked_tokens": diag_masked,
            "total_masked_words": word_eval_stats["total_words"] if is_wwm else None,
            "overall_top1_accuracy": float(primary_overall_top1),
            "subword_top1_accuracy": float(overall_subword_top1),
            "word_reconstruction_accuracy": word_recon_top1,
            "word_reconstruction_top1_accuracy": word_recon_top1,
            "word_reconstruction_top5_accuracy": word_recon_top5,
            "word_reconstruction_top10_accuracy": word_recon_top10,
            "maritime_word_reconstruction_top1_accuracy": mar_word_recon_top1,
            "maritime_word_reconstruction_top5_accuracy": mar_word_recon_top5,
            "maritime_word_reconstruction_top10_accuracy": mar_word_recon_top10,
            "overall_mlm_loss": float(overall_loss)
        },
        "general_tokens_summary": gen_summary,
        "maritime_tokens_summary": mar_summary,
        "rare_maritime_tokens_summary": rare_summary,
        "category_recall": {cat: round(st["top1_accuracy"], 4) for cat, st in cat_summaries.items()},
        "category_breakdown": cat_summaries,
        "performance_gap_top1": float(performance_gap)
    }

    if is_wwm:
        result["word_reconstruction_summary"] = {
            "total_masked_words": word_eval_stats["total_words"],
            "correct_words_top1": word_eval_stats["correct_words_top1"],
            "correct_words_top5": word_eval_stats["correct_words_top5"],
            "correct_words_top10": word_eval_stats["correct_words_top10"],
            "word_reconstruction_top1_accuracy": word_recon_top1,
            "word_reconstruction_top5_accuracy": word_recon_top5,
            "word_reconstruction_top10_accuracy": word_recon_top10,
            "maritime_masked_words": word_eval_stats["maritime_total_words"],
            "maritime_correct_words_top1": word_eval_stats["maritime_correct_words_top1"],
            "maritime_correct_words_top5": word_eval_stats["maritime_correct_words_top5"],
            "maritime_correct_words_top10": word_eval_stats["maritime_correct_words_top10"],
            "maritime_word_reconstruction_top1_accuracy": mar_word_recon_top1,
            "maritime_word_reconstruction_top5_accuracy": mar_word_recon_top5,
            "maritime_word_reconstruction_top10_accuracy": mar_word_recon_top10,
            "rare_masked_words": word_eval_stats["rare_total_words"],
            "rare_correct_words_top1": word_eval_stats["rare_correct_words_top1"],
            "rare_correct_words_top5": word_eval_stats["rare_correct_words_top5"],
            "rare_correct_words_top10": word_eval_stats["rare_correct_words_top10"],
            "rare_word_reconstruction_top1_accuracy": rare_word_recon_top1,
            "rare_word_reconstruction_top5_accuracy": rare_word_recon_top5,
            "rare_word_reconstruction_top10_accuracy": rare_word_recon_top10,
            "general_masked_words": word_eval_stats["general_total_words"],
            "general_correct_words_top1": word_eval_stats["general_correct_words_top1"],
            "general_correct_words_top5": word_eval_stats["general_correct_words_top5"],
            "general_correct_words_top10": word_eval_stats["general_correct_words_top10"],
            "general_word_reconstruction_top1_accuracy": gen_word_recon_top1,
            "general_word_reconstruction_top5_accuracy": gen_word_recon_top5,
            "general_word_reconstruction_top10_accuracy": gen_word_recon_top10
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
    Uses legacy token-ID classification for rare, maritime, and general
    evaluation categories to preserve comparability with the original
    Stage 14 benchmark.
    """
    if not docs:
        return {}

    rng = random.Random(seed)
    maritime_token_ids, rare_token_ids, category_token_ids = build_vocabulary_token_sets(tokenizer, vocab_terms)
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
                maritime_token_ids, rare_token_ids,
                category_token_ids
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
    Analyzes the representation x subset cells dynamically from Random-15 results.
    Ranks cells by mean_top1 (descending) and mean_mlm_loss (ascending).
    Selects top diverse configurations balancing strong performance,
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

    # 3. Test Mode 1: Random-15 Subword Baseline (backward compatible)
    eval_rand = evaluate_model_on_docs(
        model, tokenizer, sample_docs, vocab_terms, device,
        masking_strategy="random_15", max_docs=len(sample_docs), max_length=64, batch_size=2,
        masking_mode="subword"
    )
    assert "overall_summary" in eval_rand, "Random-15 summary missing"
    assert eval_rand["masking_mode"] == "subword", f"Expected subword masking mode, got {eval_rand['masking_mode']}"
    assert eval_rand["evaluation_unit"] == "subword", f"Expected subword evaluation unit, got {eval_rand['evaluation_unit']}"
    assert 0.0 <= eval_rand["overall_summary"]["overall_top1_accuracy"] <= 1.0, "Invalid top-1 accuracy range"
    assert math.isfinite(eval_rand["overall_summary"]["overall_mlm_loss"]), "Loss must be finite"
    logger.info(f"[OK] Mode 1 (Subword Baseline) passed: Top1={eval_rand['overall_summary']['overall_top1_accuracy']:.4f}, Loss={eval_rand['overall_summary']['overall_mlm_loss']:.4f}")

    # 4. Test Whole-Word Masking (WWM) Grouping & Mask Construction
    test_wwm_text = "The containership had unseaworthiness issues in 2024-2025 with EPIRB navigation."
    enc_test = tokenizer([test_wwm_text], return_offsets_mapping=True, return_tensors="pt")
    wids_test = enc_test.word_ids(0)
    sp_mask_test = tokenizer.get_special_tokens_mask(enc_test["input_ids"][0].tolist(), already_has_special_tokens=True)
    el_pos_test = [idx for idx, wid in enumerate(wids_test) if wid is not None and not sp_mask_test[idx]]
    w2p_test, p2w_test = extract_word_groups(wids_test, el_pos_test)

    # Verify multi-piece word grouping
    multi_piece_words = [wid for wid, poses in w2p_test.items() if len(poses) > 1]
    single_piece_words = [wid for wid, poses in w2p_test.items() if len(poses) == 1]
    assert len(w2p_test) > 0, "Failed to identify intact words"
    assert len(multi_piece_words) > 0, "Failed to detect multi-piece words in technical text"
    assert len(single_piece_words) > 0, "Failed to detect single-piece words"

    # Verify WWM mask construction: all pieces of a selected word must be masked together
    rng = random.Random(42)
    sel_test_pos, sel_test_words = create_whole_word_random_mask(w2p_test, rng, mask_budget=max(1, round(0.15 * len(el_pos_test))))
    for wid in sel_test_words:
        for p in w2p_test[wid]:
            assert p in sel_test_pos, f"Subword piece at pos {p} for word {wid} was leaked (not masked under WWM)!"
    logger.info(f"[OK] Whole-Word Masking grouping & construction passed (multi-piece: {len(multi_piece_words)}, single-piece: {len(single_piece_words)}, no sibling leakage).")

    # 5. Test Mode 2: Whole-Word Masking + Subword Diagnostic Evaluation
    eval_wwm_sub = evaluate_model_on_docs(
        model, tokenizer, sample_docs, vocab_terms, device,
        masking_strategy="random_15", max_docs=len(sample_docs), max_length=64, batch_size=2,
        masking_mode="wwm_subword"
    )
    assert eval_wwm_sub["masking_mode"] == "whole_word", f"Expected whole_word mode, got {eval_wwm_sub['masking_mode']}"
    assert eval_wwm_sub["evaluation_unit"] == "subword", f"Expected subword unit, got {eval_wwm_sub['evaluation_unit']}"
    assert 0.0 <= eval_wwm_sub["overall_summary"]["subword_top1_accuracy"] <= 1.0
    logger.info(f"[OK] Mode 2 (WWM + Subword Diagnostic) passed: SubwordTop1={eval_wwm_sub['overall_summary']['subword_top1_accuracy']:.4f}")

    # 6. Test Mode 3: Whole-Word Masking + Strict Word Reconstruction Evaluation
    eval_wwm_word = evaluate_model_on_docs(
        model, tokenizer, sample_docs, vocab_terms, device,
        masking_strategy="random_15", max_docs=len(sample_docs), max_length=64, batch_size=2,
        masking_mode="wwm_word"
    )
    assert eval_wwm_word["masking_mode"] == "whole_word"
    assert eval_wwm_word["evaluation_unit"] == "word"
    word_recon_acc = eval_wwm_word["overall_summary"]["word_reconstruction_accuracy"]
    sub_acc = eval_wwm_word["overall_summary"]["subword_top1_accuracy"]
    assert word_recon_acc is not None, "Word reconstruction accuracy must not be None in Mode 3"
    assert 0.0 <= word_recon_acc <= 1.0, f"Word reconstruction accuracy out of range: {word_recon_acc}"
    assert 0.0 <= sub_acc <= 1.0, f"Subword accuracy out of range: {sub_acc}"
    logger.info(f"[OK] Mode 3 (WWM + Strict Word Reconstruction) passed: WordReconAcc={word_recon_acc:.4f}, SubwordTop1={sub_acc:.4f}")

    # 7. Test Domain-Aware-15 with WWM
    eval_domain = evaluate_model_on_docs(
        model, tokenizer, sample_docs, vocab_terms, device,
        masking_strategy="domain_aware_15", max_docs=len(sample_docs), max_length=64, batch_size=2,
        masking_mode="wwm_word"
    )
    assert "masking_diagnostics" in eval_domain, "Domain-Aware diagnostics missing"
    diag = eval_domain["masking_diagnostics"]
    logger.info(f"[OK] Domain-Aware-15 WWM evaluation passed: MaskRate={diag['actual_mask_rate']:.4f}, RareMasked={diag['rare_maritime_tokens_masked']}, MaritimeMasked={diag['maritime_tokens_masked']}")

    # 8. Test Sampled PLL
    eval_pll = evaluate_sampled_pll(
        model, tokenizer, sample_docs, vocab_terms, device,
        max_docs=len(sample_docs), max_seq_len=64, max_positions_per_doc=8, seed=42
    )
    assert eval_pll["number_of_scored_tokens"] > 0, "PLL scored 0 tokens"
    assert not math.isnan(eval_pll["mean_token_pll"]), "PLL returned NaN"
    logger.info(f"[OK] Sampled PLL passed: ScoredTokens={eval_pll['number_of_scored_tokens']}, MeanPLL={eval_pll['mean_token_pll']:.4f}, PseudoPPL={eval_pll['pseudo_perplexity']:.4f}")

    # 9. Test Selection Logic on synthetic records
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
    assert len(selected) == min(4, len(ranked)), f"Expected {min(4, len(ranked))} selected cells, got {len(selected)}"
    assert len(ranked) == len(reps) * len(subs), f"Expected {len(reps) * len(subs)} ranked cells, got {len(ranked)}"
    logger.info(f"[OK] Selection logic passed: selected {len(selected)} diverse configurations from {len(ranked)} cells.")

    logger.info("=== [SUCCESS] All Stage 14 Smoke Tests (Modes 1, 2, 3 + WWM + PLL) Completed Successfully! ===")
    return True

def main():
    parser = argparse.ArgumentParser(description="Stage 14: Masked Language Model Benchmarking & Analysis")
    parser.add_argument("--fresh", action="store_true", help="Force fresh recomputation of all evaluations, ignoring old cache.")
    parser.add_argument("--smoke-test", action="store_true", help="Run lightweight smoke test on a minimal sample without executing full benchmark.")
    parser.add_argument("--masking_mode", "--masking-mode", dest="masking_mode", type=str,
                        choices=["subword", "whole_word", "wwm_subword", "wwm_word"], default="subword",
                        help="MLM masking strategy: 'subword' (Mode 1 baseline) or 'whole_word' (Modes 2 & 3 WWM). Default: 'subword'.")
    parser.add_argument("--evaluation_unit", "--evaluation-unit", dest="evaluation_unit", type=str,
                        choices=["subword", "word"], default=None,
                        help="Evaluation scoring unit: 'subword' (subword token accuracy) or 'word' (strict whole-word reconstruction). Default: 'subword'.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                        help="Compute device: 'auto' (use CUDA if available), 'cuda' (require CUDA), 'cpu' (force CPU). Default: 'auto'.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of models to evaluate instead of full cohort.")
    parser.add_argument("--max-docs", type=int, default=200, help="Maximum documents per configuration cell (default: 200).")
    args = parser.parse_args()

    # Resolve masking_mode and evaluation_unit
    if args.masking_mode == "wwm_word":
        resolved_masking_mode = "whole_word"
        resolved_evaluation_unit = "word"
    elif args.masking_mode == "wwm_subword":
        resolved_masking_mode = "whole_word"
        resolved_evaluation_unit = "subword"
    elif args.masking_mode == "whole_word":
        resolved_masking_mode = "whole_word"
        resolved_evaluation_unit = args.evaluation_unit if args.evaluation_unit else "subword"
    else:  # subword
        resolved_masking_mode = "subword"
        resolved_evaluation_unit = "subword"
        if args.evaluation_unit and args.evaluation_unit != "subword":
            logger.warning("Evaluation unit 'word' is only applicable with whole-word masking. Overriding to 'subword' for baseline subword masking.")

    # Determine internal execution mode key ('subword', 'wwm_subword', 'wwm_word')
    if resolved_masking_mode == "whole_word":
        if resolved_evaluation_unit == "word":
            internal_mode = "wwm_word"
        else:
            internal_mode = "wwm_subword"
    else:
        internal_mode = "subword"

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
    if args.models:
        target_models = [m for m in target_models if m in args.models]
        logger.info(f"Filtered target models to user-specified subset: {target_models}")

    vocab_path = output_dir / "stage-10" / "maritime_vocabulary.txt"
    vocab_terms = []
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fv:
            vocab_terms = [line.strip() for line in fv if line.strip()]

    reps_dir = output_dir / "stage-11" / "corpus_representations"
    subsets_dir = output_dir / "stage-12" / "subsets"

    # Dynamically Discover Matrix Dimensions from Stage 11 and Stage 12 Outputs
    representations = discover_representations(reps_dir, DEFAULT_REPRESENTATIONS)
    subsets = discover_subsets(subsets_dir, DEFAULT_SUBSETS)

    # Load General English Baseline Subset
    gen_eng_docs = []
    gen_eng_path = subsets_dir / "general_english_baseline.jsonl"
    if gen_eng_path.exists():
        with open(gen_eng_path, "r", encoding="utf-8") as f:
            gen_eng_docs = [json.loads(l)["document"] for l in f]

    eval_out_dir = stage_dir / "evaluations"
    if internal_mode == "subword":
        cache_random_dir = eval_out_dir / "cache"
    else:
        cache_random_dir = eval_out_dir / f"cache_{internal_mode}"
    cache_domain_dir = cache_random_dir / "domain_aware_15"
    cache_pll_dir = cache_random_dir / "pll"

    cache_random_dir.mkdir(parents=True, exist_ok=True)
    cache_domain_dir.mkdir(parents=True, exist_ok=True)
    cache_pll_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA compute device requested via '--device cuda', but torch.cuda is not available on this runtime.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"Using compute device: {device} | Masking mode: {resolved_masking_mode} "
        f"| Evaluation unit: {resolved_evaluation_unit} | Internal mode: {internal_mode}"
    )

    total_random_runs = len(target_models) * len(representations) * len(subsets)
    logger.info(f"Broad Random-15 evaluations: {total_random_runs} runs ({len(target_models)} models x {len(representations)} reps x {len(subsets)} subsets)")

    if args.fresh:
        logger.info("Executing fresh benchmark: existing Random-15 cache will be recomputed.")
    else:
        logger.info(f"Preserving existing Random-15 cache entries in {cache_random_dir} where available.")

    # =========================================================================
    # PHASE 1: Broad Random-15 Evaluation (factorial runs)
    # =========================================================================
    random_15_records = []
    seen_combinations = set()
    duplicate_records = []
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
            masking_strategy="random_15", seed=42, max_docs=min(args.max_docs, 200), max_length=256, batch_size=16,
            masking_mode=internal_mode
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
                    exp_key = (eval_record.get("clean_model_name", clean_model), rep, sub)
                    if exp_key in seen_combinations:
                        logger.error(f"Duplicate evaluation key encountered in cache: {exp_key}")
                        duplicate_records.append(exp_key)
                    seen_combinations.add(exp_key)
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
                ][:args.max_docs]

                if not target_docs:
                    logger.warning(f"Representation '{rep}' has 0 matching documents for subset '{sub}'. Skipping.")
                    continue

                # Deterministic reproducible seed per configuration
                cell_seed = stable_seed(model_name, rep, sub, "random_15")

                logger.info(f"[{run_count + 1}/{total_random_runs}] Evaluating {clean_model} | Rep: {rep} | Subset: {sub} | Docs: {len(target_docs)}")

                eval_res = evaluate_model_on_docs(
                    model, tokenizer, target_docs, vocab_terms, device,
                    masking_strategy="random_15", seed=cell_seed, max_docs=args.max_docs, max_length=256, batch_size=16,
                    masking_mode=internal_mode
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
                        "masking_mode": resolved_masking_mode,
                        "evaluation_unit": resolved_evaluation_unit,
                        "internal_mode": internal_mode,
                        "mask_rate": 0.15,
                        "max_length": 256,
                        "evaluation_documents": len(target_docs),
                        "seed": cell_seed,
                        "tokenizer_identifier": getattr(tokenizer, "name_or_path", str(tokenizer.__class__.__name__)),
                        "model_identifier": model_name,
                        "evaluated_document_count": len(target_docs),
                        "masked_word_count": eval_res.get("word_reconstruction_summary", {}).get("total_masked_words", 0) if internal_mode in ("wwm_subword", "wwm_word") else 0,
                        "masked_token_count": eval_res.get("overall_summary", {}).get("total_masked_tokens", 0),
                        "scoring_method": "strict_word_reconstruction" if resolved_evaluation_unit == "word" else "subword_mlm_accuracy",
                        "cache_namespace": cache_random_dir.name,
                        "code_fingerprint": "TSBC-MaritimePipeline-v2.1-A01-WWM"
                    },
                    "evaluation_metrics": eval_res
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(eval_record, f, indent=2)

                exp_key = (clean_model, rep, sub)
                if exp_key in seen_combinations:
                    logger.error(f"Duplicate evaluation key encountered: {exp_key}")
                    duplicate_records.append(exp_key)
                seen_combinations.add(exp_key)

                run_count += 1
                random_15_records.append(eval_record)
                logger.info(f"[{run_count}/{total_random_runs}] Completed: {clean_model} | Rep: {rep} | Subset: {sub} | Top1: {maritime_top1:.4f}")

        if eval_record is not None:
            with open(eval_out_dir / f"{clean_model}.json", "w", encoding="utf-8") as f:
                json.dump(eval_record, f, indent=2)

    if duplicate_records:
        raise RuntimeError(f"Duplicate evaluation keys detected during Phase 1: {duplicate_records}")

    expected_factorial_keys = {(clean_model_filename(m), r, s) for m in target_models for r in representations for s in subsets}
    observed_factorial_keys = {(r.get("clean_model_name", clean_model_filename(r.get("model_name", ""))), r["representation"], r["subset"]) for r in random_15_records}
    missing_factorial_keys = expected_factorial_keys - observed_factorial_keys
    if missing_factorial_keys and not args.models:
        logger.warning(f"Phase 1 has {len(missing_factorial_keys)} missing factorial cells: {sorted(missing_factorial_keys)}")

    # Copy BERT baseline to bert_mlm_evaluation.json for backward compatibility
    bert_clean = clean_model_filename("bert-base-uncased")
    bert_cache = list(cache_random_dir.glob(f"{bert_clean}__*.json"))
    if bert_cache:
        with open(bert_cache[0], "r", encoding="utf-8") as f_in, open(stage_dir / "bert_mlm_evaluation.json", "w", encoding="utf-8") as f_out:
            json.dump(json.load(f_in), f_out, indent=2)

    logger.info(f"Phase 1 complete: {len(random_15_records)} Random-15 evaluations available.")

    # =========================================================================
    # PHASE 2: Screen Cells & Select Diverse Configurations
    # =========================================================================
    all_ranked_cells, selected_cells = screen_and_select_configurations(random_15_records, num_select=4)
    logger.info(f"Phase 2: Screening {len(all_ranked_cells)} cells across {len(target_models)} models to select {len(selected_cells)} diverse configurations...")

    selection_artifact = {
        "selection_method": "Multi-attribute diversity optimization (v2.1): Primary ranking by mean Top-1 (descending) & mean MLM loss (ascending)",
        "screening_basis": f"Stage 14 Random-15 MLM benchmark results aggregated across all {len(target_models)} models",
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
    # PHASE 3: Focused Domain-Aware 15% Masking Evaluation
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
        ][:args.max_docs]

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
                    masking_strategy="domain_aware_15", seed=cell_seed, max_docs=args.max_docs, max_length=256, batch_size=16,
                    masking_mode=internal_mode
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
                        "masking_mode": resolved_masking_mode,
                        "evaluation_unit": resolved_evaluation_unit,
                        "internal_mode": internal_mode,
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
    # PHASE 4: Focused Sampled Pseudo-Log-Likelihood (PLL) Evaluation
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
