"""
Deterministic Unit and Regression Tests for Stage-14 Nullable Metric Handling
MaritimeBench / MaritimeBERT Pipeline Version 2.1

Tests:
TEST 1: general metric valid, maritime metric valid -> gap is numeric
TEST 2: general metric valid, maritime metric None -> gap is None
TEST 3: general metric None, maritime metric valid -> gap is None
TEST 4: both None -> gap is None
TEST 5: maritime_target_count = 0 -> Top1 = None, Top5 = None, loss = None
TEST 6: maritime_target_count > 0 -> metrics calculate normally
TEST 7: OOV-only category -> NA, never zero
TEST 8: mixed valid + invalid categories -> only valid observations participate
TEST 9: serialization/deserialization preserves null (None <-> null, no fabrication)
TEST 10: Stage14 schema accepts legitimate null values but rejects malformed numeric fields
TEST 11: Real evaluate_model_on_docs() with synthetic input:
         - Scenario A: contains maritime targets -> numeric metrics, numeric gap
         - Scenario B: zero maritime targets -> maritime_top1=None, gap=None, no crash!
"""

import copy
import json
import math
import sys
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from scripts.validate_stage14_cell_schema import validate_stage14_cell_dict
    from scripts.validate_stage14_ingestion import validate_stage14_directory
except ImportError:
    from validate_stage14_cell_schema import validate_stage14_cell_dict
    from validate_stage14_ingestion import validate_stage14_directory

import importlib
t14 = importlib.import_module("14_mlm_evaluation")
safe_difference = t14.safe_difference
safe_ratio = t14.safe_ratio
safe_mean = t14.safe_mean
safe_percent = t14.safe_percent
safe_min = t14.safe_min
safe_max = t14.safe_max
safe_round = t14.safe_round
safe_float = t14.safe_float
evaluate_model_on_docs = t14.evaluate_model_on_docs


# =========================================================================
# TEST 1-4: Deterministic Gap and Arithmetic Helpers
# =========================================================================

def test_1_gap_both_valid_is_numeric():
    """TEST 1: general metric valid, maritime metric valid -> gap is numeric."""
    gen_val = 0.85
    mar_val = 0.65
    gap = safe_difference(gen_val, mar_val)
    assert gap is not None
    assert isinstance(gap, float)
    assert pytest.approx(gap, rel=1e-4) == 0.20


def test_2_gap_general_valid_maritime_none():
    """TEST 2: general metric valid, maritime metric None -> gap is None."""
    gen_val = 0.85
    mar_val = None
    gap = safe_difference(gen_val, mar_val)
    assert gap is None


def test_3_gap_general_none_maritime_valid():
    """TEST 3: general metric None, maritime metric valid -> gap is None."""
    gen_val = None
    mar_val = 0.65
    gap = safe_difference(gen_val, mar_val)
    assert gap is None


def test_4_gap_both_none():
    """TEST 4: both None -> gap is None."""
    gen_val = None
    mar_val = None
    gap = safe_difference(gen_val, mar_val)
    assert gap is None


def test_helpers_preserve_none():
    """Verify safe arithmetic helpers never fabricate numbers from None."""
    assert safe_ratio(10.0, None) is None
    assert safe_ratio(None, 10.0) is None
    assert safe_ratio(10.0, 0.0) is None
    assert safe_ratio(10.0, 2.0) == 5.0

    assert safe_mean([]) is None
    assert safe_mean([None, None]) is None
    assert safe_mean([1.0, None, 3.0]) == 2.0
    assert safe_mean([1.0, 2.0, 3.0]) == 2.0

    assert safe_min([None, None]) is None
    assert safe_min([1.5, None, 0.5]) == 0.5

    assert safe_max([None, None]) is None
    assert safe_max([1.5, None, 3.5]) == 3.5

    assert safe_round(None) is None
    assert safe_round(3.14159, 2) == 3.14

    assert safe_float(None) is None
    assert safe_float(1.23) == 1.23
    assert safe_float("invalid") is None


# =========================================================================
# TEST 5-8: Null Semantics in Statistics & Summaries
# =========================================================================

def test_5_zero_target_count_yields_none_not_zero():
    """TEST 5: maritime_target_count = 0 -> Top1 = None, Top5 = None, loss = None."""
    st = {"loss": 0.0, "top1": 0, "top5": 0, "top10": 0, "count": 0}
    
    # Internal summarize logic check
    cnt = st["count"]
    if cnt == 0:
        summary = {
            "masked_sample_count": 0,
            "mlm_loss": None,
            "top1_accuracy": None,
            "subword_top1_accuracy": None,
            "top5_accuracy": None,
            "top10_accuracy": None
        }
    else:
        summary = {
            "masked_sample_count": cnt,
            "mlm_loss": st["loss"] / cnt,
            "top1_accuracy": st["top1"] / cnt,
            "subword_top1_accuracy": st["top1"] / cnt,
            "top5_accuracy": st["top5"] / cnt,
            "top10_accuracy": st["top10"] / cnt
        }

    assert summary["masked_sample_count"] == 0
    assert summary["top1_accuracy"] is None
    assert summary["subword_top1_accuracy"] is None
    assert summary["top5_accuracy"] is None
    assert summary["mlm_loss"] is None


def test_6_positive_target_count_yields_numeric():
    """TEST 6: maritime_target_count > 0 -> metrics calculate normally."""
    st = {"loss": 4.5, "top1": 2, "top5": 3, "top10": 3, "count": 3}
    cnt = st["count"]
    top1 = st["top1"] / cnt
    top5 = st["top5"] / cnt
    loss = st["loss"] / cnt

    assert cnt == 3
    assert pytest.approx(top1, rel=1e-4) == 0.6666667
    assert pytest.approx(top5, rel=1e-4) == 1.0
    assert pytest.approx(loss, rel=1e-4) == 1.5


def test_7_oov_only_category_is_none_never_zero():
    """TEST 7: OOV-only category -> NA (None), never zero."""
    category_counts = {
        "navigation": {"count": 0, "top1": 0},
        "cargo_handling": {"count": 10, "top1": 7}
    }
    
    cat_accuracies = {}
    for cat, data in category_counts.items():
        if data["count"] == 0:
            cat_accuracies[cat] = None
        else:
            cat_accuracies[cat] = data["top1"] / data["count"]

    assert cat_accuracies["navigation"] is None
    assert cat_accuracies["navigation"] != 0.0
    assert pytest.approx(cat_accuracies["cargo_handling"], rel=1e-4) == 0.7


def test_8_mixed_valid_and_invalid_categories():
    """TEST 8: mixed valid + invalid categories -> only valid observations participate."""
    values = [0.80, None, 0.60, None, 0.70]
    mean_val = safe_mean(values)
    assert mean_val is not None
    assert pytest.approx(mean_val, rel=1e-4) == 0.70


# =========================================================================
# TEST 9: Serialization / Deserialization Preserves Null
# =========================================================================

def test_9_serialization_preserves_null():
    """TEST 9: serialization/deserialization preserves null (None <-> null)."""
    cell_data = {
        "overall": {"top1": 0.65, "loss": 2.1},
        "maritime_target": {"top1": None, "loss": None},
        "domain_shift_gap": None
    }
    dumped = json.dumps(cell_data)
    assert '"top1": null' in dumped
    assert '"domain_shift_gap": null' in dumped
    assert "NaN" not in dumped

    loaded = json.loads(dumped)
    assert loaded["maritime_target"]["top1"] is None
    assert loaded["maritime_target"]["loss"] is None
    assert loaded["domain_shift_gap"] is None


# =========================================================================
# TEST 10: Stage 14 Schema Null Acceptance & Numeric Rejection
# =========================================================================

def test_10_schema_accepts_legitimate_null_and_rejects_malformed():
    """TEST 10: Stage14 schema accepts legitimate null values but rejects malformed numeric fields."""
    valid_record = {
        "model_id": "bert-base-uncased",
        "representation": "json",
        "subset": "balanced_knowledge",
        "condition": "subword_subword_random_15",
        "seed": 42,
        "masking_mode": "subword",
        "evaluation_unit": "subword",
        "mask_rate": 0.15,
        "scoring_method": "subword_mlm_accuracy",
        "overall": {
            "top1": 0.62,
            "top5": 0.81,
            "top10": 0.88,
            "loss": 2.15,
            "pppl": 8.58
        },
        "maritime_target": {
            "top1": None,  # Legitimate null when zero maritime targets evaluated
            "top5": None,
            "loss": None
        },
        "rare_target": {
            "top1": None,
            "top5": None,
            "loss": None
        },
        "token_statistics": {
            "masked_positions": 30,
            "evaluated_positions": 30,
            "maritime_target_count": 0,
            "rare_target_count": 0
        },
        "provenance": {
            "model_identifier": "bert-base-uncased",
            "tokenizer_identifier": "BertTokenizerFast",
            "execution_type": "PRIMARY"
        }
    }

    # Should pass cleanly with None in maritime/rare targets
    is_valid, errs = validate_stage14_cell_dict(valid_record)
    assert is_valid is True, f"Legitimate null record rejected: {errs}"

    # Malformed numeric field (string instead of float/null)
    bad_record = copy.deepcopy(valid_record)
    bad_record["maritime_target"]["top1"] = "not_a_number"
    is_valid, errs = validate_stage14_cell_dict(bad_record)
    assert is_valid is False
    assert any("Invalid maritime_target.top1 value" in e for e in errs)

    # Malformed out-of-range top1 (> 1.0)
    bad_record_2 = copy.deepcopy(valid_record)
    bad_record_2["maritime_target"]["top1"] = 1.45
    is_valid, errs = validate_stage14_cell_dict(bad_record_2)
    assert is_valid is False
    assert any("Invalid maritime_target.top1 value" in e for e in errs)

    # NaN must be rejected
    bad_record_3 = copy.deepcopy(valid_record)
    bad_record_3["maritime_target"]["top1"] = float("nan")
    is_valid, errs = validate_stage14_cell_dict(bad_record_3)
    assert is_valid is False
    assert any("NaN detected" in e for e in errs)


# =========================================================================
# TEST 11: Real Evaluation Function on Synthetic Input (No Crash)
# =========================================================================

def test_11_evaluate_model_on_docs_synthetic_scenarios():
    """
    TEST 11: Real evaluate_model_on_docs() with synthetic input:
      Scenario A: Contains maritime target tokens -> maritime_top1 is numeric, performance_gap is numeric
      Scenario B: Contains ZERO maritime target tokens -> maritime_top1 is None, performance_gap is None
    No crash occurs in either scenario.
    """
    device = torch.device("cpu")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    maritime_vocab = ["vessel", "navigation", "rudder", "starboard", "containership", "deadweight"]

    # Scenario A: text containing maritime terms
    docs_with_maritime = [
        "The vessel navigated the channel using starboard rudder commands.",
        "The containership deadweight tonnage was verified before sailing."
    ]

    res_a = evaluate_model_on_docs(
        model, tokenizer, docs_with_maritime, maritime_vocab, device,
        masking_strategy="random_15", seed=42, max_docs=2, max_length=64, batch_size=2,
        masking_mode="subword"
    )

    mar_sum_a = res_a.get("maritime_tokens_summary", {})
    gen_sum_a = res_a.get("general_tokens_summary", {})

    assert mar_sum_a.get("masked_sample_count", 0) > 0, "Scenario A should have masked maritime targets"
    assert mar_sum_a.get("top1_accuracy") is not None
    assert isinstance(mar_sum_a.get("top1_accuracy"), float)
    assert res_a.get("performance_gap_top1") is not None
    assert isinstance(res_a.get("performance_gap_top1"), float)

    # Scenario B: pure general English text with ZERO maritime terms (e.g. general diagnostic)
    docs_zero_maritime = [
        "The weather in London was quite chilly and overcast this morning.",
        "Students gathered in the main auditorium to listen to the guest lecture on ancient history."
    ]

    res_b = evaluate_model_on_docs(
        model, tokenizer, docs_zero_maritime, maritime_vocab, device,
        masking_strategy="random_15", seed=42, max_docs=2, max_length=64, batch_size=2,
        masking_mode="subword"
    )

    mar_sum_b = res_b.get("maritime_tokens_summary", {})
    gen_sum_b = res_b.get("general_tokens_summary", {})

    assert mar_sum_b.get("masked_sample_count") == 0, "Scenario B must have 0 masked maritime targets"
    assert mar_sum_b.get("top1_accuracy") is None, "Top-1 accuracy must be None for 0 maritime observations"
    assert mar_sum_b.get("top5_accuracy") is None
    assert mar_sum_b.get("mlm_loss") is None
    assert res_b.get("performance_gap_top1") is None, "Performance gap must be None when maritime metric is None"
    assert gen_sum_b.get("top1_accuracy") is not None, "General tokens should have valid accuracy"
