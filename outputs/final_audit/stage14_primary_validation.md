# Stage-14 Primary Benchmark Completeness & Protocol Validation
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**
**Date**: October 8, 2026
**Overall Status**: **PASS (175 / 175 Valid Cells)**

---

## 1. Executive Summary

Every downloaded Stage-14 primary evaluation record was verified against the strict primary benchmark contract:
- **Total Factorial Cells**: 175 Expected | 175 Observed | 0 Missing | 0 Duplicate
- **Protocol Enforcement**:
  - `masking_strategy = "random_15"` (175/175 Verified)
  - `masking_mode = "subword"` (175/175 Verified)
  - `evaluation_unit = "subword"` (175/175 Verified)
  - `mask_rate = 0.15` (175/175 Verified)
  - `execution_type = "PRIMARY"` (175/175 Verified)
- **Cohort Symmetry**:
  - Exactly 25 cells for each of the 7 models.
  - Exactly 35 cells for each of the 5 representations.
  - Exactly 35 cells for each of the 5 subsets.
- **Provenance Completeness**: 100% of cells contain non-empty `model_identifier`, `tokenizer_identifier`, and dataset/code hashes.

---

## 2. Model Cohort Cell Distribution

| Model Identifier | Expected Cells | Observed Cells | Valid Cells | Protocol Status |
| :--- | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 25 | 25 | 25 | PASS |
| `roberta-base` | 25 | 25 | 25 | PASS |
| `bert-base-uncased` | 25 | 25 | 25 | PASS |
| `allenai/scibert_scivocab_uncased` | 25 | 25 | 25 | PASS |
| `nlpaueb/legal-bert-base-uncased` | 25 | 25 | 25 | PASS |
| `dmis-lab/biobert-base-cased-v1.2` | 25 | 25 | 25 | PASS |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 25 | PASS |
| **Cohort Total** | **175** | **175** | **175** | **100% PASS** |

---

## 3. Representation & Subset Symmetry

| Factor | Level | Expected Cells | Observed Valid Cells |
| :--- | :--- | :---: | :---: |
| **Representation** | `json` | 35 | 35 |
| **Representation** | `key_value` | 35 | 35 |
| **Representation** | `mixed` | 35 | 35 |
| **Representation** | `narrative` | 35 | 35 |
| **Representation** | `template` | 35 | 35 |
| **Subset** | `balanced_knowledge` | 35 | 35 |
| **Subset** | `high_knowledge` | 35 | 35 |
| **Subset** | `low_knowledge` | 35 | 35 |
| **Subset** | `medium_knowledge` | 35 | 35 |
| **Subset** | `random_baseline` | 35 | 35 |

---

## 4. Rejection Criteria Check

| Rejection Check | Threshold | Observed Count | Result |
| :--- | :---: | :---: | :---: |
| Duplicate cells | 0 | 0 | PASS |
| Missing cells | 0 | 0 | PASS |
| Extra / orphan files | 0 | 0 | PASS |
| Whole-Word Masking (WWM) records | 0 | 0 | PASS |
| Word-level evaluation unit records | 0 | 0 | PASS |
| Disallowed models | 0 | 0 | PASS |
| Mask rate $\ne 0.15$ | 0 | 0 | PASS |
| Incomplete provenance records | 0 | 0 | PASS |

---

## 5. Formal Conclusion

The primary cache at `outputs/stage-14/evaluations/cache_legacy_subword_15/` passes all completeness, schema, and protocol validation checks with zero defects.
