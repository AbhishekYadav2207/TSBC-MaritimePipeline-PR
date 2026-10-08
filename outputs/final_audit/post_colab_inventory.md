# Stage-14 Downloaded Colab Artifact Inventory & Forensic Verification
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**
**Date**: October 8, 2026
**Scope**: Complete Forensic Inventory of Downloaded Primary Stage-14 Colab Execution Artifacts

---

## 1. Executive Summary

Stage 14 execution was performed in Google Colab using GPU acceleration under the strict Primary Legacy MLM protocol. All 175 primary cell records, 5 auxiliary artifacts, and 7 model-level summaries were downloaded into the local repository.

- **Primary Factorial Design**: 7 Models $\times$ 5 Representations $\times$ 5 Subsets = **175 Cells**
- **Observed Primary Cells**: **175 / 175 (100.0% Complete)**
- **Primary Cache Path**: `outputs/stage-14/evaluations/cache_legacy_subword_15/`
- **Masking Protocol**: Random 15% Subword MLM (`masking_mode: subword`, `evaluation_unit: subword`, `mask_rate: 0.15`)
- **Execution Type**: Strictly `PRIMARY` (zero secondary/WWM records detected in primary cache)

---

## 2. Cohort Dimensions & Factorial Grid

### Models Evaluated (N=7)
1. `answerdotai/ModernBERT-base` (Clean: `answerdotai_ModernBERT_base`)
2. `roberta-base` (Clean: `roberta_base`)
3. `bert-base-uncased` (Clean: `bert_base_uncased`)
4. `allenai/scibert_scivocab_uncased` (Clean: `allenai_scibert_scivocab_uncased`)
5. `nlpaueb/legal-bert-base-uncased` (Clean: `nlpaueb_legal_bert_base_uncased`)
6. `dmis-lab/biobert-base-cased-v1.2` (Clean: `dmis_lab_biobert_base_cased_v1.2`)
7. `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (Clean: `microsoft_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext`)

### Representations Evaluated (M=5)
1. `json`
2. `key_value`
3. `mixed`
4. `narrative`
5. `template`

### Subsets Evaluated (K=5)
1. `balanced_knowledge`
2. `high_knowledge`
3. `low_knowledge`
4. `medium_knowledge`
5. `random_baseline`

---

## 3. Auxiliary Artifacts Summary

| Artifact Name | Relative Path | Bytes | SHA-256 Digest | Description |
| :--- | :--- | :---: | :--- | :--- |
| `pll_results.json` | `outputs/stage-14/pll_results.json` | 27,619 | `7f41503c94df687be69d80d196e8555e09f584e03d42bcdafe267b1b3691684c` | Phase 4 Sampled PLL evaluations across 4 screened configurations $\times$ 7 models (28 runs) |
| `focused_domain_aware_results.json` | `outputs/stage-14/focused_domain_aware_results.json` | 159,196 | `3dff260710609335ef98fca206013ae0f918ee50495393dffb74b85cefc36466` | Phase 3 Domain-Aware-15 evaluations across 4 screened configurations $\times$ 7 models (28 runs) |
| `masking_comparison.json` | `outputs/stage-14/masking_comparison.json` | 32,408 | `784b80b7c72f7797ce3a5a415a782a20b78c80fbdfa7fc972b936a28cc12d083` | Controlled delta comparison between Random-15 and Domain-Aware-15 on screened cells |
| `bert_mlm_evaluation.json` | `outputs/stage-14/bert_mlm_evaluation.json` | 7,148 | `d0cff138bf5b722d37c6883baae17fec7755a15324d9c49d799049a460ba4477` | BERT-base baseline reference artifact |
| `pll_selection.json` | `outputs/stage-14/pll_selection.json` | 8,383 | `30a6e812543b59364a026e6efd557348981971775e5f5f3e9b62a67a07fc2f8b` | Phase 2 Multi-attribute diversity screening output selecting 4 focused configurations |

---

## 4. Model-Level Summary Artifacts (outputs/stage-14/evaluations/)

| Model Filename | Bytes | SHA-256 Digest |
| :--- | :---: | :--- |
| `allenai_scibert_scivocab_uncased.json` | 7,214 | `1f38e3aeb5c796b42b931e50523e1ca3fe0b2dcf9ce9eb6e282570086395b21e` |
| `answerdotai_ModernBERT_base.json` | 6,919 | `a5f4c58f00155b9e07297e68222a7f052520864eb137c35f92dc179b5bb69152` |
| `bert_base_uncased.json` | 7,149 | `d0cff138bf5b722d37c6883baae17fec7755a15324d9c49d799049a460ba4477` |
| `dmis_lab_biobert_base_cased_v1.2.json` | 7,318 | `8e3a2b7f3ea7b8d41cfbebb819cbbbe7ae9d532b95574c86ec558be85ff6a0eb` |
| `microsoft_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext.json` | 7,643 | `4ea3b1187c53d0dd3438e7ae361d710b7f80573e8fa9fbbfe99419c8f2baae0f` |
| `nlpaueb_legal_bert_base_uncased.json` | 7,353 | `1bfa46be83411b6da58309ca0840b3cb15bf61e3d749448831e5f8df6554dd7b` |
| `roberta_base.json` | 6,763 | `93bc67ec6bdf487e49fc40d3a5ca02d44747ebc7b41e3a647d6c66cf17dc261b` |

---

## 5. Verification Checklist

- [x] Primary cache directory contains exactly 175 `.json` cell records.
- [x] All 7 target models present with exactly 25 cells each.
- [x] All 5 representations present with exactly 35 cells each.
- [x] All 5 subsets present with exactly 35 cells each.
- [x] All records specify `masking_mode = "subword"` and `evaluation_unit = "subword"`.
- [x] No Whole-Word Masking (WWM) records in primary cache.
- [x] Provenance hashes and code commits embedded in all records.
- [x] Full machine-readable manifest stored in `outputs/final_audit/post_colab_inventory.json`.
