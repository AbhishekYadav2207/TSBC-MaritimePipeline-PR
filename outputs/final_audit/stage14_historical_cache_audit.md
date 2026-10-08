# Historical Stage-14 Primary Cache Forensic Audit Report
**Audit Target**: `outputs/stage-14/evaluations/cache/`  
**Total Records Evaluated**: 175 / 175  
**Audit Status**: **PASSED — 100% PROTOCOL COMPLIANT**  

---

## 1. Verified Protocol Specifications
* **Masking Strategy**: `random_15` (Independent 15% random subword masking)
* **Masking Mode**: `subword`
* **Evaluation Unit**: `subword` (Subword reconstruction)
* **Mask Rate**: `0.15` (15.0%)
* **Documents Per Cell**: `200`
* **Factorial Dimensions**: 7 models × 5 representations × 5 subsets = **175 complete cells**

---

## 2. Model Cohort Summary Across All 25 Cells Each
| Model Archetype | Cells | Mean Overall Top-1 | Mean Maritime Top-1 | Mean MLM Loss |
| :--- | :---: | :---: | :---: | :---: |
| `answerdotai_ModernBERT_base` | 25 | 0.5715 | 0.7282 | 2.3874 |
| `roberta_base` | 25 | 0.5046 | 0.6203 | 3.1253 |
| `nlpaueb_legal_bert_base_uncased` | 25 | 0.4284 | 0.3302 | 3.6305 |
| `allenai_scibert_scivocab_uncased` | 25 | 0.4653 | 0.3212 | 3.3996 |
| `bert_base_uncased` | 25 | 0.4507 | 0.2918 | 3.5389 |
| `dmis_lab_biobert_base_cased_v1.2` | 25 | 0.3858 | 0.2516 | 3.9590 |
| `microsoft_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext` | 25 | 0.4105 | 0.2014 | 4.1091 |

---

## 3. Protocol Integrity Verification
* **Protocol Violations Detected**: 0
* **Integrity Status**: All 175 files strictly encode the intended legacy random 15% subword MLM evaluation.
* **Preservation Notice**: This cache directory is frozen and serves as the authoritative baseline reference for regression comparison against the post-Colab regenerated run.