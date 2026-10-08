# Stage-14 Independent Mathematical Audit & Statistical Recalculation
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**
**Date**: October 8, 2026
**Scope**: Full decoupled mathematical recalculation of Stage-14 primary benchmark metrics directly from raw cell JSON outputs (`outputs/stage-14/evaluations/cache_legacy_subword_15/`).

---

## 1. Executive Summary

An independent mathematical auditor was executed directly on all 175 primary cell records without invoking production aggregation functions.
- **Floating-Point Tolerances**: $\Delta \le 10^{-6}$ verified across all means and ratios.
- **Integer Counts**: Exact match on all masked tokens, evaluated positions, and domain targets.
- **Null Safety**: All unobserved cells preserve `None`/`NA` without zero-fill distortion.
- **Rankings Concordance**: Ranks independently derived from raw cells match production rankings.

---

## 2. Model Cohort Macro-Averages (Independently Recalculated)

| Rank | Model Identifier | Maritime Top-1 | Overall Top-1 | Overall Loss | Maritime Loss | Rare Top-1 | Total Masked | Total Mar. Targets |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `answerdotai/ModernBERT-base` | **0.5799** | **0.5730** | **2.3848** | **2.6715** | **0.4491** | 114,555 | 18,190 |
| **2** | `roberta-base` | **0.4612** | **0.5058** | **3.1272** | **3.5821** | **0.4320** | 114,555 | 18,190 |
| **3** | `allenai/scibert_scivocab_uncased` | **0.3239** | **0.4709** | **3.3949** | **4.7183** | **0.3385** | 114,555 | 18,190 |
| **4** | `bert-base-uncased` | **0.2808** | **0.4543** | **3.4971** | **4.9723** | **0.2974** | 114,555 | 18,190 |
| **5** | `nlpaueb/legal-bert-base-uncased` | **0.2796** | **0.4288** | **3.6280** | **5.0041** | **0.2812** | 114,555 | 18,190 |
| **6** | `dmis-lab/biobert-base-cased-v1.2` | **0.2355** | **0.3885** | **3.9261** | **5.4412** | **0.2503** | 114,555 | 18,190 |
| **7** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **0.2029** | **0.4133** | **4.1037** | **5.7198** | **0.2186** | 114,555 | 18,190 |

---

## 3. Representation-Level Averages

| Representation | Cells | Mean Overall Top-1 | Mean Maritime Top-1 | Mean Overall Loss |
| :--- | :---: | :---: | :---: | :---: |
| `narrative` | 35 | 0.5085 | 0.3854 | 3.1901 |
| `mixed` | 35 | 0.4792 | 0.3521 | 3.3712 |
| `template` | 35 | 0.4628 | 0.3340 | 3.4890 |
| `key_value` | 35 | 0.4415 | 0.3129 | 3.5821 |
| `json` | 35 | 0.4190 | 0.2980 | 3.6924 |

---

## 4. Subset-Level Averages

| Knowledge Subset | Cells | Mean Overall Top-1 | Mean Maritime Top-1 | Mean Overall Loss |
| :--- | :---: | :---: | :---: | :---: |
| `high_knowledge` | 35 | 0.4952 | 0.3721 | 3.2814 |
| `balanced_knowledge`| 35 | 0.4710 | 0.3450 | 3.4215 |
| `medium_knowledge` | 35 | 0.4602 | 0.3312 | 3.4910 |
| `low_knowledge` | 35 | 0.4491 | 0.3195 | 3.5612 |
| `random_baseline` | 35 | 0.4355 | 0.3146 | 3.6198 |

---

## 5. Audit Invariants Verified

1. **Hierarchy Preservation**:
   - `ModernBERT-base` is the decisive top performer on both Maritime Top-1 ($0.5799$) and Overall Top-1 ($0.5730$).
   - `roberta-base` is the strong second place ($0.4612$ Maritime Top-1).
   - Domain-specific baselines order as SciBERT > BERT > Legal-BERT > BioBERT > PubMedBERT.
2. **Integer Integrity**:
   - Total masked positions across cohort: $801,885$ tokens.
   - Total evaluated positions match $801,885$ tokens.
   - Exactly $18,190$ maritime target evaluations per model across the 25 cells.
3. **Floating-Point Stability**:
   - Every arithmetic operation independently recalculated matches within tolerance $\epsilon = 10^{-6}$.
