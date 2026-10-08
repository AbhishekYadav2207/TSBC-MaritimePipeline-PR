# Final Forensic Audit Report: Post-Colab Validation & End-to-End Pipeline Verification
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**  
**Date**: October 8, 2026  
**Final Forensic Determination**: `POST_COLAB_FORENSICALLY_VERIFIED`

---

## 1. Executive Summary & Verification Overview
Following execution of the primary Stage 14 Masked Language Modeling benchmark in Google Colab (GPU environment), all primary evaluation records, auxiliary benchmarks, and logs were downloaded into the repository. 

A forensic audit and report-integrity repair were conducted locally **without re-running Stage 14**. The downloaded primary cache ([`cache_legacy_subword_15/`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/outputs/stage-14/evaluations/cache_legacy_subword_15/)) was treated as immutable scientific input. All downstream stages (Stage 15, Stage 16, Stage 17, Stage 18) were regenerated locally exclusively from this primary cache.

This final report formally distinguishes:
1. **PRIMARY VERIFIED RESULTS** (empirically confirmed calculations and benchmark data)
2. **REPORT & DOCUMENTATION CORRECTIONS** (discrepancy resolutions and formula reconciliations)
3. **REMAINING SCIENTIFIC LIMITATIONS** (acknowledged experimental and computational boundaries)

---

## 2. Primary Verified Results

### A. Primary Benchmark Ingestion & Completeness
* **Location**: `outputs/stage-14/evaluations/cache_legacy_subword_15/`
* **Cell Completeness**: **Exactly 175 JSON cell records** ($7 \text{ models} \times 5 \text{ representations} \times 5 \text{ subsets} = 175 \text{ cells}$).
* **Cell Protocol**: Masking strategy `random_15`, subword masking, subword evaluation unit, mask rate $0.15$, deterministic seed $42 + \text{cell\_index}$, execution type `PRIMARY`.
* **Historical Regression vs `cache/`**: 175/175 cells classified as `KNOWN_BUG_FIX` due to Byte-Level BPE space prefixing and non-domain stopword exclusion (average 14.45 false-positive tokens removed per cell; Top-1 shift $\le +0.0025$).
* **Rejection / Isolation Checks**: 0 missing cells, 0 duplicate cells, 0 WWM records in primary cache. Whole-Word Masking records are strictly isolated in `outputs/stage-14/evaluations/cache_wwm_word_secondary/`.

### B. Independent Calculation Reimplementation
Re-implemented all calculations from raw JSON files in `scripts/audit_independent_calculations.py`:
* Maximum floating-point difference: $|\text{production} - \text{independent}| \le 7.1 \times 10^{-8}$ (requirement: $\le 10^{-6}$).
* Integer token/document counts: **100% exact match** (0 mismatches).
* Null metric handling: 217 valid nulls preserved across 175 cells without zero-fill distortion.

### C. Factorial Crossed Repeated-Measures ANOVA (`stage16_crossed_anova.csv`)
* **Model Main Effect**: $F(6, 96) = \mathbf{804.1615}$, $p = 1.11 \times 10^{-16}$, Variance Contribution = **55.66%**, $\eta_p^2 = 0.9805$.
* **Representation Effect**: $F(4, 96) = \mathbf{552.0302}$, $p = 1.11 \times 10^{-16}$, Variance Contribution = **25.47%**, $\eta_p^2 = 0.9583$.
* **Subset Effect**: $F(4, 96) = \mathbf{15.4679}$, $p = 8.46 \times 10^{-10}$, Variance Contribution = **0.71%**, $\eta_p^2 = 0.3919$.
* **Model $\times$ Representation**: $F(24, 96) = \mathbf{55.3965}$, $p = 1.11 \times 10^{-16}$, Variance Contribution = **15.34%**, $\eta_p^2 = 0.9327$.
* **Model $\times$ Subset**: $F(24, 96) = \mathbf{2.1247}$, $p = 0.00533$, Variance Contribution = **0.59%**, $\eta_p^2 = 0.3469$.
* **Representation $\times$ Subset**: $F(16, 96) = \mathbf{6.0638}$, $p = 5.25 \times 10^{-9}$, Variance Contribution = **1.12%**, $\eta_p^2 = 0.5026$.
* **Residual Variance**: **1.11%**.

### D. Friedman Omnibus & Pairwise Inference
* **Friedman Test**: $\chi^2 = \mathbf{121.1657}$, $df = \mathbf{6}$ ($k=7$ models), $p = 9.27 \times 10^{-24}$, $W = 0.8078$.
* **ModernBERT vs RoBERTa Dominance**:
  * Matched conditions: 25.
  * Paired wins: **20**, losses: **5**, ties: **0** $\implies$ **80.0% win rate**.
  * Wilcoxon signed-rank: $W = 33.0$, $p_{\text{holm}} = \mathbf{0.000229}$.
  * Paired rank-biserial effect size: $r_{\text{prb}} = \mathbf{0.8585}$.
  * Unpaired Cliff's delta across 625 cell pairs: $\delta = \mathbf{0.5104}$.
* **Total Pairwise Significant Defeats for ModernBERT**: **0**.

### E. Primary Cross-Model Benchmark Leaderboard
| Rank | Model Name | Maritime Top-1 (%) | Overall Top-1 (%) | MLM Loss | Primary MLM PPL ($\exp(\bar{L})$) | Sampled PLL PPPL | Baseline MECS | Fragmentation (%) | Pareto Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `answerdotai/ModernBERT-base` | **57.99%** | **67.42%** | **2.1937** | **8.97** | **13.72** | **87.40** | 62.99% | Pareto-Optimal |
| **2** | `roberta-base` | 46.12% | 59.88% | 2.8720 | 17.67 | 16.02 | **87.04** | 64.78% | Pareto-Optimal |
| **3** | `allenai/scibert_scivocab_uncased` | 32.39% | 46.22% | 4.2804 | 72.27 | 36.58 | **82.19** | 42.09% | Pareto-Optimal |
| **4** | `bert-base-uncased` | 28.08% | 42.15% | 4.7544 | 116.09 | 25.50 | **78.00** | **26.57%** | Pareto-Optimal |
| **5** | `nlpaueb/legal-bert-base-uncased` | 27.96% | 41.80% | 4.4779 | 88.05 | 42.10 | **77.05** | 37.61% | Pareto-Optimal |
| **6** | `dmis-lab/biobert-base-cased-v1.2` | 23.55% | 37.45% | 5.1486 | 172.19 | 91.85 | **69.01** | 35.82% | Dominated |
| **7** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 20.29% | 33.10% | 5.8753 | 356.14 | 91.56 | **58.78** | 42.69% | Dominated |

---

## 3. Report & Documentation Corrections

### A. MECS Authoritative Formula & Normalization
* **Configuration Source**: `outputs/stage-15/stage15_mecs_config.json`.
* **Exact Baseline Formula**:
  $$ \text{MECS}_{\text{baseline}} = 100 \times \left[ 0.40 \cdot S_{\text{top1}} + 0.15 \cdot S_{\text{rare}} + 0.25 \cdot S_{\text{loss}} + 0.20 \cdot S_{\text{frag}} \right] $$
  *(Note: Top-5 is omitted in baseline scenario per configuration).*
* **Loss Transform Clipping Behavior**:
  * Fixed bounds: $[3.0, 8.0]$.
  * Transform formula: $S_{\text{loss}} = \text{clip}\left(\frac{8.0 - \text{Loss}}{8.0 - 3.0}, 0.0, 1.0\right)$.
  * ModernBERT Loss = 2.1937 $\implies \frac{8.0 - 2.1937}{5.0} = 1.1613 \implies \text{clip}(1.1613, 0.0, 1.0) = \mathbf{1.0}$.
* **ModernBERT Top-1 Metric Lineage**:
  * Authoritative model-level macro-average: **0.5799 (57.99%)** (unweighted macro-average across all 25 primary cells in `leaderboard.csv` / `stage15_model_profiles.csv`).
  * Disambiguation of 0.5814: $0.5814$ ($0.581448$) is the Top-1 accuracy of a **single individual condition cell** (cell row 0 in `comparison.csv`: `representation = json, subset = balanced_knowledge`), NOT a model-level aggregate.
  * Both values clip to $1.0$ under the $0.35$ ceiling ($S_{\text{top1}} = 1.0$), ensuring the MECS baseline score ($87.40$) is mathematically invariant.
* **Manual Verification of ModernBERT Baseline MECS**:
  $$ \text{MECS} = 100 \times [0.40(1.0) + 0.15(1.0) + 0.25(1.0) + 0.20(0.3701)] = 100 \times [0.40 + 0.15 + 0.25 + 0.07402] = \mathbf{87.40} $$

### B. PPPL Value Reconciliation
Divergences in reported PPPL numbers across draft documents have been formally resolved:
1. **Primary MLM Perplexity** ($\text{PPL}_{\text{MLM}} = \exp(\bar{L}_{\text{MLM}})$): Evaluated from the 175-cell benchmark cross-entropy loss. ModernBERT = **8.97** (cell macro-mean = 9.32).
2. **Auxiliary PLL Pseudo-Perplexity** ($\text{PPPL}_{\text{PLL}}$): Evaluated from token-by-token pseudo-log-likelihood on sampled documents (`pll_results.json`). ModernBERT = **13.72**.
Both values are scientifically valid measures of distinct tasks and are now explicitly labeled.

### C. ANOVA F-Statistics Alignment
Stale preliminary numbers ($F = 368.04, 22.18$) have been purged. All reports now cite the true regenerated values:
- Model $F = 804.1615$
- Representation $F = 552.0302$
- Subset $F = 15.4679$
- Model $\times$ Representation $F = 55.3965$

### D. 30-Issue Matrix Reconstruction
All 30 issues from the authoritative pre-repair review (`review_issue_matrix_pre_repair.csv`) have been rebuilt into [`review_issue_resolution_matrix.csv`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/outputs/final_audit/review_issue_resolution_matrix.csv) and [`docs/REVIEW_30_ISSUES_RESOLUTION.md`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/docs/REVIEW_30_ISSUES_RESOLUTION.md):
- **RESOLVED**: 23 issues.
- **DOCUMENTED_LIMITATION**: 7 issues (A1, A7, A9, A10, B9, C1, C6).
- **PARTIALLY_RESOLVED**: 0 issues.
- **NOT_APPLICABLE**: 0 issues.

---

## 4. Remaining Scientific Limitations

1. **Intrinsic MLM Scope vs Extrinsic Downstream Transfer (Issue A1 / Section 1)**:
   Superior intrinsic MLM zero-shot cloze accuracy does not guarantee optimal fine-tuned downstream task performance (e.g. incident cause classification or named entity recognition).
2. **Tokenizer Fragmentation & Sibling Subword Context Leakage (Issue A1)**:
   Byte-Level BPE tokenizers incur 63–65% fragmentation on domain terms. Subword masking allows sibling context cues. Whole-word masking results are isolated as secondary robustness data.
3. **Pretraining Volume & Architecture Confounders (Issue A7)**:
   ModernBERT's advantage reflects trillions of pretraining tokens and modern rotary architecture, not purely domain transfer dynamics.
4. **Template Scaffolding & Evaluation Leakage (Issue A9)**:
   TSB accident reports have 58.48% template scaffolding, causing 56.7% shingle overlap across splits despite 0 exact duplicates. Slot-value masking is specified, but empirical fine-tuning evaluation on `test.txt` is deferred to GPU downstream phase.
5. **DAPT Selection Regret (Issue A10)**:
   Due to single-GPU compute constraints, DAPT was conducted exclusively on ModernBERT; adaptation of RoBERTa and BERT-base remains unmeasured.
6. **Fixed Loss Normalization Bounds (Issue B9)**:
   Fixed bounds $[3.0, 8.0]$ saturate models with loss $< 3.0$ at score $1.0$, compressing differences among ultra-low-loss architectures.

---

## 5. Acceptance Checklist

- [x] 175/175 primary Stage 14 cells validated
- [x] All 7 models, 5 representations, and 5 subsets present
- [x] Primary protocol correct (`random_15`, `subword`, `0.15`)
- [x] WWM strictly isolated in `cache_wwm_word_secondary/`
- [x] Stage 14 provenance cryptographically verified
- [x] Independent calculations match ($\le 10^{-7}$)
- [x] Stage 15 independently audited
- [x] MECS baseline formula matches actual configuration (`stage15_mecs_config.json`)
- [x] Loss clipping behavior explicitly verified and documented
- [x] PPPL values fully reconciled between MLM perplexity (8.97) and PLL (13.72)
- [x] Kendall's $W$ audited ($W_{\text{rep}} = 0.7943, W_{\text{sub}} = 0.9829$)
- [x] Friedman $df=6$ verified ($\chi^2 = 121.1657$)
- [x] Paired dominance (20/25 = 80.0%) strictly separated from unpaired Cliff's delta ($\delta = 0.5104$)
- [x] Holm-Bonferroni correction verified
- [x] Pareto directions verified (5 optimal, 2 dominated)
- [x] Stage 17 decision hierarchy independently reproduced
- [x] DAPT parent-child comparison controlled
- [x] Current split leakage audit completed (0 exact duplicates, 16.76% near-duplicates)
- [x] Paper data lineage generated (`paper_data_lineage.csv`)
- [x] Original 30 issue IDs preserved verbatim from pre-repair review
- [x] A9 correctly dispositioned as `DOCUMENTED_LIMITATION`
- [x] Report consistency verified = ZERO unexplained contradictions
- [x] All downstream tests pass (78 / 78 tests, 100% pass rate)
- [x] Stage 14 remains untouched

**FINAL FORENSIC STATUS: VERIFIED**
