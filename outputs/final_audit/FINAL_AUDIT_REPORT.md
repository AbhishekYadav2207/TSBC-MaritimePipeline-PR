# Final Forensic Audit Report: Post-Colab Validation & End-to-End Pipeline Verification
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**  
**Date**: October 8, 2026  
**Final Verdict**: `POST_COLAB_FORENSICALLY_VERIFIED`

---

## 1. Executive Summary & Forensic Determination
Following execution of the primary Stage 14 Masked Language Modeling benchmark in Google Colab (GPU environment), all primary results, auxiliary benchmarks, and evaluation logs were downloaded into the repository. 

A comprehensive, 25-phase forensic audit was conducted locally **without re-running Stage 14**. Every calculation, formula, metric transformation, aggregate ranking, and statistical test was audited independently from raw primary cell files. All downstream stages (Stage 15, Stage 16, Stage 17, Stage 18) were regenerated locally exclusively from the validated primary Stage 14 cache.

### Core Findings & Audit Summary:
1. **Primary Cache Integrity**: Exactly 175 primary JSON cell records ($7 \text{ models} \times 5 \text{ representations} \times 5 \text{ subsets}$) validated in `outputs/stage-14/evaluations/cache_legacy_subword_15/`. Zero schema violations, zero duplicates, zero missing cells.
2. **Protocol Adherence**: 100% of cells adhere to the primary protocol (`random_15`, `subword` masking, `subword` reconstruction, mask rate $0.15$, seed $42 + \text{cell\_index}$). WWM records are completely isolated in `cache_wwm_word_secondary/` with zero cross-contamination.
3. **Independent Calculation Match**: Independent reimplementation of all aggregations confirmed exact integer matches on token counts and floating-point differences $\le 10^{-7}$ (well within the $10^{-6}$ audit requirement).
4. **Statistical Rigor**: 3-way crossed repeated-measures ANOVA correctly accounts for repeated conditions (Model accounts for 55.66% variance, Representation 25.47%, Interaction 15.34%). Friedman omnibus test yields $\chi^2 = 121.17, df = 6, p = 9.27 \times 10^{-24}$.
5. **Pairwise Dominance**: Wilcoxon signed-rank tests with Holm-Bonferroni correction confirm `answerdotai/ModernBERT-base` significantly outperforms all 6 competitors across 25 matched conditions ($p_{\text{holm}} \le 0.000229$, paired rank-biserial $r_{\text{prb}} \ge 0.81$, matched wins: 20/25 vs RoBERTa = 80.0%, 23/25 vs LegalBERT = 92.0%, 25/25 vs others = 100%).
6. **MECS & Pareto Verification**: Baseline MECS independently reproduced (`ModernBERT`: 87.40, `RoBERTa`: 75.38, `SciBERT`: 62.06, `BERT`: 58.74, `LegalBERT`: 57.65, `BioBERT`: 48.06, `PubMedBERT`: 42.15). Multi-objective Pareto frontier verified (5 models non-dominated, 2 dominated).
7. **DAPT Parent-Child Comparison & Leakage**: Controlled comparison of `MaritimeBERT-v1` vs parent `ModernBERT-base` shows domain adaptation gains on maritime vocabulary (20/25 wins = 80.0%). Data split audit reveals 0 exact string duplicates, 16.76% scaffold-reduced near-duplicates, and 58.48% template scaffolding.
8. **Test Suite**: 73 of 73 tests passed (100% pass rate).

---

## 2. Observed Benchmark Results (Section A)

### Primary Cross-Model Benchmark Leaderboard (Stage 15 Macro-Averages over 25 Matched Conditions)
| Rank | Model Name | Maritime Top-1 (%) | Overall Top-1 (%) | MLM Loss | PPPL | MECS (Baseline) | Latency (ms) | Fragmentation (%) | Pareto Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `answerdotai/ModernBERT-base` | **57.99%** | **67.42%** | **2.1937** | **13.72** | **87.40** | 28.7 | 62.99% | Pareto-Optimal |
| **2** | `roberta-base` | 46.12% | 59.88% | 2.8720 | 25.99 | 75.38 | 23.4 | 64.78% | Pareto-Optimal |
| **3** | `allenai/scibert_scivocab_uncased` | 32.39% | 46.22% | 4.2804 | 100.86 | 62.06 | 20.4 | 42.09% | Pareto-Optimal |
| **4** | `bert-base-uncased` | 28.08% | 42.15% | 4.7544 | 158.81 | 58.74 | **19.5** | **26.57%** | Pareto-Optimal |
| **5** | `nlpaueb/legal-bert-base-uncased` | 27.96% | 41.80% | 4.4779 | 119.58 | 57.65 | 20.8 | 37.61% | Pareto-Optimal |
| **6** | `dmis-lab/biobert-base-cased-v1.2` | 23.55% | 37.45% | 5.1486 | 237.94 | 48.06 | 20.8 | 35.82% | Dominated |
| **7** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 20.29% | 33.10% | 5.8753 | 480.95 | 42.15 | 20.6 | 42.69% | Dominated |

### Multi-Condition Concordance (Kendall's $W$)
- **Representation Concordance**: $W_{\text{rep}} = 0.7943$ ($\chi^2 = 23.83, df = 6, p = 0.00056$).
- **Subset Concordance**: $W_{\text{sub}} = 0.9829$ ($\chi^2 = 29.49, df = 6, p = 0.000049$).

---

## 3. Independently Verified Results (Section B)

Every primary metric was recalculated independently in `scripts/audit_independent_calculations.py` without importing production pipeline aggregation logic:
- **Macro-Averaging**: Floating-point difference $|\text{production} - \text{independent}| \le 7.1 \times 10^{-8}$.
- **PPPL Lineage**: Independently confirmed that PPPL is evaluated as $\exp(\text{mean NLL})$, not $\text{mean}(\exp(\text{NLL}))$.
- **MECS Formula (Eq. 2)**: Independently reproduced using fixed cohort-independent loss normalization $[3.0, 8.0]$.
- **Friedman Omnibus Test**: Independently reproduced $\chi^2 = 121.1657, df = 6, p = 9.27 \times 10^{-24}$.
- **Paired Wilcoxon & Holm Step-Down**: 21 pairwise tests independently computed; confirmed ModernBERT experiences 0 statistically significant pairwise defeats.
- **Null Metric Audit**: 217 legitimate null values across 175 cells (175 overall Top-5 in subword mode, 14 rare target slices with 0 tokens). Zero instances of `null -> 0.0` conversion found.

---

## 4. Methodological Limitations (Section C)
As formally documented in `docs/LIMITATIONS.md`:
1. **Intrinsic vs Extrinsic Transfer**: Superior intrinsic MLM performance on maritime text does not guarantee downstream fine-tuning superiority on specialized classification or NER tasks.
2. **Tokenizer Fragmentation & Sibling Leakage**: BPE tokenizers fragment maritime terms into 2–3 subwords (63–65% fragmentation). Subword masking may allow sibling context leakage. The secondary Whole-Word Masking benchmark (`cache_wwm_word_secondary`) provides a complementary evaluation.
3. **Architectural & Pretraining Confounders**: ModernBERT's performance advantage reflects larger pretraining corpus (2T+ tokens), rotary embeddings, and architectural recency in addition to domain expressiveness.
4. **Template Scaffolding & Leakage**: 58.48% template scaffolding in accident reports inflates n-gram overlap between train and test splits (56.7% shingle overlap), though exact string duplicates are 0%.
5. **DAPT Selection Regret**: Computational budget permitted DAPT on ModernBERT only; alternative adaptations on RoBERTa or BERT-base remain untested.

---

## 5. Unresolved Issues & Disposition (Section D)
All 30 pre-submission review issues (A1–A10, B1–B9, C1–C11) have been dispositioned in `docs/REVIEW_30_ISSUES_RESOLUTION.md`:
- **RESOLVED**: 24 issues.
- **DOCUMENTED_LIMITATION**: 6 issues (A1: WWM primary vs secondary; A9: template scaffolding leakage; A10: single DAPT initialization / selection regret; B9: fixed bounded normalization limits; C1: single annotator vocabulary; C6: sequence packing across documents).
- **PARTIALLY_RESOLVED**: 0 issues.
- **UNRESOLVED**: 0 issues.

---

## 6. Acceptance Checklist
- [x] 175/175 primary Stage14 cells validated
- [x] All 7 models present
- [x] All 5 representations present
- [x] All 5 subsets present
- [x] Primary protocol correct (`random_15`, `subword`, `0.15`)
- [x] WWM isolated in secondary namespace (`cache_wwm_word_secondary/`)
- [x] Stage14 provenance verified
- [x] Independent calculations match ($\le 10^{-7}$)
- [x] Stage15 independently audited
- [x] MECS independently reproduced
- [x] PPPL audited
- [x] Kendall's $W$ audited
- [x] Friedman $df=6$ verified
- [x] Paired and unpaired statistics separated
- [x] Holm-Bonferroni correction verified
- [x] Pareto directions verified
- [x] Stage17 decision independently reproduced
- [x] DAPT parent-child comparison controlled
- [x] DAPT win/loss/tie arithmetic verified (20/25 = 80.0%, never 100%)
- [x] Current split leakage audit completed
- [x] Paper data lineage generated (`outputs/final_audit/paper_data_lineage.csv`)
- [x] All 30 review issues dispositioned
- [x] No unexplained report contradictions
- [x] 73/73 test suite tests pass
- [x] Historical cache preserved (`outputs/stage-14/evaluations/cache/`)
- [x] Primary and secondary caches isolated

**FINAL FORENSIC STATUS: VERIFIED**
