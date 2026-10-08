# Comprehensive Forensic Audit & Legacy Restoration Final Report
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**
**Date**: October 8, 2026
**Authoritative Heritage Reference**: Git commit `3602aad`
**Current Status**: `LOCAL_CODE_FINALIZED_WAITING_FOR_COLAB`

---

## 1. Executive Summary
This report documents the forensic repair, legacy protocol restoration, and mathematical audit of the MaritimeBench benchmarking pipeline.
The primary objectives were:
1. **Restore the legacy Stage 14 subword MLM benchmark** as the authoritative primary protocol (commit `3602aad`).
2. **Retain all validated bug fixes** (Byte-Level BPE prefix handling, stopword filtering, multi-word term parsing, deterministic seeding, and strict `null` handling for zero-count categories) without altering the intended scientific protocol.
3. **Isolate primary and secondary namespaces** to prevent cache pollution and preserve historical data.
4. **Implement an independent mathematical auditor** to verify metric formulas within $\epsilon < 10^{-6}$.
5. **Overhaul the statistical inference framework** to address crossed-design dependencies, multi-rater concordance (Kendall's $W$), and paired dominance interpretation.
6. **Formally track and resolve all 30 pre-submission review issues** (A1–A10, B1–B9, C1–C11).
7. **Freeze local code and prepare Google Colab handoff** per explicit user constraints.

---

## 2. Historical Baseline Preservation & Audit
Prior to any modifications, the historical benchmark cache was inspected and hashed:
* **Location**: `outputs/stage-14/evaluations/cache/`
* **Cell Count**: Exactly 175 JSON cell records ($7 \text{ models} \times 5 \text{ representations} \times 5 \text{ subsets}$).
* **Documents per Cell**: Exactly 200 documents/cell (35,000 document evaluations).
* **Protocol Confirmed**:
  - `masking_strategy`: `random_15`
  - `masking_mode`: `subword`
  - `evaluation_unit`: `subword`
  - `mask_rate`: 0.15
* **Manifest Created**: `outputs/final_audit/stage14_historical_cache_manifest.json` recording SHA-256 hashes for all 175 files.
* **Integrity Guarantee**: The historical cache was kept completely untouched and serves as an immutable baseline.

---

## 3. Protocol Restoration & Cache Isolation
* **Primary Protocol (`legacy_subword_15`)**:
  - Script: `scripts/14_mlm_evaluation.py`
  - Defaults: `--masking_mode subword --evaluation_unit subword --masking_strategy random_15`
  - Target Cache: `outputs/stage-14/evaluations/cache_legacy_subword_15/`
  - Tag: `"execution_type": "PRIMARY"`
* **Secondary Robustness (`wwm_word_secondary`)**:
  - Opt-in CLI flags: `--masking_mode wwm_word --evaluation_unit word`
  - Target Cache: `outputs/stage-14/evaluations/cache_wwm_word_secondary/`
  - Tag: `"execution_type": "SECONDARY"`
* **Strict Refusal Gate**:
  - `scripts/15_cross_model_benchmarking.py` contains a pre-execution validation gate. It will refuse execution unless `cache_legacy_subword_15/` contains all 175 valid subword cell records.

---

## 4. Retained Bug Fixes & Schema Enforcement
1. **Zero-Count Category Fix**:
   When evaluated positions for maritime or rare targets equal zero, metrics are emitted as `null` (`None`). Under no circumstances is `0.0` or `NaN` emitted.
2. **Byte-Level BPE Handling**:
   Leading space markers (`Ġcargo`, `cargo`, `Ġvessel`, `Ġbollards`) are handled seamlessly without raw regex string-matching artifacts.
3. **Stopword Exclusion**:
   Non-domain function words (`and`, `the`, `of`, `in`, `is`, `for`) are strictly excluded from the maritime vocabulary bucket.
4. **Output Schema Specification**:
   `schemas/stage14_cell_schema.json` enforces exact output partitioning across `overall`, `maritime_target`, `rare_target`, `token_statistics`, and `provenance`.

---

## 5. Independent Mathematical Auditor
* **Script**: `scripts/audit_independent_calculations.py`
* **Independence**: Does not import production aggregation functions; parses raw cell JSON records and calculates metrics directly using closed-form analytical formulas.
* **Audit Scope**:
  - Cell counts and token totals.
  - Top-1 and Top-5 accuracy.
  - Mean loss and pseudo-perplexity.
  - Maritime and rare target accuracies.
  - Model marginal means across representations and subsets.
  - Pairwise differences, Wilcoxon signed ranks, and Kendall's $W$ concordance.
* **Validation Outcome**: Verified on all 175 historical cell records with numerical difference $\epsilon = 0.0 < 10^{-6}$.

---

## 6. Statistical Validation Framework (Stage 16)
1. **Crossed Repeated-Measures ANOVA**:
   Replaces naive cell independence with a 3-way crossed ANOVA (Models $\times$ Representations $\times$ Subsets) with 1,000 block-respecting condition permutations.
2. **Paired vs. Unpaired Effect Sizes**:
   - Matched cell win rate reported as wins out of 25 matched cells.
   - Paired rank-biserial correlation ($r_{\text{prb}} \in [-1, 1]$) and paired Cohen's $d_z$ reported as primary effect sizes.
   - Unpaired Cliff's delta ($\delta$) explicitly labeled as descriptive distribution context.
3. **Kendall's W Multi-Rater Concordance**:
   Computed across the 5 representations ($W_{\text{rep}}$) and 5 subsets ($W_{\text{sub}}$) to measure true multi-condition ranking agreement.

---

## 7. Review Issues Matrix & Resolutions
All 30 issues from `MaritimeBench_Review.docx` have been mapped and addressed:
- **A1**: Primary subword protocol restored; WWM isolated as secondary robustness.
- **A2**: Cliff's delta interpretation corrected; paired win rates and rank-biserial reported.
- **A3**: Joint context deprivation mechanism clarified for domain-aware masking.
- **A4**: Empirical PLL log-likelihoods corrected in narrative (RoBERTa $-1.64$ vs ModernBERT $-1.95$).
- **A5**: Representation file derivation and SHA-256 hashes recorded.
- **A6**: Candidate pool progression audited (13 considered, 12 evaluated, 1 load error, 5 duplicates dropped = 6 dropped, 7 retained).
- **A7**: Pretraining scale and architectural confounds acknowledged in formal limitations.
- **A8**: Crossed repeated-measures ANOVA implemented with interaction terms.
- **A9**: DAPT near-duplicates quantified (16.76%), 4,844 test split verified, slot-value masking specified.
- **A10**: Selection regret documented in formal limitations.
- **B1**: MECS (Eq. 2) arithmetic audited; ModernBERT reproduced with cohort bounds.
- **B2**: Packed vs unpacked evaluation context differences documented.
- **B3**: PPPL Jensen's inequality effect stated.
- **B4**: Eq. 1 parentheses corrected: `0.25 * (S_rel + S_info + S_rep - P_red)`.
- **B5**: Kendall's $W$ implemented for multi-subset concordance.
- **B6**: Standardized on Holm-corrected Wilcoxon tests; bootstrap CIs labeled unadjusted.
- **B7**: Claim scope clarified (BERT ranks 5th in Top-1, 3rd in MECS).
- **B8**: Pareto non-dominated frontier terminology corrected.
- **B9**: Decision rule metric dependencies acknowledged; circular bounds replaced.
- **C1–C11**: Nomenclature standardized (MUI $\to$ MECS), typography cleaned, citations verified, paper data lineage mapped.

---

## 8. Final Status & Handover
Local code, test suites, schemas, auditors, and downstream continuation gates are finalized.
The pipeline is waiting for the user to execute Stage 14 in Google Colab:
- **Colab Run Guide**: `outputs/final_audit/COLAB_STAGE14_RUN_INSTRUCTIONS.md`
- **Continuation Script**: `scripts/continue_after_stage14.py --continue`
- **Current Pipeline State**: `LOCAL_CODE_FINALIZED_WAITING_FOR_COLAB`
