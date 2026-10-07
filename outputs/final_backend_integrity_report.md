# MaritimeBench Final Backend Forensic Integrity Report

**Project**: TSBC-MaritimePipeline-Version2.1  
**Timestamp**: 2026-10-07 14:46:00 UTC  
**Scope**: Final Backend Forensic Fixes, Statistical Governance & DeBERTa Permanent Exclusion  
**Overall Verdict**: **PASS** (17 / 17 Verification Gates Passed)

---

## Executive Summary & Acceptance Matrix

| # | Forensic Item / Evaluation Criterion | Expected Contract | Actual Backend State | Verdict |
| :-: | :--- | :--- | :--- | :-: |
| **1** | **Active Model Count** | Exactly 7 models | Dynamically resolved: 7 active candidate models | **PASS** |
| **2** | **Excluded Model Count** | Exactly 1 model | 1 model (`microsoft/deberta-v3-base`) | **PASS** |
| **3** | **Excluded Models + Reason** | Configuration-driven registry with provenance audit | Explicit `EXCLUDED_MODELS` dict in Stages 13–17; audit at `outputs/model_selection/deberta_exclusion_audit.json` | **PASS** |
| **4** | **Representation Count** | 5 representations | 5 formats (`json`, `key_value`, `mixed`, `narrative`, `template`) | **PASS** |
| **5** | **Subset Count** | 5 knowledge subsets | 5 subsets (`high`, `medium`, `low`, `balanced`, `random_baseline`) | **PASS** |
| **6** | **Benchmark Cell Count** | $7 \times 5 \times 5 = 175$ cells | Dynamically computed: 175 standard factorial cells | **PASS** |
| **7** | **Pairwise Comparisons** | $\binom{7}{2} = 21$ pairs | Dynamically computed: 21 pairwise comparisons | **PASS** |
| **8** | **Baseline Subword Mode** | Valid 175-cell cache, zero DeBERTa | 175 clean cached evaluations preserved in `outputs/stage-14/evaluations/cache/` | **PASS** |
| **9** | **WWM Word-Level Mode** | Strict word reconstruction, separate namespace | Full multi-piece grouping (`encoding.word_ids()`), strict all-piece correctness, separate namespace `cache_wwm_word/` | **PASS** |
| **10**| **Paired Effect-Size Status** | Paired Wilcoxon/rank-biserial; Cliff's delta unpaired | Kerby's $r_{\text{prb}} = (W^+ - W^-)/(W^+ + W^-)$; wins/ties/losses reported; Cliff's $\delta$ labeled strictly unpaired | **PASS** |
| **11**| **Provenance & Hash Status** | Cryptographic SHA-256 locking | `outputs/representation_provenance.json` & `.md` generated with full SHA-256 hashes for corpus, code, and representations | **PASS** |
| **12**| **Crossed Statistical Design**| Repeated-measures ANOVA / permutations | Primary: Crossed 3-way ANOVA ($7 \times 5 \times 5$) + 1,000 block permutations (`stage16_crossed_anova.csv`); Friedman secondary | **PASS** |
| **13**| **Kendall's W Concordance** | Multi-ranking concordance across conditions | Kendall's $W$ with tie corrections implemented; reports $W=0.8714$ (reps) and $W=0.9886$ (subsets) | **PASS** |
| **14**| **MECS Cohort-Independence** | Invariant loss transformation, renamed from MUI | Permanently renamed MUI $\rightarrow$ MECS; loss normalized via $1/(1 + \text{loss})$; proven invariant across cohorts | **PASS** |
| **15**| **Test Suite Status** | 100% pass rate across test suite | 30/30 unit tests pass (`test_forensic_fixes.py`, `test_notebook_orchestration.py`, `test_quality_optimizations.py`, `verify_pipeline.py`) | **PASS** |
| **16**| **Notebook Status** | Orchestration-only, dynamic planning | `MaritimeBench_Full_Pipeline.ipynb` validates exit code and file sizes without internal schema over-validation; `EXECUTION_POLICY = "ALWAYS_RERUN"` | **PASS** |
| **17**| **Stale Contamination Check** | Zero legacy 8-model / DeBERTa artifacts in active tree | Pre-exclusion artifacts safely archived in `outputs/archive/pre_deberta_exclusion/`; zero stale entries in active outputs | **PASS** |

---

## Detailed Forensic Audit Verification

### 1. Permanent DeBERTa Exclusion (Issue Resolution)
- **Mechanism**: Configuration-driven via `EXCLUDED_MODELS = {"microsoft/deberta-v3-base": {"reason": "..."}}`.
- **Elimination of Positional Truncation**: No `[:7]` slicing anywhere in active code; cohort size is dynamically derived from `[m for m in candidates if m not in EXCLUDED_MODELS]`.
- **Persistent Audit Trail**: Created [`outputs/model_selection/deberta_exclusion_audit.json`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/outputs/model_selection/deberta_exclusion_audit.json) recording model ID, exclusion status, exclusion reason, source stage, and evaluation history.
- **Active Exclusion**: Verified absence from Stage 13 candidates, Stage 14 benchmark caches, Stage 15 comparisons, Stage 16 statistical tables, and Stage 17 model selection.

### 2. A1 — Tokenizer / Subword Masking Confound (WWM & Strict Reconstruction)
- **Baseline Subword Mode**: Preserved 175 clean cell results in `outputs/stage-14/evaluations/cache/`.
- **Whole-Word Masking Mode**: Implemented in `scripts/14_mlm_evaluation.py` using `encoding.word_ids()` for fast tokenizers across WordPiece, BPE, and SentencePiece architectures without substring assumptions.
- **Strict Reconstruction Rule**: A word is counted as correct if and only if **all** predicted constituent subwords match their ground-truth token IDs.
- **Namespacing**: Strict cache isolation between `cache/`, `cache_wwm_subword/`, and `cache_wwm_word/`.

### 3. A2 — Paired Effect Size Governance
- **Defect Remediation**: Cliff's delta is an unpaired statistic and is now strictly segregated under `unpaired_statistics` as a secondary descriptive metric.
- **Primary Paired Inference**: Primary effect size is Kerby's paired rank-biserial correlation $r_{\text{prb}} = \frac{W^+ - W^-}{W^+ + W^-} \in [-1.0, +1.0]$.
- **Cell Accounting**: Explicitly outputs $\text{wins}_a$, $\text{ties}$, $\text{losses}_a$, and $\text{paired\_win\_rate}_a = \frac{\text{wins}_a + 0.5 \times \text{ties}}{N_{\text{paired}}}$ for every one of the 21 model pairs.

### 4. A5 — Representation Provenance & Hash Traceability
- **Traceability Manifest**: Published in machine-readable [`outputs/representation_provenance.json`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/outputs/representation_provenance.json) and publication markdown [`outputs/representation_provenance.md`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/outputs/representation_provenance.md).
- **Exact Lineage**: Fully traces Frozen Corpus (`maritime_corpus.txt`, SHA-256: `6a4d6958...`) $\rightarrow$ Stage 07 Normalized Documents (`clean_documents.jsonl`, SHA-256: `aaf0c6c3...`) $\rightarrow$ Generation Script (`11_corpus_representations.py`, SHA-256: `a6f54957...`) $\rightarrow$ 5 Representation JSONL artifacts.

### 5. A8 — Crossed Repeated-Measures Benchmark Design
- **Analytical Solution**: Implemented balanced 3-way crossed repeated-measures ANOVA decomposition across Model (fixed effect), Representation (blocking factor), and Subset (blocking factor).
- **Variance Decomposition**: Exported in [`outputs/stage-16/stage16_crossed_anova.csv`](file:///d:/CAIR/TSBC-MaritimePipeline-Version2.1/outputs/stage-16/stage16_crossed_anova.csv).
- **Block-Respecting Permutations**: 1,000 permutations shuffling model assignments strictly within (representation, subset) blocks to preserve experimental blocking structure.
- **Secondary Reference**: Omnibus Friedman Chi-Square test is retained as a secondary/reference analysis and clearly labeled as such.

### 6. B4 — Kendall's W for Multi-Ranking Concordance
- **Replacement**: Replaced Kendall tau with Kendall's $W$ (coefficient of concordance) for measuring ranking agreement across multiple representations and subsets.
- **Tie Handling**: Standard average-rank correction applied:
  $$W = \frac{12 S}{k^2(n^3 - n) - k \sum T}$$
- **Observed Concordance**: Across the 7 active models:
  - Representation Ranking Concordance: Kendall's $W = 0.8714$ ($\chi^2 = 26.14, p = 0.0002$)
  - Knowledge Subset Ranking Concordance: Kendall's $W = 0.9886$ ($\chi^2 = 29.66, p = 0.0000$)

### 7. B9 — Cohort-Independent Loss Normalization & MECS Renaming
- **Semantic Renaming**: Permanently retired "Maritime Understanding Index" / "MUI" in favor of **Maritime Encoder Composite Score (MECS)** across all active code, tables, and reports.
- **Transformation Formula**: Replaced sample min-max loss scaling with a fixed, monotonic, cohort-independent transformation:
  $$\tilde{x}_{\text{loss}} = \frac{1.0}{1.0 + \text{loss}}$$
- **Proof of Invariance**: Verified via unit test that model scores for models A, B, C remain mathematically identical whether model D is present or absent in the evaluated cohort.

---

## Conclusion
The MaritimeBench backend is clean, scientifically defensible, statistically rigorous, and fully compliant with all forensic requirements.
