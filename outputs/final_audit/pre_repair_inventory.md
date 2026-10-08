# Forensic Pre-Repair Inventory Report
**Repository**: `TSBC-MaritimePipeline-Version2.1`  
**Execution Timestamp**: 2026-10-08T11:25:00+05:30  
**Execution Mode**: STRICTLY READ-ONLY FORENSIC INVENTORY  
**Current Git Commit**: `37cd3fde87cdcddb53e2dd99becbf7e54419d3bb` (branch: `main`, status: `clean`)  
**Legacy Stage-14 Historical Commit**: `3602aad`  

---

## 1. Executive Summary & Inventory Objectives
This forensic inventory captures the exact baseline state of all source code, configurations, data manifests, representations, caches, output tables, and test suites prior to initiating any repair or regeneration.

No production code, scientific outputs, caches, reports, paper files, or benchmark results were modified during this phase.

---

## 2. Git State & Commit Lineage
* **Current Working Branch**: `main`
* **Current Commit Hash**: `37cd3fde87cdcddb53e2dd99becbf7e54419d3bb`
* **Working Tree Status**: `Clean (No uncommitted changes)`
* **Historical Legacy Benchmark Commit**: `3602aad` ("Fix formatting and remove unnecessary characters in README")
  * In commit `3602aad`, Stage 14 implemented the primary legacy protocol: **Random 15% Subword MLM** with subword-level reconstruction across 175 factorial cells ($7 \text{ models} \times 5 \text{ representations} \times 5 \text{ subsets}$).
  * In subsequent commits (`37cd3fd`, `9bb3d0d`), whole-word masking (WWM) and strict word reconstruction were introduced, modifying default flags in `scripts/14_mlm_evaluation.py` and overwriting downstream tables in `outputs/stage-15/`.

---

## 3. Authoritative Corpus & Representation Integrity
* **Authoritative Corpus**: **Corpus A Only** (`data/MARSISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv`).
* **Source Tables Inventoried**:
  * `data/MARSISdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv` (Size: 96,555,062 bytes, SHA-256: `c8c30874460cf63d7efb7dfb3a7979bf15154324e27c457b7f61eb72d781b331`)
  * `data/MARSISdb_MDOTW_VW_OCCURRENCE_VESSEL_PUBLIC.csv` (Size: 77,624,653 bytes, SHA-256: `c9a66b1c7fde96f41d0115fb65d27b1b3a0d80d84667a4068a1008518a1763de`)
  * `data/MARSISdb_MDOTW_VW_OCCURRENCE_VESSEL_NAV_EQUIPMENT_PUBLIC.csv` (Size: 27,095,353 bytes, SHA-256: `bdce3c175758e12f957e9fb6075b0cc4839a8917ea329f2bdadcc27ce82803bc`)
* **Corpus Exports (Stage 08)**:
  * `outputs/stage-08/maritime_corpus.txt` (Size: 24,556,413 bytes, SHA-256: `602a848485750b465f560b1ca68ba29de60b4adbb28b3a5c5d9a1ebaf270eaec`)
  * `outputs/stage-08/manifest.json` (Size: 688 bytes, SHA-256: `048b9d4ed6a169f7d7cbb2c480b0fac525d96ddea67643c391b9cf9de122ad03`)
* **Cryptographic Representation Manifest (Stage 11)**:
  * `json.jsonl`: SHA-256 `14122f5bd95b401346f84ed5c724c65caf4353524ed1eedf55dd84d13b4b5028` (759,980,595 bytes, 96,874 docs)
  * `key_value.jsonl`: SHA-256 `0bde74c8bf4b9db52f455593577756fd09860f0f4a1f39ce51eda823a583500b` (65,092,615 bytes, 96,874 docs)
  * `mixed.jsonl`: SHA-256 `11643d5d7118da3b4fceb471204ce87d8a8559b9e922f2592c0953cc4b6e4c28` (91,761,238 bytes, 96,874 docs)
  * `narrative.jsonl`: SHA-256 `b63a6be92121a7e1ee0150b609ae00fc7f39c80e83ac8a0bf0520ccff67f6b9f` (31,398,377 bytes, 96,874 docs)
  * `template.jsonl`: SHA-256 `aef2c13fb338b4dae41b7d4b2929b4bdc22c0e0ad780bf4cbc33c1644e3e0ab0` (51,667,477 bytes, 96,874 docs)
  * *Verification Status*: All 5 representation hashes match `outputs/representation_provenance.json` exactly.

---

## 4. Candidate Pool & Selected Model Cohort
* **Candidate Pool**: 13 candidates considered.
* **Evaluated Successfully**: 12 candidates.
* **Excluded Due to Tokenizer Loading Failure**: 1 candidate (`anferico/bert-for-patents`).
* **Excluded Due to Tokenizer Redundancy**: 5 candidates (`bert-large-uncased`, `emilyalsentzer/Bio_ClinicalBERT`, `ProsusAI/finbert`, `google/electra-base-discriminator`, `distilbert-base-uncased`).
* **Excluded Due to Evaluation Anomaly**: `microsoft/deberta-v3-base` (anomalous MLM head incompatibility documented in `outputs/model_selection/deberta_exclusion_audit.json`).
* **Authoritative 7-Model Benchmarked Cohort**:
  1. `bert-base-uncased` (Standard WordPiece, vocab 30,522)
  2. `dmis-lab/biobert-base-cased-v1.2` (Bio/Clinical WordPiece, vocab 28,996)
  3. `nlpaueb/legal-bert-base-uncased` (Legal WordPiece, vocab 30,522)
  4. `allenai/scibert_scivocab_uncased` (SciVocab WordPiece, vocab 31,090)
  5. `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (PubMed WordPiece, vocab 30,522)
  6. `roberta-base` (Byte-Level BPE, vocab 50,265)
  7. `answerdotai/ModernBERT-base` (Modern Extended BPE, vocab 50,280)

---

## 5. Cache Architecture & Discovered Artifacts
| Cache Namespace | Directory Path | File Count | Masking Protocol | Evaluation Unit | Role / Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cache` | `outputs/stage-14/evaluations/cache/` | 175 cells + 2 subdirs | `random_15` | `subword` | **Historical Primary Legacy MLM** |
| `cache_wwm_word` | `outputs/stage-14/evaluations/cache_wwm_word/` | 175 cells + 2 subdirs | `whole_word` | `word` | **Secondary Robustness WWM** |
| `smoke_stage14` | `outputs/stage-14/evaluations/smoke_stage14/` | 6 cells | `subword` / `word` | Diagnostic | Temporary Smoke Test |
| `smoke_20_wwm_word` | `outputs/stage-14/smoke_20_wwm_word/` | 20 cells | `whole_word` | `word` | Temporary 20-cell Test |
| `dapt_wwm` | `outputs/maritimebert_dapt_wwm/cache/` | 50 cells | `whole_word` | `word` | Secondary DAPT Analysis |

---

## 6. DAPT Artifacts & Split Verification
* **Split File Verification (`dapt/outputs/data/`)**:
  * `train.txt`: 87,174 documents (22,098,271 bytes)
  * `val.txt`: 4,843 documents (1,233,481 bytes)
  * `test.txt`: 4,844 documents (1,224,476 bytes)
  * Total Split Sum: $87,174 + 4,843 + 4,844 = 96,861$ documents (matches split manifest).
* **Model Checkpoint**:
  * `dapt/outputs/experiments/MaritimeBERT-v1/model.safetensors` (598,635,032 bytes)
  * `dapt/outputs/experiments/MaritimeBERT-v1/config.json` (1,904 bytes)

---

## 7. Baseline Test Suite Status
* **Test Framework**: `pytest 9.0.3` running on Python 3.11.9.
* **Test Command Requirement**: Must run with `-s` (`pytest -s`) on Windows to prevent `pytest` capture wrapper closed file exceptions.
* **Execution Results**:
  * `tests/test_forensic_fixes.py`: 26 passed
  * `tests/test_stage14_classification_audit.py`: 9 passed
  * `tests/test_quality_optimizations.py`: 7 passed
  * `tests/test_notebook_orchestration.py`: 6 passed
  * **Total Baseline Passing Tests**: **48 passed, 0 failed, 0 skipped**.

---

## 8. Discrepancies Discovered During Forensic Inventory
1. **Stage 14 CLI Default Inversion**: `scripts/14_mlm_evaluation.py` CLI default arguments were changed to `whole_word` and `word`. Legacy primary protocol requires `random_15` subword masking and subword reconstruction.
2. **Stage 15 Downstream Overwrite**: `outputs/stage-15/comparison.csv` currently contains WWM results, displacing the primary legacy MLM evaluation.
3. **MECS ModernBERT Arithmetic Mismatch (B1)**: ModernBERT reported MECS (78.56) exceeds the theoretical upper bound (75.77) under Eq. 2 inputs in Table 4.
4. **Cliff's Delta Interpretation (A2)**: Cliff's $\delta$ was interpreted as paired cell win rate instead of cross-cell dominance across all 625 pairs.
5. **Candidate Pool Documentation (A6)**: Paper text states 5 candidates dropped instead of 6 (omitted the 1 model with tokenizer loading failure).
6. **DAPT Held-Out Test Split (A9)**: DAPT tables evaluate only the 4,843 validation documents; the 4,844 test split remains un-evaluated.
7. **Ranking Concordance Metric (B5)**: Concordance across 5 subsets was labeled as Kendall's $\tau$ rather than Kendall's $W$.
8. **Pareto Dominance Claim (B8)**: Paper claimed ModernBERT holds "rank 1" under Pareto analysis, whereas 5 models are non-dominated on the frontier.

---

## 9. Readiness Assessment for Phase 3
The repository forensic inventory is **COMPLETE and INTERNALLY CONSISTENT**. All historical artifacts, caches, and hashes are safely cataloged. The pipeline is fully prepared to proceed with Phase 3 (Stage-14 Legacy Protocol Restoration) without loss of provenance.
