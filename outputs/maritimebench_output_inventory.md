# MaritimeBench Execution & Artifact Inventory Report

- **Timestamp (UTC)**: `2026-10-08T07:39:04.912862+00:00`
- **Git Commit SHA**: `NOT_A_GIT_REPO`
- **Environment**: `Linux 6.6.122+ (x86_64)`
- **Python / PyTorch**: `3.13.15` / `2.11.0+cu130`
- **CUDA Available**: `True` (Tesla T4)
- **MLM Masking Mode / Evaluation Unit**: `subword` / `subword`
- **Overall Pipeline Status**: **INCOMPLETE_OR_FAILED**

## Stage-by-Stage Artifact Inventory

### Stage 01: scripts/01_parse_dictionary.py (PASS, 2.9s)
- `outputs/stage-01/dictionary_metadata.json`: **FILE**, 177.3 KB (1 items)

### Stage 02: scripts/02_profile_dataset.py (PASS, 16.9s)
- `outputs/stage-02/profiling_report.json`: **FILE**, 79.2 KB (1 items)

### Stage 03: scripts/03_discover_relationships.py (PASS, 1.8s)
- `outputs/stage-03/relationships.json`: **FILE**, 6.5 KB (1 items)

### Stage 04: scripts/04_select_semantic_columns.py (PASS, 1.0s)
- `outputs/stage-04/selected_semantic_columns.json`: **FILE**, 7.0 KB (1 items)

### Stage 05: scripts/05_merge_tables.py (PASS, 94.7s)
- `outputs/stage-05/merged_records.jsonl`: **FILE**, 316.1 MB (1 items)
- `outputs/stage-05/merge_reconciliation_report.json`: **FILE**, 1.2 KB (1 items)

### Stage 05a: scripts/05a_validate_records.py (PASS, 25.3s)
- `outputs/stage-05a/validation_report.json`: **FILE**, 873 B (1 items)

### Stage 06: scripts/06_generate_documents.py (PASS, 40.4s)
- `outputs/stage-06/raw_documents.jsonl`: **FILE**, 822.0 MB (1 items)

### Stage 07: scripts/07_clean_documents.py (PASS, 44.8s)
- `outputs/stage-07/clean_documents.jsonl`: **FILE**, 749.0 MB (1 items)

### Stage 08: scripts/08_export_corpus.py (PASS, 39.4s)
- `outputs/stage-08/maritime_corpus.jsonl`: **FILE**, 738.3 MB (1 items)
- `outputs/stage-08/maritime_corpus.txt`: **FILE**, 23.4 MB (1 items)
- `outputs/stage-08/manifest.json`: **FILE**, 688 B (1 items)

### Stage 09: scripts/09_statistics.py (PASS, 52.8s)
- `outputs/stage-09/statistics.json`: **FILE**, 3.4 KB (1 items)
- `outputs/stage-09/corpus_quality_report.md`: **FILE**, 3.2 KB (1 items)

### Stage 10: scripts/10_extract_vocabulary.py (PASS, 27.3s)
- `outputs/stage-10/maritime_vocabulary.txt`: **FILE**, 2.7 KB (1 items)

### Stage 11: scripts/11_corpus_representations.py (PASS, 39.6s)
- `outputs/stage-11/corpus_representations`: **DIRECTORY**, 953.3 MB (5 items)

### Stage 12: scripts/12_semantic_importance.py (PASS, 87.5s)
- `outputs/stage-12/document_importance.jsonl`: **FILE**, 83.5 MB (1 items)
- `outputs/stage-12/subsets`: **DIRECTORY**, 395.2 MB (16 items)

### Stage 13: scripts/13_tokenizer_analysis.py (PASS, 70.0s)
- `outputs/stage-13/selected_models.json`: **FILE**, 23.6 KB (1 items)
- `outputs/stage-13/candidate_selection_audit.json`: **FILE**, 15.1 KB (1 items)
- `outputs/stage-13/candidate_selection_audit.csv`: **FILE**, 9.8 KB (1 items)
- `outputs/stage-13/tokenizer_analysis`: **DIRECTORY**, 361.6 KB (13 items)

### Stage 14: scripts/14_mlm_evaluation.py (FAILED (Exit Code 1), 74.4s)
- *(No artifacts recorded)*

### Stage 14: scripts/14_mlm_evaluation.py (PASS, 1192.1s)
- `outputs/stage-14/evaluations`: **DIRECTORY**, 49.2 KB (8 items)
- `outputs/stage-14/pll_results.json`: **FILE**, 27.0 KB (1 items)
- `outputs/stage-14/pll_selection.json`: **FILE**, 8.2 KB (1 items)

### Stage 15: scripts/15_cross_model_benchmarking.py (PASS, 8.2s)
- `outputs/stage-15/leaderboard.csv`: **FILE**, 1.1 KB (1 items)
- `outputs/stage-15/comparison.csv`: **FILE**, 68.2 KB (1 items)
- `outputs/stage-15/stage15_model_profiles.csv`: **FILE**, 4.2 KB (1 items)
- `outputs/stage-15/stage15_pareto.csv`: **FILE**, 1.4 KB (1 items)
- `outputs/stage-15/stage15_mecs_sensitivity.csv`: **FILE**, 661 B (1 items)
- `outputs/stage-15/stage15_report.md`: **FILE**, 9.1 KB (1 items)

### Stage 16: scripts/16_statistical_analysis.py (PASS, 5.2s)
- `outputs/stage-16/stage16_pairwise_tests.csv`: **FILE**, 4.1 KB (1 items)
- `outputs/stage-16/stage16_global_tests.csv`: **FILE**, 277 B (1 items)
- `outputs/stage-16/stage16_bootstrap.csv`: **FILE**, 3.0 KB (1 items)
- `outputs/stage-16/ablation_study.json`: **FILE**, 2.6 KB (1 items)
- `outputs/stage-16/stage16_final_report.md`: **FILE**, 19.4 KB (1 items)

### Stage 17: scripts/17_decision_engine.py (PASS, 11.2s)
- `outputs/stage-17/decision_summary.json`: **FILE**, 2.4 KB (1 items)
- `outputs/stage-17/stage17_model_selection.csv`: **FILE**, 2.5 KB (1 items)
- `outputs/stage-17/benchmark_report.md`: **FILE**, 10.5 KB (1 items)

### Stage 18: scripts/18_lint_corpus.py (PASS, 21.0s)
- `outputs/stage-18/corpus_lint_report.json`: **FILE**, 3.4 KB (1 items)
