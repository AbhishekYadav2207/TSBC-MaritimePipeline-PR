# MaritimeBench Execution & Artifact Inventory Report

- **Timestamp (UTC)**: `2026-10-07T07:14:57.901179+00:00`
- **Git Commit SHA**: `NOT_A_GIT_REPO`
- **Environment**: `Linux 6.6.122+ (x86_64)`
- **Python / PyTorch**: `3.13.15` / `2.11.0+cu130`
- **CUDA Available**: `True` (Tesla T4)
- **MLM Masking Mode / Evaluation Unit**: `subword` / `subword`
- **Overall Pipeline Status**: **SUCCESS**

## Stage-by-Stage Artifact Inventory

### Stage 01: scripts/01_parse_dictionary.py (PASS, 3.3s)
- `outputs/stage-01/dictionary_metadata.json`: **FILE**, 177.2 KB (1 items)

### Stage 02: scripts/02_profile_dataset.py (PASS, 11.5s)
- `outputs/stage-02/profiling_report.json`: **FILE**, 79.2 KB (1 items)

### Stage 03: scripts/03_discover_relationships.py (PASS, 1.6s)
- `outputs/stage-03/relationships.json`: **FILE**, 6.5 KB (1 items)

### Stage 04: scripts/04_select_semantic_columns.py (PASS, 1.2s)
- `outputs/stage-04/selected_semantic_columns.json`: **FILE**, 7.0 KB (1 items)

### Stage 05: scripts/05_merge_tables.py (PASS, 87.4s)
- `outputs/stage-05/merged_records.jsonl`: **FILE**, 316.1 MB (1 items)
- `outputs/stage-05/merge_reconciliation_report.json`: **FILE**, 1.2 KB (1 items)

### Stage 05a: scripts/05a_validate_records.py (PASS, 24.4s)
- `outputs/stage-05a/validation_report.json`: **FILE**, 873 B (1 items)

### Stage 06: scripts/06_generate_documents.py (PASS, 36.1s)
- `outputs/stage-06/raw_documents.jsonl`: **FILE**, 822.0 MB (1 items)

### Stage 07: scripts/07_clean_documents.py (PASS, 39.7s)
- `outputs/stage-07/clean_documents.jsonl`: **FILE**, 749.1 MB (1 items)

### Stage 08: scripts/08_export_corpus.py (PASS, 37.5s)
- `outputs/stage-08/maritime_corpus.jsonl`: **FILE**, 738.4 MB (1 items)
- `outputs/stage-08/maritime_corpus.txt`: **FILE**, 23.4 MB (1 items)
- `outputs/stage-08/manifest.json`: **FILE**, 688 B (1 items)

### Stage 09: scripts/09_statistics.py (PASS, 41.6s)
- `outputs/stage-09/statistics.json`: **FILE**, 3.2 KB (1 items)
- `outputs/stage-09/corpus_quality_report.md`: **FILE**, 2.8 KB (1 items)

### Stage 10: scripts/10_extract_vocabulary.py (PASS, 25.7s)
- `outputs/stage-10/maritime_vocabulary.txt`: **FILE**, 2.7 KB (1 items)

### Stage 11: scripts/11_corpus_representations.py (PASS, 38.6s)
- `outputs/stage-11/corpus_representations`: **DIRECTORY**, 953.4 MB (5 items)

### Stage 12: scripts/12_semantic_importance.py (PASS, 84.0s)
- `outputs/stage-12/document_importance.jsonl`: **FILE**, 83.5 MB (1 items)
- `outputs/stage-12/subsets`: **DIRECTORY**, 395.2 MB (16 items)

### Stage 13: scripts/13_tokenizer_analysis.py (PASS, 70.6s)
- `outputs/stage-13/selected_models.json`: **FILE**, 10.0 KB (1 items)
- `outputs/stage-13/candidate_selection_audit.json`: **FILE**, 6.2 KB (1 items)
- `outputs/stage-13/candidate_selection_audit.csv`: **FILE**, 3.2 KB (1 items)
- `outputs/stage-13/tokenizer_analysis`: **DIRECTORY**, 383.3 KB (14 items)

### Stage 14: scripts/14_mlm_evaluation.py (PASS, 1534.5s)
- `outputs/stage-14/evaluations`: **DIRECTORY**, 35.5 KB (9 items)
- `outputs/stage-14/pll_results.json`: **FILE**, 30.6 KB (1 items)
- `outputs/stage-14/pll_selection.json`: **FILE**, 8.2 KB (1 items)

### Stage 15: scripts/15_cross_model_benchmarking.py (PASS, 8.1s)
- `outputs/stage-15/leaderboard.csv`: **FILE**, 1.2 KB (1 items)
- `outputs/stage-15/comparison.csv`: **FILE**, 49.8 KB (1 items)
- `outputs/stage-15/stage15_model_profiles.csv`: **FILE**, 4.6 KB (1 items)
- `outputs/stage-15/stage15_pareto.csv`: **FILE**, 2.0 KB (1 items)
- `outputs/stage-15/stage15_mui_sensitivity.csv`: **FILE**, 724 B (1 items)
- `outputs/stage-15/stage15_report.md`: **FILE**, 9.1 KB (1 items)

### Stage 16: scripts/16_statistical_analysis.py (PASS, 3.5s)
- `outputs/stage-16/stage16_pairwise_tests.csv`: **FILE**, 5.6 KB (1 items)
- `outputs/stage-16/stage16_global_tests.csv`: **FILE**, 235 B (1 items)
- `outputs/stage-16/stage16_bootstrap.csv`: **FILE**, 3.8 KB (1 items)
- `outputs/stage-16/ablation_study.json`: **FILE**, 2.6 KB (1 items)
- `outputs/stage-16/stage16_final_report.md`: **FILE**, 20.4 KB (1 items)

### Stage 17: scripts/17_decision_engine.py (PASS, 11.4s)
- `outputs/stage-17/decision_summary.json`: **FILE**, 2.4 KB (1 items)
- `outputs/stage-17/stage17_model_selection.csv`: **FILE**, 2.8 KB (1 items)
- `outputs/stage-17/benchmark_report.md`: **FILE**, 10.5 KB (1 items)

### Stage 18: scripts/18_lint_corpus.py (PASS, 19.6s)
- `outputs/stage-18/corpus_lint_report.json`: **FILE**, 3.4 KB (1 items)
