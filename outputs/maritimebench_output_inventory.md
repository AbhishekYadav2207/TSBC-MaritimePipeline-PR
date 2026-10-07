# MaritimeBench Execution & Artifact Inventory Report

- **Timestamp (UTC)**: `2026-10-07T19:18:39.335554+00:00`
- **Git Commit SHA**: `NOT_A_GIT_REPO`
- **Environment**: `Linux 6.6.122+ (x86_64)`
- **Python / PyTorch**: `3.13.15` / `2.11.0+cu130`
- **CUDA Available**: `True` (Tesla T4)
- **MLM Masking Mode / Evaluation Unit**: `whole_word` / `word`
- **Overall Pipeline Status**: **SUCCESS**

## Stage-by-Stage Artifact Inventory

### Stage 11: scripts/11_corpus_representations.py (PASS, 48.4s)
- `outputs/stage-11/corpus_representations`: **DIRECTORY**, 953.6 MB (5 items)

### Stage 12: scripts/12_semantic_importance.py (PASS, 89.8s)
- `outputs/stage-12/document_importance.jsonl`: **FILE**, 83.5 MB (1 items)
- `outputs/stage-12/subsets`: **DIRECTORY**, 395.3 MB (16 items)

### Stage 13: scripts/13_tokenizer_analysis.py (PASS, 76.2s)
- `outputs/stage-13/selected_models.json`: **FILE**, 23.6 KB (1 items)
- `outputs/stage-13/candidate_selection_audit.json`: **FILE**, 15.1 KB (1 items)
- `outputs/stage-13/candidate_selection_audit.csv`: **FILE**, 9.8 KB (1 items)
- `outputs/stage-13/tokenizer_analysis`: **DIRECTORY**, 361.6 KB (13 items)

### Stage 14: scripts/14_mlm_evaluation.py (PASS, 1350.0s)
- `outputs/stage-14/evaluations`: **DIRECTORY**, 50.7 KB (8 items)
- `outputs/stage-14/pll_results.json`: **FILE**, 27.0 KB (1 items)
- `outputs/stage-14/pll_selection.json`: **FILE**, 8.2 KB (1 items)

### Stage 15: scripts/15_cross_model_benchmarking.py (PASS, 6.8s)
- `outputs/stage-15/leaderboard.csv`: **FILE**, 1.1 KB (1 items)
- `outputs/stage-15/comparison.csv`: **FILE**, 57.5 KB (1 items)
- `outputs/stage-15/stage15_model_profiles.csv`: **FILE**, 4.2 KB (1 items)
- `outputs/stage-15/stage15_pareto.csv`: **FILE**, 1.6 KB (1 items)
- `outputs/stage-15/stage15_mecs_sensitivity.csv`: **FILE**, 662 B (1 items)
- `outputs/stage-15/stage15_report.md`: **FILE**, 9.3 KB (1 items)

### Stage 16: scripts/16_statistical_analysis.py (PASS, 4.9s)
- `outputs/stage-16/stage16_pairwise_tests.csv`: **FILE**, 4.1 KB (1 items)
- `outputs/stage-16/stage16_global_tests.csv`: **FILE**, 258 B (1 items)
- `outputs/stage-16/stage16_bootstrap.csv`: **FILE**, 3.0 KB (1 items)
- `outputs/stage-16/ablation_study.json`: **FILE**, 2.6 KB (1 items)
- `outputs/stage-16/stage16_final_report.md`: **FILE**, 18.6 KB (1 items)

### Stage 17: scripts/17_decision_engine.py (PASS, 8.8s)
- `outputs/stage-17/decision_summary.json`: **FILE**, 2.4 KB (1 items)
- `outputs/stage-17/stage17_model_selection.csv`: **FILE**, 2.5 KB (1 items)
- `outputs/stage-17/benchmark_report.md`: **FILE**, 10.4 KB (1 items)

### Stage 18: scripts/18_lint_corpus.py (PASS, 20.5s)
- `outputs/stage-18/corpus_lint_report.json`: **FILE**, 3.4 KB (1 items)
