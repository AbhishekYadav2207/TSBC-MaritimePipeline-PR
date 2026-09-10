# Appendix B: Model Evaluation Results & Empirical Benchmarks (Stages 11–18)

## Executive Overview
Appendix B provides comprehensive empirical tables and benchmarks from Stages 11 through 18 of the maritime pipeline. It documents multi-format representation volumes, Stage 12 Domain Informativeness scores and leave-one-out feature ablation, Stage 13 multi-architecture tokenizer comparisons and redundancy clustering, the complete 175-cell Stage 14 MLM evaluation grid, Stage 15 multi-criteria model profiles, representation robustness, subset consistency, weighting sensitivity, and Pareto dominance analysis.

---

## 1. Stage 11: Multi-Format Representation Volume

### Table 1.1: Multi-Format Corpus Representation Artifacts (`outputs/stage-11/corpus_representations/`)
| Representation Identifier | Format Description | File Size | Document Count | Average Doc Length (words) | Line-Level Integrity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `narrative.jsonl` | Sanitized natural language paragraphs | 84.8 MB | **96,848** | 33.89 | 100% Valid JSON |
| `key_value.jsonl` | Line-oriented `Key: Value` attributes | 78.1 MB | **96,848** | 28.54 | 100% Valid JSON |
| `template.jsonl` | Standardized semi-structured template text | 73.9 MB | **96,848** | 26.12 | 100% Valid JSON |
| `json.jsonl` | Serialized JSON format strings | 97.6 MB | **96,848** | 41.20 | 100% Valid JSON |
| `mixed.jsonl` | Hybrid key-value header + narrative body | 111.5 MB | **96,848** | 62.43 | 100% Valid JSON |

---

## 2. Stage 12: Domain Informativeness Distribution & Knowledge Tiers

### Table 2.1: Domain Informativeness Score Summary (`outputs/stage-12/importance_statistics.json`)
| Metric Name | Empirical Value | Description / Benchmark |
| :--- | :---: | :--- |
| **Total Documents Evaluated** | **96,848** | 100% of clean corpus documents |
| **Mean Importance Score** | **38.93** | Corpus-wide mean ($0.0 \le S \le 100.0$) |
| **Median Importance Score** | **38.33** | Central tendency index |
| **Standard Deviation** | **10.02** | Score distribution dispersion |
| **Score Quartiles [P25, P50, P75]**| **[32.26, 38.33, 45.43]** | Interquartile distribution bounds |
| **Minimum / Maximum Score** | **10.67 / 74.21** | Range extremes |

---

### Table 2.2: Knowledge Tier Partitioning
| Knowledge Tier | Importance Score Range | Document Count | Percentage of Corpus | Operational Definition |
| :--- | :---: | :---: | :---: | :--- |
| **High Knowledge** | $S \ge 42.0$ ($S \ge \text{P80}$) and $P_{\text{red}} < 0.40$ | **19,376** | **20.01%** | Dense technical narratives, complex vessel interactions |
| **Medium Knowledge**| $32.26 \le S < 42.0$ ($\text{P20} \le S < \text{P80}$) | **51,737** | **53.42%** | Standard operational reports with complete metadata |
| **Low Knowledge** | $S < 32.26$ ($S < \text{P20}$) and $P_{\text{red}} < 0.40$ | **13,517** | **13.96%** | Brief incident summaries, sparse equipment data |
| **Redundant / Boilerplate**| Redundancy penalty $P_{\text{red}} \ge 0.40$ | **12,218** | **12.61%** | Short template boilerplates ($< 120$ characters) |
| **Total Clean Corpus** | Full evaluation coverage | **96,848** | **100.0%** | Zero unprocessed records |

---

### Table 2.3: Leave-One-Dimension-Out Informativeness Ablation (`outputs/stage-12/informativeness_ablation.json`)
| Ablation Condition | Spearman $\rho$ | Kendall $\tau$ | Mean Score | Std Score | Key Impact Summary |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Equal-Weight Hybrid** | **1.0000** | **1.0000** | **0.2487** | **0.0753** | 4-signal balanced baseline |
| `minus_domain_relevance` | 0.9753 | 0.8567 | 0.3074 | 0.0851 | Minor rank shift; scores inflate |
| `minus_information_content` | 0.9052 | 0.8168 | 0.0545 | 0.0680 | Severe score compression; length scaling vital |
| `minus_tfidf_representativeness` | 0.7956 | 0.7032 | 0.2519 | 0.0742 | Significant rank divergence ($\rho < 0.80$) |
| `minus_redundancy_noise` | 0.7894 | 0.6819 | 0.3976 | 0.0925 | Strongest rank disruption; boilerplate unpenalized |

---

## 3. Stage 13: Multi-Architecture Tokenizer Benchmarking

### Table 3.1: Comparative Tokenizer Benchmark (`outputs/stage-13/tokenizer_analysis/tokenizer_comparison.csv`)
| Model Identifier | Selected Stage 14 | Vocab Size | Fertility (subwords/word) | Single-Token Coverage (%) | Frag Rate (%) | OOV Rate (%) | OOV Status | Speed (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bert-base-uncased` | **True** | 30,522 | **1.3984** | **73.43%** | **26.57%** | 0.00% | measured | 61,089 |
| `dmis-lab/biobert-base-cased-v1.2` | **True** | 28,996 | 1.4789 | 64.48% | 35.52% | 0.00% | measured | 64,953 |
| `nlpaueb/legal-bert-base-uncased` | **True** | 30,522 | 1.4806 | 62.09% | 37.91% | 0.03% | measured | 62,213 |
| `allenai/scibert_scivocab_uncased` | **True** | 31,090 | 1.4517 | 57.91% | 42.09% | 0.00% | measured | 61,481 |
| `microsoft/BiomedNLP-PubMedBERT...` | **True** | 30,522 | 1.4543 | 57.31% | 42.69% | 0.00% | measured | 63,963 |
| `answerdotai/ModernBERT-base` | **True** | 50,280 | 1.5236 | 36.72% | 63.28% | N/A | byte_fallback | **73,967** |
| `roberta-base` | **True** | 50,265 | 1.5609 | 34.93% | 65.07% | N/A | byte_fallback | 69,639 |
| `bert-large-uncased` | False | 30,522 | 1.3984 | 73.43% | 26.57% | 0.00% | measured | 61,926 |
| `ProsusAI/finbert` | False | 30,522 | 1.3984 | 73.43% | 26.57% | 0.00% | measured | 53,347 |
| `google/electra-base-discriminator`| False | 30,522 | 1.3984 | 73.43% | 26.57% | 0.00% | measured | 61,445 |
| `distilbert-base-uncased` | False | 30,522 | 1.3984 | 73.43% | 26.57% | 0.00% | measured | 57,714 |
| `emilyalsentzer/Bio_ClinicalBERT` | False | 28,996 | 1.4789 | 64.48% | 35.52% | 0.00% | measured | 59,273 |

---

## 4. Stage 14: 175-Cell Matrix MLM Evaluation Grid

### Table 4.1: Cross-Model Intrinsic Capability Summary (Full Matrix Averages)
| Representative Model | Vocab Type | Maritime Top-1 (%) | Top-5 (%) | Rare Top-1 (%) | MLM Loss | Pseudo-Perplexity | Domain Shift Gap (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | Extended BPE | **56.04%** | **72.36%** | 29.09% | **2.3063** | **13.43** | $-16.42\%$ |
| `roberta-base` | Byte-BPE | 47.03% | 66.64% | 30.07% | 2.7948 | 16.11 | $-14.15\%$ |
| `allenai/scibert_scivocab_uncased` | Sci-WordPiece | 31.24% | 47.30% | 6.44% | 4.2628 | 35.28 | $+18.21\%$ |
| `bert-base-uncased` | WordPiece | 29.08% | 46.69% | **54.35%** | 4.6237 | 22.56 | $+19.54\%$ |
| `nlpaueb/legal-bert-base-uncased` | Legal WordPiece | 28.66% | 44.99% | 4.49% | 4.4541 | 44.78 | $+12.87\%$ |
| `dmis-lab/biobert-base-cased-v1.2` | Bio WordPiece | 24.65% | 36.98% | 32.35% | 4.9969 | 98.54 | $+17.65\%$ |
| `microsoft/BiomedNLP-PubMedBERT...` | PubMed WordPiece| 20.59% | 30.71% | 3.40% | 5.7259 | 103.80 | $+26.94\%$ |

---

## 5. Stage 15: Cross-Model Benchmarking, Sensitivity & Pareto Optimality

### Table 5.1: Final Maritime Understanding Index (MUI) Leaderboard (`outputs/stage-15/leaderboard.csv`)
| Rank | Model Name | Baseline MUI | Top-1 Acc (%) | Rare Acc (%) | MLM Loss | PPL | Frag (%) | Single Cov (%) | Latency (ms) | Throughput (docs/s) | Params (M) | Pareto Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `answerdotai/ModernBERT-base` | **68.29** | **56.04%** | 29.09% | **2.3063** | **13.43** | 63.3% | 36.7% | 458.6ms | 2.2 | 149M | **Pareto-Optimal** |
| **2** | `bert-base-uncased` | **59.25** | 29.08% | **54.35%** | 4.6237 | 22.56 | **26.6%** | **73.4%** | **174.9ms** | 5.7 | 110M | **Pareto-Optimal** |
| **3** | `roberta-base` | **56.70** | 47.03% | 30.07% | 2.7948 | 16.11 | 65.1% | 34.9% | 382.7ms | 2.6 | 125M | **Pareto-Optimal** |
| **4** | `dmis-lab/biobert-base-cased-v1.2` | **42.24** | 24.65% | 32.35% | 4.9969 | 98.54 | 35.5% | 64.5% | **154.1ms** | **6.5** | 110M | **Pareto-Optimal** |
| **5** | `allenai/scibert_scivocab_uncased` | **37.09** | 31.24% | 6.44% | 4.2628 | 35.28 | 42.1% | 57.9% | 323.4ms | 3.1 | 110M | **Pareto-Optimal** |
| **6** | `nlpaueb/legal-bert-base-uncased` | **27.08** | 28.66% | 4.49% | 4.4541 | 44.78 | 37.9% | 62.1% | 249.3ms | 4.0 | 110M | **Pareto-Optimal** |
| **7** | `microsoft/BiomedNLP-PubMedBERT...`| **23.72** | 20.59% | 3.40% | 5.7259 | 103.80 | 42.7% | 57.3% | 326.8ms | 3.1 | 110M | **Dominated** |

---

### Table 5.2: MUI Weighting Sensitivity Matrix (`outputs/stage-15/stage15_mui_sensitivity.csv`)
| Model Identifier | Baseline Score (Rank) | Perf-Heavy Score (Rank) | Domain-Heavy Score (Rank) | Balanced Score (Rank) | Total Wins | Win Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | **68.29 (#1)** | **90.14 (#1)** | 64.76 (#2) | 51.02 (#3) | **2** | **50.0%** |
| `bert-base-uncased` | 59.25 (#2) | 43.41 (#3) | **70.82 (#1)** | **67.68 (#1)** | **2** | **50.0%** |
| `roberta-base` | 56.70 (#3) | 74.72 (#2) | 55.39 (#3) | 44.52 (#4) | 0 | 0.0% |
| `dmis-lab/biobert-base-cased-v1.2` | 42.24 (#4) | 23.49 (#6) | 47.48 (#4) | 53.31 (#2) | 0 | 0.0% |
| `allenai/scibert_scivocab_uncased` | 37.09 (#5) | 29.87 (#4) | 35.04 (#5) | 31.94 (#6) | 0 | 0.0% |
| `nlpaueb/legal-bert-base-uncased` | 27.08 (#6) | 23.98 (#5) | 22.56 (#6) | 34.98 (#5) | 0 | 0.0% |
| `microsoft/BiomedNLP-PubMedBERT...` | 23.72 (#7) | 0.00 (#7) | 18.73 (#7) | 15.68 (#7) | 0 | 0.0% |

---

### Table 5.3: Multi-Objective Pareto Dominance Classification (`outputs/stage-15/stage15_pareto.csv`)
| Model Identifier | Pareto Status | Dominates Count | Dominated By Count | Dominating Models | Non-Dominated Trade-off Dimension |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `answerdotai/ModernBERT-base` | **Pareto-Optimal** | 0 | 0 | None | Top Intrinsic MLM Accuracy (56.04%), Lowest Loss (2.3063) |
| `bert-base-uncased` | **Pareto-Optimal** | 1 | 0 | None | Top Rare Token Accuracy (54.35%), Lowest Frag (26.6%) |
| `roberta-base` | **Pareto-Optimal** | 0 | 0 | None | Strong general capability (47.03% Top-1) at 125M footprint |
| `dmis-lab/biobert-base-cased-v1.2` | **Pareto-Optimal** | 1 | 0 | None | Top Throughput (6.5 docs/s), Lowest Latency (154.1ms) |
| `allenai/scibert_scivocab_uncased` | **Pareto-Optimal** | 1 | 0 | None | Balanced scientific vocabulary at 110M footprint |
| `nlpaueb/legal-bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None | Intermediate latency (249.3ms) with custom domain vocabulary |
| `microsoft/BiomedNLP-PubMedBERT...` | **Dominated** | 0 | **3** | SciBERT, BERT-base, BioBERT | Inferior across capability, loss, and latency |
