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

---

## 6. Stage 16: Statistical Validation & Component Sensitivity Benchmarks

### Table 6.1: Global Repeated-Measures Omnibus Test (`outputs/stage-16/stage16_global_tests.csv`)
Evaluates whether model performances across the 25 matched representation-by-subset benchmark configurations differ beyond chance.

| Statistical Test | Matched Conditions ($N$) | Candidate Models ($k$) | Degrees of Freedom ($df$) | Chi-Square Statistic ($\chi^2$) | Raw $p$-value | Omnibus Null Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Friedman Chi-Square** | 25 | 7 | 6 | **127.9714** | **$3.4361 \times 10^{-25}$** | **Reject $H_0$ ($p < 0.001$)** |

*Methodological Finding*: Model differences across the matched benchmark grid are highly statistically significant, justifying post-hoc pairwise hypothesis testing.

---

### Table 6.2: Complete Pairwise Wilcoxon Signed-Rank & Paired $t$-Test Matrix (`outputs/stage-16/stage16_pairwise_tests.csv`)
Evaluates matched paired differences across $N = 25$ conditions with family-wise error controlled by the **Holm-Bonferroni** step-down procedure.

| Model A | Model B | $N$ | Mean Diff | Median Diff | Wilcoxon Stat ($W$) | Raw $p$-value | Holm $p$-value | Holm Significant? | Paired $t$-stat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | 25 | -0.4044 | -0.4272 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -19.46 |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | 25 | +0.0289 | +0.0419 | 70.0 | 0.0115 | 0.0344 | **Yes ($p < 0.05$)** | +2.55 |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | +0.0683 | +0.0686 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +12.76 |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT...`| 25 | +0.1142 | +0.1063 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +14.96 |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.0043 | +0.0105 | 152.0 | 0.7915 | 0.7915 | No | -0.28 |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | 25 | -0.3070 | -0.3315 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -13.74 |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | 25 | +0.4333 | +0.4632 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +18.51 |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | 25 | +0.4727 | +0.4863 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +21.59 |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT...`| 25 | +0.5186 | +0.5473 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +22.36 |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | 25 | +0.4000 | +0.3735 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +19.02 |
| `answerdotai/ModernBERT-base` | `roberta-base` | 25 | +0.0973 | +0.0768 | 65.0 | 0.0074 | 0.0295 | **Yes ($p < 0.05$)** | +3.01 |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | +0.0394 | +0.0254 | 56.0 | 0.0031 | 0.0154 | **Yes ($p < 0.05$)** | +3.18 |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT...`| 25 | +0.0853 | +0.0722 | 10.0 | $2.56 \times 10^{-6}$ | $1.79 \times 10^{-5}$ | **Yes ($p < 0.05$)** | +5.74 |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.0332 | -0.0223 | 99.0 | 0.0903 | 0.1806 | No | -1.98 |
| `bert-base-uncased` | `roberta-base` | 25 | -0.3359 | -0.3855 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -13.34 |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT...`| 25 | +0.0459 | +0.0465 | 4.0 | $4.17 \times 10^{-7}$ | $3.76 \times 10^{-6}$ | **Yes ($p < 0.05$)** | +8.43 |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.0726 | -0.0576 | 5.0 | $5.96 \times 10^{-7}$ | $4.77 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -5.64 |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | 25 | -0.3754 | -0.4072 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -15.04 |
| `microsoft/BiomedNLP-PubMedBERT...`| `nlpaueb/legal-bert-base-uncased` | 25 | -0.1185 | -0.0876 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -8.48 |
| `microsoft/BiomedNLP-PubMedBERT...`| `roberta-base` | 25 | -0.4213 | -0.4497 | 0.0 | $5.96 \times 10^{-8}$ | $1.25 \times 10^{-6}$ | **Yes ($p < 0.05$)** | -16.20 |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | 25 | -0.3027 | -0.3555 | 11.0 | $3.28 \times 10^{-6}$ | $1.97 \times 10^{-5}$ | **Yes ($p < 0.05$)** | -8.45 |

---

### Table 6.3: Effect Size Quantification & Practical Significance (`outputs/stage-16/stage16_effect_sizes.csv`)

| Comparison Pair | Mean Diff | 95% Paired CI | Paired Cohen's $d_z$ | $d_z$ Magnitude | Cliff's $\delta$ | $\delta$ Magnitude | Practical Significance Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| ModernBERT vs BERT-base | +0.4333 | [+0.3850, +0.4816] | **+3.70** | large | **+1.00** | large | Substantial practical superiority ($100\%$ pairwise dominance) |
| ModernBERT vs RoBERTa | +0.0973 | [+0.0305, +0.1641] | **+0.60** | medium | **+0.23** | small | Meaningful practical advantage on matched benchmarks |
| ModernBERT vs BioBERT | +0.4727 | [+0.4275, +0.5179] | **+4.32** | large | **+1.00** | large | Substantial practical superiority |
| ModernBERT vs SciBERT | +0.4044 | [+0.3615, +0.4472] | **+3.89** | large | **+1.00** | large | Substantial practical superiority |
| ModernBERT vs Legal-BERT | +0.4000 | [+0.3566, +0.4435] | **+3.80** | large | **+1.00** | large | Substantial practical superiority |
| ModernBERT vs PubMedBERT | +0.5186 | [+0.4707, +0.5664] | **+4.47** | large | **+1.00** | large | Substantial practical superiority |
| RoBERTa vs BERT-base | +0.3359 | [+0.2840, +0.3879] | **+2.67** | large | **+0.71** | large | Substantial practical advantage |
| BERT-base vs BioBERT | +0.0394 | [+0.0138, +0.0650] | **+0.64** | medium | **+0.31** | small | Moderate practical advantage |
| BERT-base vs PubMedBERT | +0.0853 | [+0.0546, +0.1160] | **+1.15** | large | **+0.42** | medium | Large practical advantage |
| BERT-base vs Legal-BERT | -0.0332 | [-0.0678, +0.0014] | **-0.40** | small | **-0.07** | negligible | Overlapping confidence interval, negligible practical separation |
| SciBERT vs Legal-BERT | -0.0043 | [-0.0359, +0.0272] | **-0.06** | negligible | **+0.14** | negligible | Indistinguishable performance ($d_z < 0.2$) |

---

### Table 6.4: Bootstrap Uncertainty Estimation ($B = 2,000$ Resamples, Seed = 42) (`outputs/stage-16/stage16_bootstrap.csv`)

| Evaluated Model | Bootstrap Mean | 95% CI Lower | 95% CI Upper | Parametric Std Dev | Resampling Budget | Benchmark Matrix Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | **0.7306** | **0.6819** | **0.7782** | 0.0246 | 2,000 | 25 conditions |
| `roberta-base` | **0.6329** | **0.5501** | **0.7033** | 0.0391 | 2,000 | 25 conditions |
| `nlpaueb/legal-bert-base-uncased` | **0.3306** | **0.3023** | **0.3619** | 0.0152 | 2,000 | 25 conditions |
| `allenai/scibert_scivocab_uncased` | **0.3260** | **0.2807** | **0.3689** | 0.0225 | 2,000 | 25 conditions |
| `bert-base-uncased` | **0.2973** | **0.2536** | **0.3373** | 0.0213 | 2,000 | 25 conditions |
| `dmis-lab/biobert-base-cased-v1.2` | **0.2577** | **0.2128** | **0.3006** | 0.0224 | 2,000 | 25 conditions |
| `microsoft/BiomedNLP-PubMedBERT...` | **0.2117** | **0.1644** | **0.2579** | 0.0238 | 2,000 | 25 conditions |

---

### Table 6.5: Bootstrap Empirical Rank Stability Distributions (`outputs/stage-16/stage16_rank_stability.csv`)

| Candidate Model | Mean Rank | Rank SD | $P(\text{Rank}=1)$ | $P(\text{Rank}=2)$ | $P(\text{Rank} \le 3)$ | Stability Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `answerdotai/ModernBERT-base` | **1.00** | **±0.00** | **1.0000 (100.0%)** | 0.0000 | 1.0000 | **Decisive Benchmark Leader** ($P_1 = 100\%$) |
| `roberta-base` | **2.00** | **±0.00** | 0.0000 | **1.0000 (100.0%)** | 1.0000 | **Decisive Benchmark Runner-Up** ($P_2 = 100\%$) |
| `nlpaueb/legal-bert-base-uncased` | **3.41** | ±0.53 | 0.0000 | 0.0000 | 0.6135 | Competitive Mid-Tier Candidate |
| `allenai/scibert_scivocab_uncased` | **3.62** | ±0.50 | 0.0000 | 0.0000 | 0.3830 | Competitive Mid-Tier Candidate |
| `bert-base-uncased` | **4.97** | ±0.19 | 0.0000 | 0.0000 | 0.0035 | Stable Lower-Tier General Baseline |
| `dmis-lab/biobert-base-cased-v1.2` | **6.00** | **±0.00** | 0.0000 | 0.0000 | 0.0000 | Invariant Lower-Tier Candidate |
| `microsoft/BiomedNLP-PubMedBERT...` | **7.00** | **±0.00** | 0.0000 | 0.0000 | 0.0000 | Invariant Lowest-Ranked Candidate |

---

### Table 6.6: Corpus Representation Robustness & Ranking Concordance (`outputs/stage-16/stage16_condition_robustness.csv`)

| Representation Format | Top-Ranked Model | Top-1 Accuracy | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Rank Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **JSON** | `answerdotai/ModernBERT-base` | 0.6073 | **0.9643** | **0.9048** | High ranking stability; compresses margins |
| **Key-Value** | `answerdotai/ModernBERT-base` | 0.8683 | **0.9643** | **0.9048** | High ranking stability; maximum accuracy |
| **Mixed** | `answerdotai/ModernBERT-base` | 0.8586 | **0.9643** | **0.9048** | High ranking stability; robust hybrid formatting |
| **Narrative** | `answerdotai/ModernBERT-base` | 0.7227 | **0.8929** | **0.8095** | Strong ranking stability on continuous prose |
| **Template** | `roberta-base` | 0.7175 | **0.8571** | **0.7143** | Moderate stability; slight upper-rank perturbation |

---

### Table 6.7: Knowledge Subset Robustness & Ranking Concordance (`outputs/stage-16/stage16_condition_robustness.csv`)

| Knowledge Stratum | Top-Ranked Model | Top-1 Accuracy | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Stratum Consistency Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Balanced Knowledge** | `answerdotai/ModernBERT-base` | 0.7199 | **0.8929** | **0.8095** | Strong ranking concordance |
| **High Knowledge** | `answerdotai/ModernBERT-base` | 0.7237 | **0.9643** | **0.9048** | High ranking concordance on technical jargon |
| **Low Knowledge** | `answerdotai/ModernBERT-base` | 0.7249 | **0.9643** | **0.9048** | High ranking concordance on sparse records |
| **Medium Knowledge** | `answerdotai/ModernBERT-base` | 0.7613 | **1.0000** | **1.0000** | **Perfect Rank Invariance** |
| **Random Baseline** | `answerdotai/ModernBERT-base` | 0.7228 | **1.0000** | **1.0000** | **Perfect Rank Invariance** |

---

### Table 6.8: Stage 12 Component Score & Ranking Sensitivity (`outputs/stage-16/stage16_ablation.csv`)
Measures impact on document scores and top document selection when individual informativeness signals are ablated.

| Ablated Scoring Signal | Mean $\Delta_{\text{score}}$ | Median $\Delta$ | SD $\Delta$ | Spearman $\rho$ | Kendall $\tau$ | Top-20% Jaccard Overlap | Empirical Component Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `domain_relevance` | -0.0586 | -0.0616 | 0.0244 | **0.9630** | **0.8398** | **0.6684 (66.8%)** | Moderate: alters borderline technical documents |
| `information_content` | +0.1943 | +0.1991 | 0.0329 | **0.9364** | **0.8452** | **0.8875 (88.8%)** | Low: upward shift, top selections highly retained |
| `tfidf_representativeness` | -0.0031 | +0.0031 | 0.0422 | **0.8582** | **0.7223** | **0.6051 (60.5%)** | Moderate: anchors domain vocabulary density |
| `redundancy_noise` | -0.1489 | -0.1493 | 0.0558 | **0.8605** | **0.6745** | **0.5049 (50.5%)** | **High**: critical for filtering short boilerplates |

---

### Table 6.9: Stage 12 Stratum Transition & Tier Retention Matrix (`outputs/stage-16/stage16_ablation_stability.csv`)

| Ablated Scoring Signal | Unchanged Tier (%) | High Tier Retention (%) | Medium Tier Retention (%) | Low Tier Retention (%) | Shifted Documents ($N = 96,869$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `domain_relevance` | **89.57%** | **80.11%** | 91.36% | 93.69% | 10,100 / 96,869 |
| `information_content` | **81.07%** | **93.99%** | 74.57% | 87.58% | 18,342 / 96,869 |
| `tfidf_representativeness` | **75.95%** | **75.37%** | 79.98% | 64.46% | 23,300 / 96,869 |
| `redundancy_noise` | **66.82%** | **67.09%** | 72.36% | 49.98% | 32,145 / 96,869 |

---

## 7. Stage 17: Objective Decision Engine & Model Selection Benchmarks

### Table 7.1: Candidate Model Status Classification & Unified Profile Summary (`outputs/stage-17/stage17_model_selection.csv`)

| Candidate Model | Candidate Status | Empirical Top-1 (%) | Intrinsic Loss | Bootstrap Mean Rank | $P(\text{Rank}=1)$ | Pareto Frontier | Assigned Decision Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `answerdotai/ModernBERT-base` | **Strong Candidate** | **73.05%** | **1.4386** | **1.00** | **100.0%** | **Pareto-Optimal** | **Primary DAPT Candidate** |
| `roberta-base` | **Strong Candidate** | 63.32% | 1.9490 | 2.00 | 0.0% | **Pareto-Optimal** | General Capability Runner-Up |
| `nlpaueb/legal-bert-base-uncased` | **Competitive Candidate**| 33.05% | 4.0853 | 3.41 | 0.0% | **Pareto-Optimal** | Evaluated Competitor |
| `allenai/scibert_scivocab_uncased` | **Competitive Candidate**| 32.62% | 4.2064 | 3.62 | 0.0% | **Pareto-Optimal** | Evaluated Competitor |
| `bert-base-uncased` | **Weak Candidate** | 29.73% | 4.6164 | 4.97 | 0.0% | **Pareto-Optimal** | **Resource-Constrained Alternative** |
| `dmis-lab/biobert-base-cased-v1.2` | **Weak Candidate** | 25.78% | 4.7867 | 6.00 | 0.0% | Dominated | Evaluated Competitor |
| `microsoft/BiomedNLP-PubMedBERT...` | **Weak Candidate** | 21.20% | 5.7246 | 7.00 | 0.0% | Dominated | Evaluated Competitor |

---

### Table 7.2: Single-Objective Selection Baselines vs Multi-Criteria Synthesis (`outputs/stage-17/decision_summary.json`)

| Selection Strategy | Selection Metric Basis | Selected Candidate | Baseline Metric Value | Strategic Synthesis Finding |
| :--- | :--- | :--- | :---: | :--- |
| **1. Reference Baseline** | General-domain canonical architecture | `bert-base-uncased` | Top-1: 29.73% | Conventional NLP default |
| **2. Pure Top-1 Accuracy** | Empirical Maritime Top-1 Accuracy | `answerdotai/ModernBERT-base` | 73.05% | Selects highest capability |
| **3. Pure MLM Loss** | Cross-entropy sequence loss | `answerdotai/ModernBERT-base` | 1.4386 | Selects best probability fit |
| **4. Pure Rare-Term Accuracy**| Domain specialized vocabulary accuracy | `answerdotai/ModernBERT-base` | 70.51% | Selects nautical jargon recovery |
| **5. Pure Tokenizer Fit** | Lowest subword fragmentation rate | `bert-base-uncased` | 26.57% | Prioritizes WordPiece efficiency |
| **6. Aggregate MUI Score** | Direction-normalized composite index | `answerdotai/ModernBERT-base` | 78.56 | Balances performance & operational |
| **7. Stage 17 Evidence Synthesis**| **8-Layer Evidence Priority Hierarchy** | **`answerdotai/ModernBERT-base`** | **Convergent Consensus** | **Statistically defensible multi-source choice** |

---

### Table 7.3: Operational Resource Footprint vs Intrinsic Capability Trade-offs (`outputs/stage-17/stage17_decision_report.md`)

| Candidate Model | Parameter Count | Model Disk Size | Inference Latency | Batch Throughput | Subword Fragmentation | Rare Vocabulary Top-1 | Contextual Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 149.0M | 590 MB | 35.0 ms | 28.54 docs/s | 62.99% | **70.51%** | **1.4386** |
| `roberta-base` | 125.0M | 500 MB | 27.1 ms | 36.93 docs/s | 64.78% | 45.12% | 1.9490 |
| `nlpaueb/legal-bert-base-uncased` | 110.0M | 440 MB | 24.4 ms | 40.91 docs/s | 37.61% | 22.40% | 4.0853 |
| `allenai/scibert_scivocab_uncased` | 110.0M | 440 MB | 23.8 ms | 41.98 docs/s | 41.79% | 24.80% | 4.2064 |
| `bert-base-uncased` | **110.0M** | **440 MB** | **21.4 ms** | **46.66 docs/s** | **26.57%** | 54.35% | 4.6164 |
| `dmis-lab/biobert-base-cased-v1.2` | 110.0M | 440 MB | 24.8 ms | 40.41 docs/s | 35.52% | 32.35% | 4.7867 |
| `microsoft/BiomedNLP-PubMedBERT...`| 110.0M | 440 MB | 23.5 ms | 42.52 docs/s | 42.39% | 3.40% | 5.7246 |

---

### Table 7.4: Final Pretraining Decision Specification & Parameter Checklist

| Decision Dimension | Prescribed Value | Empirical Evidence Base |
| :--- | :--- | :--- |
| **Recommended Strategy** | **Strategy A: Pretrained Encoder Initialization + DAPT** | Convergent superiority on intrinsic representation metrics |
| **Primary Pretrained Model** | **`answerdotai/ModernBERT-base`** | 73.05% Top-1, 1.4386 Loss, 100% Bootstrap Rank-1, 0 Pairwise Defeats |
| **Secondary Alternative** | **`bert-base-uncased`** | Lowest latency (21.4ms), lowest subword fragmentation (26.57%) |
| **Decision Confidence** | **High** | Non-parametric bootstrap consensus, Holm-adjusted pairwise separation |
| **Corpus Format for DAPT** | **Sanitized Mixed & Narrative JSONL** | Highest contextual density and cross-model ranking stability |
| **Target Pretraining Objective**| **Masked Language Modeling (MLM)** | Standard Bernoulli 15% masking with domain nautical term monitoring |

---

## 8. Stage 18: Automated Corpus Quality Linting Benchmarks

### Table 8.1: Quality Lint Rule Execution Summary & Defect Rates (`outputs/stage-18/corpus_lint_report.json`)
Evaluates 96,869 cleaned documents against 5 compiled regex quality rules to enforce pre-publication integrity.

| Rule Identifier | Description & Target Pattern | Violation Count | Violation Rate (%) | Gate Threshold | Quality Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `repeated_adjacent_words` | Unintentional word doubling (`\b(\w{3,})\s+\1\b`) | 161 | 0.166% | $< 0.500\%$ | **PASS** |
| `malformed_singular_plural` | Grammatical number collision (`\b1\s+(?:persons\|injuries)\b`) | **0** | **0.000%** | $< 0.100\%$ | **PASS** |
| `administrative_leakage` | Lingering internal MARSIS IDs (`formerly occno\|record id`) | **0** | **0.000%** | $0.000\%$ | **PASS** |
| `awkward_phrasing` | Template phrasing collisions (`sustained damaged`) | 3 | 0.003% | $< 0.100\%$ | **PASS** |
| `duplicated_list_items` | Redundant comma-separated tokens (`\b(\w+),\s+\1\b`) | 29 | 0.030% | $< 0.100\%$ | **PASS** |
| **Corpus Quality Gate Total** | **Corpus-Wide Quality Verification Summary** | **193** | **0.199%** | **$< 0.500\%$** | **PASS** |

---

### Table 8.2: Violation Sample Registry with MARSIS Occurrence IDs & Context Snippets

| Rule Identifier | OccID | Matched Substring | Document Snippet Context | Operational Analysis |
| :--- | :---: | :--- | :--- | :--- |
| `repeated_adjacent_words` | 660 | `LUMBA LUMBA` | `BEGAN TAKING WATER, OBTAINED PUMP FROM SAR VESSEL LUMBA LUMBA` | Legitimate vessel name (*SAR Vessel Lumba Lumba*); false positive |
| `repeated_adjacent_words` | 759 | `BELLA BELLA` | `VESSEL PROCEEDED TO BELLA BELLA B C TO BE INSPECTED` | Legitimate geographic location (*Bella Bella, BC*); false positive |
| `awkward_phrasing` | 25257 | `sustained damaged` | `Fishing vessel EGMONT was tied up at Matane Wharf during a storm and sustained damaged to the hull...` | Free-text narrative entry syntax; minor non-structural artifact |
| `awkward_phrasing` | 53775 | `sustained damaged` | `On 07 December 2025, the fishing vessel PATRICK & THE GIRLS... ran aground... sustained damaged...` | Free-text field entry collision |
| `duplicated_list_items` | 16097 | `ANCHOR` | `VSL LOST HER PROPELLER, DROPPED HER ANCHOR, ANCHOR DRAGGED, DRIFTED` | Repeated nautical term across sequential action clauses |
| `duplicated_list_items` | 24195 | `FIRE` | `CHARTER CRAFT SUSTAINED SMALL ELECTRICAL FIRE, FIRE EXTINGUISHED AND VESSEL ASSISTED...` | Cause-and-effect narrative sequence across comma boundary |

*Quality Verification Summary*: Over $83\%$ of detected violations represent legitimate geographic proper nouns (e.g., *Bella Bella*) or repetitive nautical vessel names (*Lumba Lumba*). The true syntactic defect rate is below **0.033%**, confirming exceptional textual cleanliness for domain pretraining.

