# Stage 17: Evidence-Synthesis & Pretrained Model Selection Report
**Project**: TSBC-MaritimePipeline-Version2.1  
**Generated**: 2026-09-10 11:01:30  
**Evaluation Architecture**: Stage 15 Compatibility + Stage 16 Statistical Robustness Synthesis

---

## 1. Executive Decision
* **Selected Pretrained Model**: `answerdotai/ModernBERT-base`
* **Recommended Decision**: **Recommended DAPT Candidate**
* **Recommended Action**: Proceed with Domain-Adaptive Pretraining (DAPT)
* **Decision Confidence**: **High**
* **Methodological Framing**: The empirical benchmark evidence supports `answerdotai/ModernBERT-base` as the preferred pretrained initialization for the subsequent Domain-Adaptive Pretraining (DAPT) experiment. This decision reflects strong intrinsic masked language modeling capability, structural ranking invariance across representations and subsets, and rigorous pairwise statistical support with zero observed defeats.

---

## 2. Evidence Summary
The model selection decision is grounded in empirical artifacts from Stages 15 and 16, without recalculating or fabricating metrics:

* **Primary Capability**: `answerdotai/ModernBERT-base` achieves an empirical Maritime Top-1 Accuracy of **56.04%** and an intrinsic MLM cross-entropy loss of **2.3063** (Pseudo-Perplexity: 13.43).
* **Statistical Standing**: Stage 16 Wilcoxon signed-rank testing with Holm-Bonferroni correction confirms that `answerdotai/ModernBERT-base` demonstrates statistically significant superiority over all 6 competing evaluated models ($p_{holm} < 0.05$) with large parametric (Cohen's $d > 2.0$) and non-parametric (Cliff's $\delta > 0.8$) effect sizes.
* **Bootstrap Stability**: Across 2,000 bootstrap resamples of matched evaluation conditions, `answerdotai/ModernBERT-base` attained a Rank-1 probability of **100.0%** with a mean rank of **1.00 ± 0.00**.
* **Condition Robustness**: Rank 1 status was maintained across all 5 structural representations (Narrative, Key-Value, Template, JSON, Mixed) and all 5 semantic knowledge subsets.
* **Pareto Status**: Verified as non-dominated (**Pareto-Optimal**) in the 10-dimensional Stage 15 multi-objective evaluation.

---

## 3. Candidate Comparison
Models are categorized based on relative empirical evidence into three transparent tiers:

| Model Name | Candidate Status | Top-1 Accuracy (%) | MLM Loss | Mean Rank | $P(\text{Rank}=1)$ | Pareto Status | Decision Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `answerdotai/ModernBERT-base` | **Strong Candidate** | 56.04% | 2.3063 | 1.00 | 100.0% | Pareto-Optimal | Primary DAPT Candidate |
| `roberta-base` | **Strong Candidate** | 47.03% | 2.7948 | 2.00 | 0.0% | Pareto-Optimal | Runner-Up Capability Benchmark |
| `allenai/scibert_scivocab_uncased` | **Strong Candidate** | 31.24% | 4.2628 | 3.05 | 0.0% | Pareto-Optimal | Evaluated Competitor |
| `bert-base-uncased` | **Competitive Candidate** | 29.08% | 4.6237 | 4.34 | 0.0% | Pareto-Optimal | Resource-Constrained Alternative |
| `nlpaueb/legal-bert-base-uncased` | **Competitive Candidate** | 28.66% | 4.4541 | 4.61 | 0.0% | Pareto-Optimal | Evaluated Competitor |
| `dmis-lab/biobert-base-cased-v1.2` | **Weak Candidate** | 24.65% | 4.9969 | 6.00 | 0.0% | Pareto-Optimal | Evaluated Competitor |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **Weak Candidate** | 20.59% | 5.7259 | 7.00 | 0.0% | Dominated | Evaluated Competitor |

---

## 4. Robustness
The benchmark evaluated candidate models across diverse structural representations and domain knowledge subsets:

1. **Representation Invariance**:
   * Evaluated formats: `Narrative`, `Key-Value`, `Template`, `JSON`, and `Mixed`.
   * `answerdotai/ModernBERT-base` demonstrated perfect rank stability across all representations (Mean Rank: 1.0, Std: 0.00).
   * Overall representation ranking concordance is high (Kendall $\tau = 0.7143$).
2. **Knowledge Subset Agreement**:
   * Evaluated subsets: `Balanced Knowledge`, `High Knowledge`, `Medium Knowledge`, `Low Knowledge`, and `Random Baseline`.
   * `answerdotai/ModernBERT-base` maintained Rank 1 across all 5 subsets (Mean Rank: 1.0, Std: 0.00).
   * Knowledge subset ranking concordance demonstrates strong agreement (Kendall $\tau = 0.9238$).
3. **Bootstrap Resampling**:
   * 95% Confidence Interval for `answerdotai/ModernBERT-base` Top-1 accuracy: `[0.523, 0.5992]`.
   * Demonstrates complete confidence interval separation from all baseline models except runner-up general encoders.

---

## 5. Statistical Support
All statistical evidence is consumed directly from Stage 16 without re-computation:

* **Global Hypothesis Test**: Friedman's omnibus test across matched conditions confirms highly statistically significant differences among models ($\chi^2 = 123.19$, $df = 6$, $p = 3.48 \times 10^{-24}$).
* **Pairwise Wilcoxon Tests**: With family-wise error controlled using the Holm-Bonferroni step-down procedure, `answerdotai/ModernBERT-base` achieves statistically significant superiority over:
  * `allenai/scibert_scivocab_uncased` ($p_{holm} = 1.25 \times 10^{-6}$, Cliff's $\delta = -0.95$, Large)
  * `bert-base-uncased` ($p_{holm} = 1.25 \times 10^{-6}$, Cliff's $\delta = -0.94$, Large)
  * `dmis-lab/biobert-base-cased-v1.2` ($p_{holm} = 1.25 \times 10^{-6}$, Cliff's $\delta = -0.97$, Large)
  * `microsoft/BiomedNLP-PubMedBERT-base` ($p_{holm} = 1.25 \times 10^{-6}$, Cliff's $\delta = -0.97$, Large)
  * `nlpaueb/legal-bert-base-uncased` ($p_{holm} = 1.25 \times 10^{-6}$, Cliff's $\delta = -0.93$, Large)
  * `roberta-base` ($p_{holm} = 1.43 \times 10^{-5}$, Cliff's $\delta = -0.80$, Large)
* **Zero Empirical Defeats**: `answerdotai/ModernBERT-base` experienced 0 statistically significant pairwise defeats across all conditions.

---

## 6. MUI and Pareto Evidence
* **MUI Role**: The Maritime Understanding Index (MUI) is utilized exclusively as a supporting aggregate index, not as an unchallengeable ground truth.
* **MUI Baseline**: `answerdotai/ModernBERT-base` ranked 1st with a baseline score of **68.29**.
* **Sensitivity Analysis Findings**:
  * `answerdotai/ModernBERT-base` won 2 of 4 weight scenarios (Baseline and Performance-Heavy).
  * `bert-base-uncased` won 2 of 4 scenarios (Domain-Heavy and Balanced), driven by its low tokenizer fragmentation and high rare vocabulary coverage.
  * *Methodological Insight*: This scenario divergence highlights a clear structural trade-off between intrinsic language modeling capability and tokenization efficiency, rather than a methodology defect.
* **Pareto Status**: Verified as **Pareto-Optimal** in Stage 15 across 10 evaluation objectives. Only one model (`microsoft/BiomedNLP-PubMedBERT-base`) was strictly Pareto-dominated.

---

## 7. Resource Trade-offs
Operational dimensions are documented transparently and kept distinct from capability metrics:

| Model Name | Parameters (M) | Model Disk Size (MB) | Inference Latency (ms) | Throughput (docs/sec) | Subword Fragmentation (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `answerdotai/ModernBERT-base` | 149M | 590 MB | 458.6 ms | 2.18 | 63.28% |
| `roberta-base` | 125M | 500 MB | 382.8 ms | 2.61 | 65.07% |
| `allenai/scibert_scivocab_uncased` | 110M | 440 MB | 323.4 ms | 3.09 | 42.09% |
| `bert-base-uncased` | 110M | 440 MB | 174.9 ms | 5.72 | 26.57% |
| `nlpaueb/legal-bert-base-uncased` | 110M | 440 MB | 249.3 ms | 4.01 | 37.91% |
| `dmis-lab/biobert-base-cased-v1.2` | 110M | 440 MB | 154.1 ms | 6.49 | 35.52% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 110M | 440 MB | 326.8 ms | 3.06 | 42.69% |

### Operational Trade-off Analysis:
* **Selected Candidate (`answerdotai/ModernBERT-base`)**: Demonstrates leading language modeling representation capability, but incurs a higher latency (458.6ms) and higher subword fragmentation (63.28%) than older BERT architectures.
* **Resource-Constrained Alternative (`bert-base-uncased`)**: Offers 2.6x lower inference latency (174.9ms vs 458.6ms), lower parameter footprint (110M vs 149M), and substantially lower subword fragmentation (26.57% vs 63.28%), making it the preferred candidate under constrained deployment budgets.

---

## 8. Selection Baselines
To assess the impact of the multi-criteria evidence framework, we record the candidate that would be chosen by simple single-objective selection strategies:

| Selection Strategy | Selection Metric | Selected Candidate | Metric Value |
| :--- | :--- | :--- | :--- |
| **Baseline / Reference** | Canonical general-domain pretrained baseline reference | `bert-base-uncased` | Top-1: 29.08% |
| **Highest Top-1 Accuracy** | Empirical Maritime Top-1 Accuracy | `answerdotai/ModernBERT-base` | 56.04% |
| **Lowest MLM Loss** | Intrinsic Masked Language Modeling Cross-Entropy Loss | `answerdotai/ModernBERT-base` | 2.3063 |
| **Highest Rare-Domain Accuracy** | Domain Specialized Vocabulary Top-1 Accuracy | `bert-base-uncased` | 54.35% |
| **Best Tokenizer Fit** | Lowest Subword Tokenizer Fragmentation Rate | `bert-base-uncased` | 26.57% |
| **MUI (Baseline Aggregate Index)** | Stage 15 Maritime Understanding Index (Baseline) | `answerdotai/ModernBERT-base` | 68.29 |
| **Evidence-Based Selection (Stage 17)** | Multi-Dimensional Evidence Priority Hierarchy | `answerdotai/ModernBERT-base` | Convergent Consensus |

* **Key Finding**: Simple single-criterion strategies yield divergent choices: pure accuracy and loss metrics select `answerdotai/ModernBERT-base`, whereas pure vocabulary fit selects `bert-base-uncased`. The Stage 17 multi-criteria hierarchy transparently synthesizes these trade-offs rather than arbitrarily collapsing them into a single opaque score.

---

## 9. Final Recommendation
1. **Primary Recommendation**: Proceed with Domain-Adaptive Pretraining (DAPT) utilizing **`answerdotai/ModernBERT-base`** as the pretrained initialization.
2. **Secondary Recommendation (Resource-Constrained)**: Retain **`bert-base-uncased`** as the candidate initialization when inference latency, compute budget, or tokenization fragmentation is the primary deployment constraint.
3. **Action Plan**:
   * Initialize DAPT on the full sanitized maritime incident corpus using `answerdotai/ModernBERT-base`.
   * Monitor subword fertility on domain-specific maritime jargon during continual pretraining.
   * Evaluate learning dynamics and convergence compared against the baseline `bert-base-uncased` checkpoint.

---

## 10. Limitations
* **Benchmark Scope**: The intrinsic benchmark establishes that `answerdotai/ModernBERT-base` possesses superior zero-shot language representation of maritime English text under MLM masking conditions. It does **not** prove that `answerdotai/ModernBERT-base` is mathematically guaranteed to achieve optimal downstream task performance.
* **DAPT Empirical Requirement**: True task optimality can only be verified empirically through downstream fine-tuning evaluations (e.g. classification, named entity recognition, incident cause extraction) conducted following the DAPT phase.
* **No Score Conflation**: Stage 17 intentionally avoids generating a new composite 'selection score', preserving full transparency and auditability for peer review.
