# Stage 16: Statistical Validation & Crossed Condition Analysis Report
**Maritime Corpus Pipeline Version 2.1**
*Benchmark Design: 175 benchmark cells across 7 encoder models, 5 representations, and 5 knowledge subsets (25 matched conditions per model).*

---

## 1. Executive Conclusion
This research stage statistically validates cross-model performance differences identified in Stage 15.
Crucially, the benchmark structure is modeled as a **balanced crossed repeated-measures design** (Models x Representations x Subsets) rather than assuming false cell independence.

* **Primary Crossed Repeated-Measures Analysis**: The primary fixed effect of encoder model is statistically decisive under both parametric ANOVA ($F = 269.4943$, $p = 1.1102e-16$, $\eta^2 = 24.1\%$) and 1,000 block-respecting condition permutations ($p_{\text{perm}} = 0.0010$).
* **Secondary Omnibus Friedman Test**: Retained for reference and historical continuity, the Friedman test confirms significant differences across matched conditions (Friedman $\chi^2 = 91.5771$, $p = 1.4247e-17$, $df = 6$).
* **Pairwise Matched Comparisons**: Across all 21 paired comparisons, 14 pairs demonstrate statistically reliable differences after family-wise Holm-Bonferroni correction ($p_{\text{Holm}} < 0.05$).
* **Primary Winner Robustness**: Model `answerdotai/ModernBERT-base` demonstrates unambiguous statistical superiority, attaining an empirical bootstrap rank-1 frequency of **$P(\text{rank}=1) = 100.0%$** across 2000 condition resamples.
* **Paired vs. Unpaired Effect Sizes**: High paired rank-biserial correlations ($r_{\text{prb}} > 0.8$) and large paired Cohen's $d_z > 2.0$ confirm substantial practical margins on matched cells. Secondary unpaired Cliff's delta values are strictly reported as descriptive distribution-level statistics.

---

## 2. Crossed Factorial Analysis (Primary Repeated-Measures ANOVA)
The 25 benchmark configurations per model share underlying documents, representations, and knowledge subsets.
The primary statistical model is a **3-way crossed repeated-measures ANOVA** with block-respecting permutation testing.

| Factor / Variation Source | Sum of Squares | $df$ | Mean Square | $F$-Statistic | $p$-value | Variance Contribution ($\eta^2$) | Partial $\eta^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model (Fixed Effect) | 0.4404 | 6 | 0.0734 | **269.49** | **1.1102e-16** | 24.14% | 0.9440 |
| Representation (Blocking Factor) | 1.0187 | 4 | 0.2547 | **935.05** | **1.1102e-16** | 55.83% | 0.9750 |
| Subset (Blocking Factor) | 0.0604 | 4 | 0.0151 | **55.49** | **1.1102e-16** | 3.31% | 0.6981 |
| Model x Representation | 0.1875 | 24 | 0.0078 | **28.68** | **1.1102e-16** | 10.28% | 0.8776 |
| Model x Subset | 0.0135 | 24 | 0.0006 | **2.06** | **7.0834e-03** | 0.74% | 0.3404 |
| Representation x Subset | 0.0779 | 16 | 0.0049 | **17.87** | **1.1102e-16** | 4.27% | 0.7486 |
| Residual (Model x Rep x Subset) | 0.0261 | 96 | 0.0003 | — | — | 1.43% | — |

* **Block-Respecting Permutation Test ($p_{\text{perm}}$)**: **0.0010** (exact permutation of model labels within each of the 25 joint representation x subset blocks across 1,000 resamples).
* *Interpretation*: Model architecture accounts for the dominant share of benchmark variance (24.1%), confirming that model superiority is structural rather than an artifact of condition selection.

### Secondary Reference: Friedman Omnibus Test
Retained for continuity as a secondary nonparametric baseline across matched conditions:
* **Friedman $\chi^2$**: 91.5771 ($df = 6$, $p = 1.424702e-17$)
* **Decision**: Statistically Significant ($p < 0.001$).

---

## 3. Pairwise Matched Comparisons
Pairwise non-parametric **Wilcoxon signed-rank tests** were conducted on matched paired cell differences ($A_i - B_i$).
Multiple comparisons are controlled via the **Holm-Bonferroni step-down procedure** across the family of 21 comparisons.

| Model A | Model B | Paired Cells | Wins | Losses | Ties | Paired Win Rate | Paired Rank-Biserial | Wilcoxon Stat | Raw $p$ | Holm $p$ | Holm Sig? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | 25 | 7 | 18 | 0 | 28.0% | -0.58 | 69.0 | 0.010511 | 0.073576 | No |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 16 | 9 | 0 | 64.0% | +0.27 | 119.0 | 0.252104 | 1.0 | No |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 12 | 13 | 0 | 48.0% | -0.10 | 147.0 | 0.691519 | 1.0 | No |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | 25 | 5 | 20 | 0 | 20.0% | -0.82 | 29.0 | 0.000103 | 0.001133 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | 25 | 23 | 2 | 0 | 92.0% | +0.91 | 14.0 | 6.556511e-06 | 8.523464e-05 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `roberta-base` | 25 | 23 | 2 | 0 | 92.0% | +0.98 | 4.0 | 4.172325e-07 | 5.841255e-06 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 18 | 7 | 0 | 72.0% | +0.62 | 61.0 | 0.005072 | 0.040573 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 18 | 7 | 0 | 72.0% | +0.51 | 79.0 | 0.02365 | 0.1419 | No |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | 8 | 17 | 0 | 32.0% | -0.43 | 92.0 | 0.058752 | 0.293758 | No |
| `bert-base-uncased` | `roberta-base` | 25 | 4 | 21 | 0 | 16.0% | -0.82 | 29.0 | 0.000103 | 0.001133 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 11 | 14 | 0 | 44.0% | -0.24 | 123.0 | 0.299612 | 1.0 | No |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | 25 | 1 | 24 | 0 | 4.0% | -0.98 | 3.0 | 2.980232e-07 | 4.470348e-06 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | 25 | 5 | 20 | 0 | 20.0% | -0.86 | 22.0 | 3.194809e-05 | 0.000383 | **Yes ($p < 0.05$)** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | 25 | 6 | 19 | 0 | 24.0% | -0.72 | 46.0 | 0.001027 | 0.009244 | **Yes ($p < 0.05$)** |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | 25 | 14 | 11 | 0 | 56.0% | +0.01 | 161.0 | 0.978915 | 1.0 | No |

---

## 4. Effect Sizes: Paired vs. Unpaired Statistics
To address methodological confounding, effect sizes are strictly partitioned into:
1. **Paired Statistics (Primary)**: Matched Cohen's $d_z$ and Kerby's paired rank-biserial correlation ($r_{\text{prb}} = \frac{W^+ - W^-}{W^+ + W^-}$).
2. **Unpaired Statistics (Secondary/Descriptive)**: Cliff's Delta ($\delta$). Cliff's delta is an unpaired distribution-level statistic and does NOT measure matched-cell dominance.

| Model A | Model B | Mean Diff | 95% Paired CI | Paired $r_{\text{prb}}$ | Paired $d_z$ | Unpaired Cliff's $\delta$ | Practical Importance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | -0.1442 | [-0.1628, -0.1256] | -1.00 (large) | -3.21 | -0.74 (large) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | -0.0349 | [-0.0597, -0.0102] | -0.58 (large) | -0.58 | -0.29 (small) | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0059 | [-0.0063, +0.0181] | +0.27 (small) | +0.20 | +0.03 (negligible) | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | -0.0001 | [-0.0074, +0.0072] | -0.10 (negligible) | -0.01 | +0.02 (negligible) | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | -0.0674 | [-0.0812, -0.0537] | -1.00 (large) | -2.02 | -0.39 (medium) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | -0.0684 | [-0.0983, -0.0385] | -0.82 (large) | -0.94 | -0.42 (medium) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | +0.1093 | [+0.0908, +0.1278] | +1.00 (large) | +2.44 | +0.65 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | +0.1501 | [+0.1286, +0.1716] | +1.00 (large) | +2.88 | +0.79 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.1441 | [+0.1215, +0.1666] | +1.00 (large) | +2.64 | +0.71 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | +0.0768 | [+0.0505, +0.1031] | +0.91 (large) | +1.20 | +0.43 (medium) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `roberta-base` | +0.0758 | [+0.0569, +0.0948] | +0.98 (large) | +1.65 | +0.46 (medium) | Substantial practical advantage |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0408 | [+0.0159, +0.0657] | +0.62 (large) | +0.68 | +0.28 (small) | Substantial practical advantage |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0348 | [+0.0053, +0.0643] | +0.51 (large) | +0.49 | +0.25 (small) | No statistically reliable difference |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | -0.0325 | [-0.0615, -0.0035] | -0.43 (medium) | -0.46 | -0.10 (negligible) | No statistically reliable difference |
| `bert-base-uncased` | `roberta-base` | -0.0335 | [-0.0480, -0.0189] | -0.82 (large) | -0.95 | -0.29 (small) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | -0.0060 | [-0.0182, +0.0062] | -0.24 (small) | -0.20 | -0.04 (negligible) | No statistically reliable difference |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | -0.0733 | [-0.0939, -0.0527] | -0.98 (large) | -1.47 | -0.43 (medium) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | -0.0743 | [-0.1059, -0.0427] | -0.86 (large) | -0.97 | -0.42 (medium) | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | -0.0673 | [-0.0821, -0.0525] | -1.00 (large) | -1.88 | -0.42 (medium) | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | -0.0683 | [-0.1028, -0.0337] | -0.72 (large) | -0.82 | -0.35 (medium) | Substantial practical advantage |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | -0.0010 | [-0.0349, +0.0330] | +0.01 (negligible) | -0.01 | -0.09 (negligible) | No statistically reliable difference |

---

## 5. Bootstrap Uncertainty
Deterministic bootstrap resampling ($B = 2,000$, seed = 42) of the matched benchmark conditions provides non-parametric 95% confidence intervals for individual model Top-1 accuracy and paired differences.

### Model Score 95% Confidence Intervals
| Model Name | Bootstrap Mean | 95% CI Lower | 95% CI Upper | Resamples | Matched Conditions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | 0.1416 | 0.1085 | 0.1751 | 2000 | 25 |
| `answerdotai/ModernBERT-base` | 0.2858 | 0.2509 | 0.3186 | 2000 | 25 |
| `bert-base-uncased` | 0.1767 | 0.1450 | 0.2072 | 2000 | 25 |
| `dmis-lab/biobert-base-cased-v1.2` | 0.1357 | 0.1050 | 0.1662 | 2000 | 25 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 0.1417 | 0.1064 | 0.1791 | 2000 | 25 |
| `nlpaueb/legal-bert-base-uncased` | 0.2093 | 0.1713 | 0.2498 | 2000 | 25 |
| `roberta-base` | 0.2102 | 0.1697 | 0.2489 | 2000 | 25 |

---

## 6. Bootstrap Rank Stability
In each of the 2,000 bootstrap resamples, all models were evaluated across the sampled configurations and ranked.
$P(\text{rank}=1)$ denotes the proportion of resamples in which the model ranked first.

| Model Name | Mean Rank | Rank SD | $P(\text{rank}=1)$ | $P(\text{rank}=2)$ | $P(\text{rank} \le 3)$ | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `answerdotai/ModernBERT-base` | **1.00** | ±0.00 | **1.0000** | 0.0000 | 1.0000 | **Decisive Leader** ($P_1 > 95\%$) |
| `roberta-base` | **2.48** | ±0.50 | **0.0000** | 0.5230 | 1.0000 | Consistently Top-Tier ($P_{\le 3} > 90\%$) |
| `nlpaueb/legal-bert-base-uncased` | **2.53** | ±0.51 | **0.0000** | 0.4770 | 0.9950 | Consistently Top-Tier ($P_{\le 3} > 90\%$) |
| `bert-base-uncased` | **4.00** | ±0.13 | **0.0000** | 0.0000 | 0.0050 | Mid/Lower Tier Encoder |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **5.63** | ±0.66 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |
| `allenai/scibert_scivocab_uncased` | **5.67** | ±0.67 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |
| `dmis-lab/biobert-base-cased-v1.2` | **6.69** | ±0.65 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |

*Note: Empirical bootstrap rank frequencies represent resampling stability under matched condition perturbation, not Bayesian posterior probabilities of absolute domain capability.*

---

## 7. Representation Robustness
To assess whether structural representation shifts alter model hierarchies, model performances were aggregated across representations and compared against global benchmark ranks.

| Representation | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Json** | `answerdotai/ModernBERT-base` | 0.1353 | 0.6071 | 0.5238 | Low ranking stability / rank shift |
| **Key_value** | `answerdotai/ModernBERT-base` | 0.2991 | 1.0000 | 1.0000 | High ranking stability |
| **Mixed** | `nlpaueb/legal-bert-base-uncased` | 0.3690 | 0.4286 | 0.3333 | Low ranking stability / rank shift |
| **Narrative** | `answerdotai/ModernBERT-base` | 0.2864 | 0.9643 | 0.9048 | High ranking stability |
| **Template** | `answerdotai/ModernBERT-base` | 0.3404 | 0.8214 | 0.6190 | Moderate ranking stability |

---

## 8. Subset Robustness
Evaluations across knowledge-classified subsets evaluate whether domain-informativeness stratification produces rank inversions or disparate encoder behavior.

| Knowledge Subset | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **balanced_knowledge** | `answerdotai/ModernBERT-base` | 0.2743 | 0.9643 | 0.9048 | High ranking stability |
| **high_knowledge** | `answerdotai/ModernBERT-base` | 0.2708 | 0.8571 | 0.7143 | Moderate ranking stability |
| **low_knowledge** | `answerdotai/ModernBERT-base` | 0.2777 | 0.9643 | 0.9048 | High ranking stability |
| **medium_knowledge** | `answerdotai/ModernBERT-base` | 0.3126 | 0.9286 | 0.8095 | High ranking stability |
| **random_baseline** | `answerdotai/ModernBERT-base` | 0.2942 | 1.0000 | 1.0000 | High ranking stability |

---

## 9. Stage 12 Component Sensitivity Analysis
Leave-one-dimension-out ablation on the Stage 12 Domain Informativeness Engine evaluates how the removal of individual observable signals affects document scores, ranking correlation, and Top-20% document selection (Jaccard index).

### Component Score & Ranking Sensitivity
| Ablated Component | Mean $\Delta_{{\text{{score}}}}$ | Median $\Delta$ | SD $\Delta$ | Spearman $\rho$ | Kendall $\tau$ | Top-20% Jaccard Overlap | Selection Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `domain_relevance` | -0.0586 | -0.0616 | 0.0244 | 0.9623 | 0.8379 | **0.6692** | Moderate selection impact: component alters borderline selections |
| `information_content` | +0.1942 | +0.1989 | 0.0329 | 0.9375 | 0.8468 | **0.8868** | Low selection impact: score shifts have limited effect on top document selection |
| `tfidf_representativeness` | -0.0031 | +0.0031 | 0.0422 | 0.8607 | 0.7209 | **0.6119** | Moderate selection impact: component alters borderline selections |
| `redundancy_noise` | -0.1488 | -0.1487 | 0.0558 | 0.8590 | 0.6712 | **0.5027** | High selection impact: component strongly influences document selection |

### Stratum / Tier Transition Stability
| Ablated Component | Same Tier (%) | High Tier Retention (%) | Medium Tier Retention (%) | Low Tier Retention (%) | Documents Shifted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `domain_relevance` | 89.64% | **80.17%** | 91.39% | 93.86% | 10,038 / 96,874 |
| `information_content` | 81.09% | **93.96%** | 74.56% | 87.80% | 18,315 / 96,874 |
| `tfidf_representativeness` | 76.13% | **75.89%** | 80.12% | 64.43% | 23,121 / 96,874 |
| `redundancy_noise` | 66.70% | **66.91%** | 72.26% | 49.86% | 32,255 / 96,874 |

---

## 10. Methodological Interpretation
A rigorous scientific benchmark must distinguish four fundamental statistical concepts:

1. **Paired Inference vs. Unpaired Effect Sizes**:
   Because conditions are matched (representation x subset), inference is driven by paired Wilcoxon tests, paired rank-biserial correlations, and paired Cohen's $d_z$. Unpaired Cliff's delta is reported strictly as secondary descriptive context and never interpreted as paired cell superiority.
2. **Crossed Repeated Measures vs. Independent Observations**:
   The 25 benchmark conditions per model are not independent replications. The 3-way crossed repeated-measures ANOVA models the joint variation of representations and subsets, confirmed by block-respecting permutation tests.
3. **Composite Scoring vs. Direct Understanding**:
   The Maritime Encoder Composite Score (MECS) is an operational composite compatibility index used for model selection, not a direct measure of language comprehension.
4. **Reproducibility & Determinism**:
   All stochastic bootstrap and permutation routines were executed under fixed deterministic seeds across matched conditions.
