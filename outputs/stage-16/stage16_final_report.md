# Stage 16: Statistical Validation & Crossed Condition Analysis Report
**Maritime Corpus Pipeline Version 2.1**
*Benchmark Design: 175 benchmark cells across 7 encoder models, 5 representations, and 5 knowledge subsets (25 matched conditions per model).*

---

## 1. Executive Conclusion
This research stage statistically validates cross-model performance differences identified in Stage 15.
Crucially, the benchmark structure is modeled as a **balanced crossed repeated-measures design** (Models x Representations x Subsets) rather than assuming false cell independence.

* **Primary Crossed Repeated-Measures Analysis**: The primary fixed effect of encoder model is statistically decisive under both parametric ANOVA ($F = 1520.415$, $p = 1.1102e-16$, $\eta^2 = 70.3\%$) and 1,000 block-respecting condition permutations ($p_{\text{perm}} = 0.0010$).
* **Secondary Omnibus Friedman Test**: Retained for reference and historical continuity, the Friedman test confirms significant differences across matched conditions (Friedman $\chi^2 = 129.2229$, $p = 1.8735e-25$, $df = 6$).
* **Pairwise Matched Comparisons**: Across all 21 paired comparisons, 18 pairs demonstrate statistically reliable differences after family-wise Holm-Bonferroni correction ($p_{\text{Holm}} < 0.05$).
* **Primary Winner Robustness**: Model `answerdotai/ModernBERT-base` demonstrates unambiguous statistical superiority, attaining an empirical bootstrap rank-1 frequency of **$P(\text{rank}=1) = 100.0%$** across 2000 condition resamples.
* **Paired vs. Unpaired Effect Sizes**: High paired rank-biserial correlations ($r_{\text{prb}} > 0.8$) and large paired Cohen's $d_z > 2.0$ confirm substantial practical margins on matched cells. Secondary unpaired Cliff's delta values are strictly reported as descriptive distribution-level statistics.

---

## 2. Crossed Factorial Analysis (Primary Repeated-Measures ANOVA)
The 25 benchmark configurations per model share underlying documents, representations, and knowledge subsets.
The primary statistical model is a **3-way crossed repeated-measures ANOVA** with block-respecting permutation testing.

| Factor / Variation Source | Sum of Squares | $df$ | Mean Square | $F$-Statistic | $p$-value | Variance Contribution ($\eta^2$) | Partial $\eta^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model (Fixed Effect) | 6.0005 | 6 | 1.0001 | **1520.41** | **1.1102e-16** | 70.29% | 0.9896 |
| Representation (Blocking Factor) | 1.7446 | 4 | 0.4361 | **663.06** | **1.1102e-16** | 20.44% | 0.9651 |
| Subset (Blocking Factor) | 0.0569 | 4 | 0.0142 | **21.62** | **9.6778e-13** | 0.67% | 0.4739 |
| Model x Representation | 0.6078 | 24 | 0.0253 | **38.50** | **1.1102e-16** | 7.12% | 0.9059 |
| Model x Subset | 0.0209 | 24 | 0.0009 | **1.32** | **1.7146e-01** | 0.24% | 0.2484 |
| Representation x Subset | 0.0429 | 16 | 0.0027 | **4.08** | **6.9774e-06** | 0.50% | 0.4047 |
| Residual (Model x Rep x Subset) | 0.0631 | 96 | 0.0007 | — | — | 0.74% | — |

* **Block-Respecting Permutation Test ($p_{\text{perm}}$)**: **0.0010** (exact permutation of model labels within each of the 25 joint representation x subset blocks across 1,000 resamples).
* *Interpretation*: Model architecture accounts for the dominant share of benchmark variance (70.3%), confirming that model superiority is structural rather than an artifact of condition selection.

### Secondary Reference: Friedman Omnibus Test
Retained for continuity as a secondary nonparametric baseline across matched conditions:
* **Friedman $\chi^2$**: 129.2229 ($df = 6$, $p = 1.873454e-25$)
* **Decision**: Statistically Significant ($p < 0.001$).

---

## 3. Pairwise Matched Comparisons
Pairwise non-parametric **Wilcoxon signed-rank tests** were conducted on matched paired cell differences ($A_i - B_i$).
Multiple comparisons are controlled via the **Holm-Bonferroni step-down procedure** across the family of 21 comparisons.

| Model A | Model B | Paired Cells | Wins | Losses | Ties | Paired Win Rate | Paired Rank-Biserial | Wilcoxon Stat | Raw $p$ | Holm $p$ | Holm Sig? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | 25 | 19 | 6 | 0 | 76.0% | +0.54 | 75.0 | 0.017312 | 0.051937 | No |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | 15 | 10 | 0 | 60.0% | +0.06 | 153.0 | 0.81193 | 0.81193 | No |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `roberta-base` | 25 | 20 | 5 | 0 | 80.0% | +0.78 | 35.0 | 0.00025 | 0.001249 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 20 | 5 | 0 | 80.0% | +0.61 | 63.0 | 0.006129 | 0.024517 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 24 | 1 | 0 | 96.0% | +0.97 | 5.0 | 5.960464e-07 | 4.172325e-06 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | 12 | 13 | 0 | 48.0% | -0.37 | 102.0 | 0.107315 | 0.214629 | No |
| `bert-base-uncased` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | 25 | 2 | 23 | 0 | 8.0% | -0.98 | 3.0 | 2.980232e-07 | 2.384186e-06 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | 25 | 4 | 21 | 0 | 16.0% | -0.91 | 14.0 | 6.556511e-06 | 3.933907e-05 | **Yes ($p < 0.05$)** |

---

## 4. Effect Sizes: Paired vs. Unpaired Statistics
To address methodological confounding, effect sizes are strictly partitioned into:
1. **Paired Statistics (Primary)**: Matched Cohen's $d_z$ and Kerby's paired rank-biserial correlation ($r_{\text{prb}} = \frac{W^+ - W^-}{W^+ + W^-}$).
2. **Unpaired Statistics (Secondary/Descriptive)**: Cliff's Delta ($\delta$). Cliff's delta is an unpaired distribution-level statistic and does NOT measure matched-cell dominance.

| Model A | Model B | Mean Diff | 95% Paired CI | Paired $r_{\text{prb}}$ | Paired $d_z$ | Unpaired Cliff's $\delta$ | Practical Importance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | -0.4070 | [-0.4478, -0.3662] | -1.00 (large) | -4.12 | -1.00 (large) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | +0.0294 | [+0.0047, +0.0540] | +0.54 (large) | +0.49 | +0.19 (small) | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0696 | [+0.0572, +0.0819] | +1.00 (large) | +2.33 | +0.40 (medium) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.1198 | [+0.1084, +0.1312] | +1.00 (large) | +4.35 | +0.52 (large) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | -0.0090 | [-0.0419, +0.0239] | +0.06 (negligible) | -0.11 | +0.08 (negligible) | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | -0.2991 | [-0.3431, -0.2550] | -1.00 (large) | -2.80 | -0.69 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | +0.4364 | [+0.3898, +0.4830] | +1.00 (large) | +3.87 | +1.00 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | +0.4766 | [+0.4322, +0.5209] | +1.00 (large) | +4.43 | +1.00 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.5268 | [+0.4815, +0.5720] | +1.00 (large) | +4.81 | +1.00 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | +0.3980 | [+0.3544, +0.4415] | +1.00 (large) | +3.77 | +0.99 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `roberta-base` | +0.1079 | [+0.0495, +0.1663] | +0.78 (large) | +0.76 | +0.29 (small) | Substantial practical advantage |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0402 | [+0.0142, +0.0662] | +0.61 (large) | +0.64 | +0.31 (small) | Substantial practical advantage |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0904 | [+0.0616, +0.1192] | +0.97 (large) | +1.30 | +0.43 (medium) | Substantial practical advantage |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | -0.0384 | [-0.0750, -0.0018] | -0.37 (medium) | -0.43 | -0.07 (negligible) | No statistically reliable difference |
| `bert-base-uncased` | `roberta-base` | -0.3285 | [-0.3770, -0.2799] | -1.00 (large) | -2.79 | -0.71 (large) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0502 | [+0.0420, +0.0585] | +1.00 (large) | +2.52 | +0.28 (small) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | -0.0786 | [-0.1069, -0.0503] | -0.98 (large) | -1.15 | -0.50 (large) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | -0.3686 | [-0.4193, -0.3180] | -1.00 (large) | -3.00 | -0.83 (large) | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | -0.1288 | [-0.1573, -0.1003] | -1.00 (large) | -1.87 | -0.65 (large) | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | -0.4189 | [-0.4705, -0.3673] | -1.00 (large) | -3.35 | -0.89 (large) | Substantial practical advantage |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | -0.2901 | [-0.3622, -0.2179] | -0.91 (large) | -1.66 | -0.66 (large) | Substantial practical advantage |

---

## 5. Bootstrap Uncertainty
Deterministic bootstrap resampling ($B = 2,000$, seed = 42) of the matched benchmark conditions provides non-parametric 95% confidence intervals for individual model Top-1 accuracy and paired differences.

### Model Score 95% Confidence Intervals
| Model Name | Bootstrap Mean | 95% CI Lower | 95% CI Upper | Resamples | Matched Conditions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | 0.3210 | 0.2762 | 0.3632 | 2000 | 25 |
| `answerdotai/ModernBERT-base` | 0.7283 | 0.6825 | 0.7745 | 2000 | 25 |
| `bert-base-uncased` | 0.2920 | 0.2492 | 0.3305 | 2000 | 25 |
| `dmis-lab/biobert-base-cased-v1.2` | 0.2515 | 0.2072 | 0.2956 | 2000 | 25 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 0.2013 | 0.1575 | 0.2463 | 2000 | 25 |
| `nlpaueb/legal-bert-base-uncased` | 0.3303 | 0.3025 | 0.3615 | 2000 | 25 |
| `roberta-base` | 0.6202 | 0.5412 | 0.6875 | 2000 | 25 |

---

## 6. Bootstrap Rank Stability
In each of the 2,000 bootstrap resamples, all models were evaluated across the sampled configurations and ranked.
$P(\text{rank}=1)$ denotes the proportion of resamples in which the model ranked first.

| Model Name | Mean Rank | Rank SD | $P(\text{rank}=1)$ | $P(\text{rank}=2)$ | $P(\text{rank} \le 3)$ | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `answerdotai/ModernBERT-base` | **1.00** | ±0.00 | **1.0000** | 0.0000 | 1.0000 | **Decisive Leader** ($P_1 > 95\%$) |
| `roberta-base` | **2.00** | ±0.00 | **0.0000** | 1.0000 | 1.0000 | **Strong Second** ($P_2 > 95\%$) |
| `nlpaueb/legal-bert-base-uncased` | **3.30** | ±0.48 | **0.0000** | 0.0000 | 0.7150 | Mid/Lower Tier Encoder |
| `allenai/scibert_scivocab_uncased` | **3.73** | ±0.46 | **0.0000** | 0.0000 | 0.2810 | Mid/Lower Tier Encoder |
| `bert-base-uncased` | **4.98** | ±0.17 | **0.0000** | 0.0000 | 0.0040 | Mid/Lower Tier Encoder |
| `dmis-lab/biobert-base-cased-v1.2` | **6.00** | ±0.00 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **7.00** | ±0.00 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |

*Note: Empirical bootstrap rank frequencies represent resampling stability under matched condition perturbation, not Bayesian posterior probabilities of absolute domain capability.*

---

## 7. Representation Robustness
To assess whether structural representation shifts alter model hierarchies, model performances were aggregated across representations and compared against global benchmark ranks.

| Representation | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Json** | `answerdotai/ModernBERT-base` | 0.6156 | 0.9643 | 0.9048 | High ranking stability |
| **Key_value** | `answerdotai/ModernBERT-base` | 0.8561 | 0.8929 | 0.8095 | Moderate ranking stability |
| **Mixed** | `answerdotai/ModernBERT-base` | 0.8506 | 0.9286 | 0.8095 | High ranking stability |
| **Narrative** | `answerdotai/ModernBERT-base` | 0.7264 | 0.8929 | 0.8095 | Moderate ranking stability |
| **Template** | `roberta-base` | 0.6420 | 0.8571 | 0.7143 | Moderate ranking stability |

---

## 8. Subset Robustness
Evaluations across knowledge-classified subsets evaluate whether domain-informativeness stratification produces rank inversions or disparate encoder behavior.

| Knowledge Subset | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **balanced_knowledge** | `answerdotai/ModernBERT-base` | 0.7222 | 1.0000 | 1.0000 | High ranking stability |
| **high_knowledge** | `answerdotai/ModernBERT-base` | 0.7440 | 1.0000 | 1.0000 | High ranking stability |
| **low_knowledge** | `answerdotai/ModernBERT-base` | 0.7288 | 0.9643 | 0.9048 | High ranking stability |
| **medium_knowledge** | `answerdotai/ModernBERT-base` | 0.7513 | 1.0000 | 1.0000 | High ranking stability |
| **random_baseline** | `answerdotai/ModernBERT-base` | 0.6947 | 1.0000 | 1.0000 | High ranking stability |

---

## 9. Stage 12 Component Sensitivity Analysis
Leave-one-dimension-out ablation on the Stage 12 Domain Informativeness Engine evaluates how the removal of individual observable signals affects document scores, ranking correlation, and Top-20% document selection (Jaccard index).

### Component Score & Ranking Sensitivity
| Ablated Component | Mean $\Delta_{{\text{{score}}}}$ | Median $\Delta$ | SD $\Delta$ | Spearman $\rho$ | Kendall $\tau$ | Top-20% Jaccard Overlap | Selection Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `domain_relevance` | -0.0587 | -0.0616 | 0.0245 | 0.9634 | 0.8425 | **0.6707** | Moderate selection impact: component alters borderline selections |
| `information_content` | +0.1942 | +0.1989 | 0.0328 | 0.9357 | 0.8454 | **0.8901** | Low selection impact: score shifts have limited effect on top document selection |
| `tfidf_representativeness` | -0.0032 | +0.0031 | 0.0422 | 0.8624 | 0.7269 | **0.6118** | Moderate selection impact: component alters borderline selections |
| `redundancy_noise` | -0.1487 | -0.1483 | 0.0557 | 0.8601 | 0.6735 | **0.5003** | High selection impact: component strongly influences document selection |

### Stratum / Tier Transition Stability
| Ablated Component | Same Tier (%) | High Tier Retention (%) | Medium Tier Retention (%) | Low Tier Retention (%) | Documents Shifted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `domain_relevance` | 89.67% | **80.28%** | 91.42% | 93.81% | 10,008 / 96,860 |
| `information_content` | 81.19% | **94.15%** | 74.67% | 87.77% | 18,218 / 96,860 |
| `tfidf_representativeness` | 76.15% | **75.89%** | 80.14% | 64.46% | 23,103 / 96,860 |
| `redundancy_noise` | 66.63% | **66.68%** | 72.20% | 49.89% | 32,325 / 96,860 |

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
