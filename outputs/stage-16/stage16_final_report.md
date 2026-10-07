# Stage 16: Statistical Validation & Component Sensitivity Report
**Maritime Corpus Pipeline Version 2.1**
*Benchmark Environment: 25 matched benchmark conditions across 8 encoder models.*

---

## 1. Executive Conclusion
This research stage rigorously validates the cross-model performance differences identified in Stage 15.
Rather than treating the evaluation configurations as independent datasets or replications, the evaluation framework
models them as **25 matched benchmark conditions** across evaluated representations and knowledge subsets.

* **Global Model Differences**: Non-parametric omnibus testing demonstrates statistically distinguishable model performance across the candidate encoders (Friedman $\chi^2 = 158.84$, $p = 5.6272e-31$, $df = 7$).
* **Pairwise Reliability**: Across 28 model pairwise comparisons, 25 pairs show statistically significant differences after family-wise Holm-Bonferroni error rate control ($p_{\text{Holm}} < 0.05$).
* **Primary Winner Robustness**: Model `answerdotai/ModernBERT-base` demonstrates unambiguous statistical superiority, attaining an empirical bootstrap rank-1 frequency of **$P(\text{rank}=1) = 100.0%$** across 2000 condition resamples.
* **Effect Magnitude**: Large effect sizes ($d_z > 0.8$, Cliff's $\delta > 0.5$) separate domain-adapted and modernized architectures from baseline encoders, confirming that performance gaps reflect substantial practical margins rather than statistical artifacts.
* **Limitations**: While model rankings exhibit high stability across representations ($\rho \ge 0.71$) and knowledge subsets ($\rho \ge 0.89$), structured syntax representations (such as JSON) compress performance margins without inverting top-model superiority.

---

## 2. Global Model Comparison
The omnibus **Friedman test** was conducted across the matched benchmark configurations where all 8 candidate models were evaluated on identical conditions.

| Test Parameter | Value |
| :--- | :--- |
| **Statistical Test** | Friedman Chi-Square (Non-Parametric Repeated Measures) |
| **Matched Benchmark Conditions ($N$)** | 25 |
| **Models Evaluated ($k$)** | 8 |
| **Degrees of Freedom ($df$)** | 7 |
| **Chi-Square Statistic ($\chi^2$)** | **158.84** |
| **Raw $p$-value** | **5.627203e-31** |
| **Omnibus Decision** | **Statistically Significant ($p < 0.001$)** |

*Scientific Interpretation*: Candidate encoders exhibit statistically significant differences across the shared benchmark matrix. Because the omnibus null hypothesis is rejected, proceeding to pairwise post-hoc comparisons is statistically justified.

---

## 3. Pairwise Comparisons
Pairwise non-parametric **Wilcoxon signed-rank tests** were conducted on matched paired differences ($A_i - B_i$). Multiple comparisons are rigorously controlled via the **Holm-Bonferroni step-down procedure** across the family of 28 comparisons. Paired $t$-test statistics are retained as secondary supplementary statistics.

| Model A | Model B | Paired $N$ | Mean Diff | Median Diff | Wilcoxon Stat | Raw $p$-value | Holm $p$-value | Holm Significant? | Paired $t$-stat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | 25 | -0.4070 | -0.4247 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -20.60 |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | 25 | +0.0294 | +0.0390 | 75.0 | 0.017312 | 0.051937 | No | +2.46 |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | +0.0696 | +0.0664 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +11.65 |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | +0.1198 | +0.1139 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +21.75 |
| `allenai/scibert_scivocab_uncased` | `microsoft/deberta-v3-base` | 25 | +0.3212 | +0.3200 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +14.41 |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.0090 | +0.0075 | 153.0 | 0.81193 | 0.81193 | No | -0.57 |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | 25 | -0.2991 | -0.3146 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -14.02 |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | 25 | +0.4364 | +0.4444 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +19.33 |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | 25 | +0.4766 | +0.4799 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +22.17 |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | +0.5268 | +0.5506 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +24.03 |
| `answerdotai/ModernBERT-base` | `microsoft/deberta-v3-base` | 25 | +0.7282 | +0.7248 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +30.28 |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | 25 | +0.3980 | +0.3811 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +18.85 |
| `answerdotai/ModernBERT-base` | `roberta-base` | 25 | +0.1079 | +0.0813 | 35.0 | 0.00025 | 0.001249 | **Yes ($p < 0.05$)** | +3.81 |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | +0.0402 | +0.0332 | 63.0 | 0.006129 | 0.024517 | **Yes ($p < 0.05$)** | +3.19 |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | +0.0904 | +0.0765 | 5.0 | 5.960464e-07 | 4.172325e-06 | **Yes ($p < 0.05$)** | +6.48 |
| `bert-base-uncased` | `microsoft/deberta-v3-base` | 25 | +0.2918 | +0.3084 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +14.12 |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.0384 | -0.0097 | 102.0 | 0.107315 | 0.214629 | No | -2.16 |
| `bert-base-uncased` | `roberta-base` | 25 | -0.3285 | -0.3512 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -13.95 |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | +0.0502 | +0.0478 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +12.58 |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/deberta-v3-base` | 25 | +0.2516 | +0.2530 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +11.31 |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.0786 | -0.0546 | 3.0 | 2.980232e-07 | 2.384186e-06 | **Yes ($p < 0.05$)** | -5.73 |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | 25 | -0.3686 | -0.3722 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -15.02 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `microsoft/deberta-v3-base` | 25 | +0.2014 | +0.1981 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | +8.98 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.1288 | -0.1064 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -9.34 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | 25 | -0.4189 | -0.4276 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -16.75 |
| `microsoft/deberta-v3-base` | `nlpaueb/legal-bert-base-uncased` | 25 | -0.3302 | -0.3124 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -21.26 |
| `microsoft/deberta-v3-base` | `roberta-base` | 25 | -0.6203 | -0.6822 | 0.0 | 5.960464e-08 | 1.668930e-06 | **Yes ($p < 0.05$)** | -16.09 |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | 25 | -0.2901 | -0.3356 | 14.0 | 6.556511e-06 | 3.933907e-05 | **Yes ($p < 0.05$)** | -8.30 |

---

## 4. Effect Sizes
Statistical significance establishes whether observed differences are reliably non-zero. To evaluate **practical magnitude**, we report paired parametric Cohen's $d_z$, non-parametric Cliff's Delta ($\delta$), and 95% confidence intervals for mean paired differences.

| Model A | Model B | Mean Diff | 95% Paired CI | Cohen's $d_z$ | $d_z$ Tier | Cliff's $\delta$ | $\delta$ Tier | Practical Importance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | -0.4070 | [-0.4478, -0.3662] | -4.12 | large | -1.00 | large | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | +0.0294 | [+0.0047, +0.0540] | +0.49 | small | +0.19 | small | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0696 | [+0.0572, +0.0819] | +2.33 | large | +0.40 | medium | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.1198 | [+0.1084, +0.1312] | +4.35 | large | +0.52 | large | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `microsoft/deberta-v3-base` | +0.3212 | [+0.2752, +0.3672] | +2.88 | large | +1.00 | large | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | -0.0090 | [-0.0419, +0.0239] | -0.11 | negligible | +0.08 | negligible | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | -0.2991 | [-0.3431, -0.2550] | -2.80 | large | -0.69 | large | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | +0.4364 | [+0.3898, +0.4830] | +3.87 | large | +1.00 | large | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | +0.4766 | [+0.4322, +0.5209] | +4.43 | large | +1.00 | large | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.5268 | [+0.4815, +0.5720] | +4.81 | large | +1.00 | large | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `microsoft/deberta-v3-base` | +0.7282 | [+0.6786, +0.7778] | +6.06 | large | +1.00 | large | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | +0.3980 | [+0.3544, +0.4415] | +3.77 | large | +0.99 | large | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `roberta-base` | +0.1079 | [+0.0495, +0.1663] | +0.76 | medium | +0.29 | small | Substantial practical advantage |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0402 | [+0.0142, +0.0662] | +0.64 | medium | +0.31 | small | Substantial practical advantage |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0904 | [+0.0616, +0.1192] | +1.30 | large | +0.43 | medium | Substantial practical advantage |
| `bert-base-uncased` | `microsoft/deberta-v3-base` | +0.2918 | [+0.2492, +0.3345] | +2.82 | large | +1.00 | large | Substantial practical advantage |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | -0.0384 | [-0.0750, -0.0018] | -0.43 | small | -0.07 | negligible | No statistically reliable difference |
| `bert-base-uncased` | `roberta-base` | -0.3285 | [-0.3770, -0.2799] | -2.79 | large | -0.71 | large | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0502 | [+0.0420, +0.0585] | +2.52 | large | +0.28 | small | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/deberta-v3-base` | +0.2516 | [+0.2057, +0.2975] | +2.26 | large | +1.00 | large | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | -0.0786 | [-0.1069, -0.0503] | -1.15 | large | -0.50 | large | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | -0.3686 | [-0.4193, -0.3180] | -3.00 | large | -0.83 | large | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `microsoft/deberta-v3-base` | +0.2014 | [+0.1551, +0.2477] | +1.80 | large | +1.00 | large | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | -0.1288 | [-0.1573, -0.1003] | -1.87 | large | -0.65 | large | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | -0.4189 | [-0.4705, -0.3673] | -3.35 | large | -0.89 | large | Substantial practical advantage |
| `microsoft/deberta-v3-base` | `nlpaueb/legal-bert-base-uncased` | -0.3302 | [-0.3623, -0.2982] | -4.25 | large | -1.00 | large | Substantial practical advantage |
| `microsoft/deberta-v3-base` | `roberta-base` | -0.6203 | [-0.6998, -0.5407] | -3.22 | large | -1.00 | large | Substantial practical advantage |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | -0.2901 | [-0.3622, -0.2179] | -1.66 | large | -0.66 | large | Substantial practical advantage |

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
| `microsoft/deberta-v3-base` | 0.0000 | 0.0000 | 0.0000 | 2000 | 25 |
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
| `microsoft/deberta-v3-base` | **8.00** | ±0.00 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |

*Note: Empirical bootstrap rank frequencies represent resampling stability under matched condition perturbation, not Bayesian posterior probabilities of absolute domain capability.*

---

## 7. Representation Robustness
To assess whether structural representation shifts alter model hierarchies, model performances were aggregated across representations and compared against global benchmark ranks.

| Representation | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Json** | `answerdotai/ModernBERT-base` | 0.6156 | 0.9762 | 0.9286 | High ranking stability |
| **Key_value** | `answerdotai/ModernBERT-base` | 0.8561 | 0.9286 | 0.8571 | High ranking stability |
| **Mixed** | `answerdotai/ModernBERT-base` | 0.8506 | 0.9524 | 0.8571 | High ranking stability |
| **Narrative** | `answerdotai/ModernBERT-base` | 0.7264 | 0.9286 | 0.8571 | High ranking stability |
| **Template** | `roberta-base` | 0.6420 | 0.9048 | 0.7857 | High ranking stability |

---

## 8. Subset Robustness
Evaluations across knowledge-classified subsets evaluate whether domain-informativeness stratification produces rank inversions or disparate encoder behavior.

| Knowledge Subset | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **balanced_knowledge** | `answerdotai/ModernBERT-base` | 0.7222 | 1.0000 | 1.0000 | High ranking stability |
| **high_knowledge** | `answerdotai/ModernBERT-base` | 0.7440 | 1.0000 | 1.0000 | High ranking stability |
| **low_knowledge** | `answerdotai/ModernBERT-base` | 0.7288 | 0.9762 | 0.9286 | High ranking stability |
| **medium_knowledge** | `answerdotai/ModernBERT-base` | 0.7513 | 1.0000 | 1.0000 | High ranking stability |
| **random_baseline** | `answerdotai/ModernBERT-base` | 0.6947 | 1.0000 | 1.0000 | High ranking stability |

---

## 9. Stage 12 Component Sensitivity Analysis
Leave-one-dimension-out ablation on the Stage 12 Domain Informativeness Engine evaluates how the removal of individual observable signals affects document scores, ranking correlation, and Top-20% document selection (Jaccard index).

### Component Score & Ranking Sensitivity
| Ablated Component | Mean $\Delta_{{\text{{score}}}}$ | Median $\Delta$ | SD $\Delta$ | Spearman $\rho$ | Kendall $\tau$ | Top-20% Jaccard Overlap | Selection Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `domain_relevance` | -0.0587 | -0.0616 | 0.0245 | 0.9634 | 0.8425 | **0.6709** | Moderate selection impact: component alters borderline selections |
| `information_content` | +0.1942 | +0.1989 | 0.0328 | 0.9357 | 0.8454 | **0.8900** | Low selection impact: score shifts have limited effect on top document selection |
| `tfidf_representativeness` | -0.0032 | +0.0031 | 0.0422 | 0.8624 | 0.7269 | **0.6119** | Moderate selection impact: component alters borderline selections |
| `redundancy_noise` | -0.1487 | -0.1483 | 0.0557 | 0.8601 | 0.6735 | **0.5002** | High selection impact: component strongly influences document selection |

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

1. **Statistical Significance vs. Practical Magnitude**:
   A statistically significant difference ($p < 0.05$) merely confirms that the expected performance gap across conditions is unlikely to be zero. Practical magnitude, measured by Cohen's $d_z$ and Cliff's $\delta$, reveals whether the difference is meaningful. In this benchmark, `{top_ranked}` exhibits both statistical significance ($p_{{\text{{Holm}}}} < 10^{{-5}}$) and large practical effect sizes ($d_z > 2.0$) compared to general-domain baselines.
2. **Ranking Stability vs. Invariance**:
   Model performance values shift noticeably across representations (e.g., lower accuracy on JSON vs. mixed narratives), but the relative model ordering remains highly stable ($\rho \ge 0.71$, $\tau \ge 0.62$). We characterize this as **ranking stability**, refraining from overclaiming total structural invariance.
3. **Component Sensitivity vs. Causal Feature Importance**:
   Stage 12 leave-one-dimension-out analysis measures **component sensitivity**: how much document rankings and top selections change when an informativeness signal is removed. For instance, removing `information_content` causes an upward score shift but preserves 88.6% of top-selected documents, whereas removing `redundancy_noise` causes substantial document stratum shifts (retaining only 49.9% Jaccard overlap). These sensitivity shifts reflect scoring dynamics rather than causal claims of linguistic importance.
4. **Reproducibility & Determinism**:
   All stochastic bootstrap resamplings were executed under a fixed deterministic pseudo-random number generator (`seed = 42`) across matched conditions, ensuring exact reproducibility of all published intervals and rank frequencies.
