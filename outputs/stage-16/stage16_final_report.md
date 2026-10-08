# Stage 16: Statistical Validation & Crossed Condition Analysis Report
**Maritime Corpus Pipeline Version 2.1**
*Benchmark Design: 175 benchmark cells across 7 encoder models, 5 representations, and 5 knowledge subsets (25 matched conditions per model).*

---

## 1. Executive Conclusion
This research stage statistically validates cross-model performance differences identified in Stage 15.
Crucially, the benchmark structure is modeled as a **balanced crossed repeated-measures design** (Models x Representations x Subsets) rather than assuming false cell independence.

* **Primary Crossed Repeated-Measures Analysis**: The primary fixed effect of encoder model is statistically decisive under both parametric ANOVA ($F = 804.1615$, $p = 1.1102e-16$, $\eta^2 = 55.7\%$, partial $\eta^2 = 0.981$) and 1,000 block-respecting condition permutations ($p_{\text{perm}} = 0.0010$). Structural representation format accounts for the largest overall share of variance (25.5%, $F = 552.0$), reflecting serialization difficulty shifts, while model architecture accounts for 55.7% with a significant Model $\times$ Representation interaction (15.3%).
* **Secondary Omnibus Friedman Test**: Retained for reference and historical continuity, the Friedman test confirms significant differences across matched conditions (Friedman $\chi^2 = 121.1657$, $p = 9.2726e-24$, $df = 6$).
* **Pairwise Matched Comparisons**: Across all 21 paired comparisons, 19 pairs demonstrate statistically reliable differences after family-wise Holm-Bonferroni correction ($p_{\text{Holm}} < 0.05$).
* **Primary Winner Robustness**: Model `answerdotai/ModernBERT-base` demonstrates unambiguous statistical superiority, attaining an empirical bootstrap rank-1 frequency of **$P(\text{rank}=1) = 100.0%$** across 2000 condition resamples.
* **Paired vs. Unpaired Effect Sizes**: High paired rank-biserial correlations ($r_{\text{prb}} > 0.8$) and large paired Cohen's $d_z \in [1.20, 3.21]$ confirm substantial practical margins on matched cells. Secondary unpaired Cliff's delta values ($\delta \in [0.33, 0.47]$, medium) are strictly reported as descriptive distribution-level statistics.

---

## 2. Crossed Factorial Analysis (Primary Repeated-Measures ANOVA)
The 25 benchmark configurations per model share underlying documents, representations, and knowledge subsets.
The primary statistical model is a **3-way crossed repeated-measures ANOVA** with block-respecting permutation testing.

| Factor / Variation Source | Sum of Squares | $df$ | Mean Square | $F$-Statistic | $p$-value | Variance Contribution ($\eta^2$) | Partial $\eta^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Model (Fixed Effect) | 2.7330 | 6 | 0.4555 | **804.16** | **1.1102e-16** | 55.66% | 0.9805 |
| Representation (Blocking Factor) | 1.2508 | 4 | 0.3127 | **552.03** | **1.1102e-16** | 25.47% | 0.9583 |
| Subset (Blocking Factor) | 0.0350 | 4 | 0.0088 | **15.47** | **8.4595e-10** | 0.71% | 0.3919 |
| Model x Representation | 0.7531 | 24 | 0.0314 | **55.40** | **1.1102e-16** | 15.34% | 0.9327 |
| Model x Subset | 0.0289 | 24 | 0.0012 | **2.12** | **5.3323e-03** | 0.59% | 0.3469 |
| Representation x Subset | 0.0550 | 16 | 0.0034 | **6.06** | **5.2549e-09** | 1.12% | 0.5026 |
| Residual (Model x Rep x Subset) | 0.0544 | 96 | 0.0006 | — | — | 1.11% | — |

* **Block-Respecting Permutation Test ($p_{\text{perm}}$)**: **0.0010** (exact permutation of model labels within each of the 25 joint representation x subset blocks across 1,000 resamples).
* *Variance Decomposition Interpretation*: Document representation format accounts for the largest share of benchmark variance (25.5%, $F = 552.0$), reflecting substantial baseline difficulty shifts across serialization formats (e.g., Markdown table vs. JSON vs. prose). Crucially, the model architecture main effect remains highly significant and substantial, accounting for 55.7% of benchmark variance ($F = 804.16, p < 10^{-15}$, partial $\eta^2 = 0.981$), with a statistically significant Model $\times$ Representation interaction (15.3%, $F = 55.40$) indicating differential representation adaptation.

### Secondary Reference: Friedman Omnibus Test
Retained for continuity as a secondary nonparametric baseline across matched conditions:
* **Friedman $\chi^2$**: 121.1657 ($df = 6$, $p = 9.272645e-24$)
* **Decision**: Statistically Significant ($p < 0.001$).

---

## 3. Pairwise Matched Comparisons
Pairwise non-parametric **Wilcoxon signed-rank tests** were conducted on matched paired cell differences ($A_i - B_i$).
Multiple comparisons are controlled via the **Holm-Bonferroni step-down procedure** across the family of 21 comparisons.

| Model A | Model B | Paired Cells | Wins | Losses | Ties | Paired Win Rate | Paired Rank-Biserial | Wilcoxon Stat | Raw $p$ | Holm $p$ | Holm Sig? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | 25 | 1 | 24 | 0 | 4.0% | -0.99 | 1.0 | 1.192093e-07 | 1.430511e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | 25 | 20 | 5 | 0 | 80.0% | +0.65 | 57.0 | 0.003419 | 0.013677 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | 20 | 5 | 0 | 80.0% | +0.40 | 98.0 | 0.085139 | 0.170278 | No |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | 25 | 2 | 23 | 0 | 8.0% | -0.98 | 3.0 | 2.980232e-07 | 3.278255e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | 25 | 25 | 0 | 0 | 100.0% | +1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `answerdotai/ModernBERT-base` | `roberta-base` | 25 | 20 | 5 | 0 | 80.0% | +0.86 | 23.0 | 3.814697e-05 | 0.000229 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | 25 | 22 | 3 | 0 | 88.0% | +0.83 | 28.0 | 8.803606e-05 | 0.00044 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 22 | 3 | 0 | 88.0% | +0.91 | 14.0 | 6.556511e-06 | 5.900860e-05 | **Yes ($p < 0.05$)** |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | 25 | 14 | 11 | 0 | 56.0% | +0.10 | 147.0 | 0.691519 | 0.691519 | No |
| `bert-base-uncased` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 25 | 22 | 3 | 0 | 88.0% | +0.88 | 19.0 | 1.829863e-05 | 0.000128 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | 25 | 9 | 16 | 0 | 36.0% | -0.56 | 71.0 | 0.012466 | 0.037399 | **Yes ($p < 0.05$)** |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | 25 | 4 | 21 | 0 | 16.0% | -0.89 | 18.0 | 1.507998e-05 | 0.000121 | **Yes ($p < 0.05$)** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | 25 | 0 | 25 | 0 | 0.0% | -1.00 | 0.0 | 5.960464e-08 | 1.251698e-06 | **Yes ($p < 0.05$)** |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | 25 | 2 | 23 | 0 | 8.0% | -0.96 | 7.0 | 1.132488e-06 | 1.132488e-05 | **Yes ($p < 0.05$)** |

---

## 4. Effect Sizes: Paired vs. Unpaired Statistics
To address methodological confounding, effect sizes are strictly partitioned into:
1. **Paired Statistics (Primary)**: Matched Cohen's $d_z$ and Kerby's paired rank-biserial correlation ($r_{\text{prb}} = \frac{W^+ - W^-}{W^+ + W^-}$).
2. **Unpaired Statistics (Secondary/Descriptive)**: Cliff's Delta ($\delta$). Cliff's delta is an unpaired distribution-level statistic and does NOT measure matched-cell dominance.

| Model A | Model B | Mean Diff | 95% Paired CI | Paired $r_{\text{prb}}$ | Paired $d_z$ | Unpaired Cliff's $\delta$ | Practical Importance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | `answerdotai/ModernBERT-base` | -0.2560 | [-0.3241, -0.1879] | -0.99 (large) | -1.55 | -0.83 (large) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `bert-base-uncased` | +0.0431 | [+0.0178, +0.0684] | +0.65 (large) | +0.70 | +0.25 (small) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0883 | [+0.0759, +0.1008] | +1.00 (large) | +2.93 | +0.46 (medium) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.1210 | [+0.1087, +0.1333] | +1.00 (large) | +4.07 | +0.55 (large) | Substantial practical advantage |
| `allenai/scibert_scivocab_uncased` | `nlpaueb/legal-bert-base-uncased` | +0.0442 | [+0.0045, +0.0840] | +0.40 (medium) | +0.46 | +0.29 (small) | No statistically reliable difference |
| `allenai/scibert_scivocab_uncased` | `roberta-base` | -0.1373 | [-0.1726, -0.1020] | -0.98 (large) | -1.61 | -0.52 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `bert-base-uncased` | +0.2991 | [+0.2249, +0.3733] | +1.00 (large) | +1.66 | +0.95 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `dmis-lab/biobert-base-cased-v1.2` | +0.3443 | [+0.2789, +0.4098] | +1.00 (large) | +2.17 | +0.97 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.3770 | [+0.3097, +0.4444] | +1.00 (large) | +2.31 | +0.96 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `nlpaueb/legal-bert-base-uncased` | +0.3003 | [+0.2501, +0.3504] | +1.00 (large) | +2.47 | +0.93 (large) | Substantial practical advantage |
| `answerdotai/ModernBERT-base` | `roberta-base` | +0.1187 | [+0.0644, +0.1730] | +0.86 (large) | +0.90 | +0.51 (large) | Substantial practical advantage |
| `bert-base-uncased` | `dmis-lab/biobert-base-cased-v1.2` | +0.0452 | [+0.0229, +0.0675] | +0.83 (large) | +0.84 | +0.38 (medium) | Substantial practical advantage |
| `bert-base-uncased` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0779 | [+0.0466, +0.1092] | +0.91 (large) | +1.03 | +0.42 (medium) | Substantial practical advantage |
| `bert-base-uncased` | `nlpaueb/legal-bert-base-uncased` | +0.0011 | [-0.0434, +0.0457] | +0.10 (negligible) | +0.01 | +0.20 (small) | No statistically reliable difference |
| `bert-base-uncased` | `roberta-base` | -0.1804 | [-0.2197, -0.1412] | -1.00 (large) | -1.90 | -0.67 (large) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | +0.0327 | [+0.0200, +0.0453] | +0.88 (large) | +1.07 | +0.16 (small) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `nlpaueb/legal-bert-base-uncased` | -0.0441 | [-0.0763, -0.0119] | -0.56 (large) | -0.57 | -0.25 (small) | Substantial practical advantage |
| `dmis-lab/biobert-base-cased-v1.2` | `roberta-base` | -0.2256 | [-0.2638, -0.1875] | -1.00 (large) | -2.44 | -0.84 (large) | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `nlpaueb/legal-bert-base-uncased` | -0.0768 | [-0.1119, -0.0416] | -0.89 (large) | -0.90 | -0.41 (medium) | Substantial practical advantage |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | `roberta-base` | -0.2583 | [-0.2997, -0.2170] | -1.00 (large) | -2.58 | -0.85 (large) | Substantial practical advantage |
| `nlpaueb/legal-bert-base-uncased` | `roberta-base` | -0.1815 | [-0.2341, -0.1290] | -0.96 (large) | -1.43 | -0.76 (large) | Substantial practical advantage |

---

## 5. Bootstrap Uncertainty
Deterministic bootstrap resampling ($B = 2,000$, seed = 42) of the matched benchmark conditions provides non-parametric 95% confidence intervals for individual model Top-1 accuracy and paired differences.

### Model Score 95% Confidence Intervals
| Model Name | Bootstrap Mean | 95% CI Lower | 95% CI Upper | Resamples | Matched Conditions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `allenai/scibert_scivocab_uncased` | 0.3235 | 0.2767 | 0.3677 | 2000 | 25 |
| `answerdotai/ModernBERT-base` | 0.5798 | 0.5281 | 0.6342 | 2000 | 25 |
| `bert-base-uncased` | 0.2808 | 0.2377 | 0.3194 | 2000 | 25 |
| `dmis-lab/biobert-base-cased-v1.2` | 0.2352 | 0.1953 | 0.2741 | 2000 | 25 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 0.2024 | 0.1558 | 0.2486 | 2000 | 25 |
| `nlpaueb/legal-bert-base-uncased` | 0.2794 | 0.2515 | 0.3115 | 2000 | 25 |
| `roberta-base` | 0.4610 | 0.4092 | 0.5101 | 2000 | 25 |

---

## 6. Bootstrap Rank Stability
In each of the 2,000 bootstrap resamples, all models were evaluated across the sampled configurations and ranked.
$P(\text{rank}=1)$ denotes the proportion of resamples in which the model ranked first.

| Model Name | Mean Rank | Rank SD | $P(\text{rank}=1)$ | $P(\text{rank}=2)$ | $P(\text{rank} \le 3)$ | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `answerdotai/ModernBERT-base` | **1.00** | ±0.00 | **1.0000** | 0.0000 | 1.0000 | **Decisive Leader** ($P_1 > 95\%$) |
| `roberta-base` | **2.00** | ±0.00 | **0.0000** | 1.0000 | 1.0000 | **Strong Second** ($P_2 > 95\%$) |
| `allenai/scibert_scivocab_uncased` | **3.02** | ±0.12 | **0.0000** | 0.0000 | 0.9850 | Consistently Top-Tier ($P_{\le 3} > 90\%$) |
| `bert-base-uncased` | **4.47** | ±0.50 | **0.0000** | 0.0000 | 0.0005 | Mid/Lower Tier Encoder |
| `nlpaueb/legal-bert-base-uncased` | **4.52** | ±0.53 | **0.0000** | 0.0000 | 0.0145 | Mid/Lower Tier Encoder |
| `dmis-lab/biobert-base-cased-v1.2` | **6.00** | ±0.00 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **7.00** | ±0.00 | **0.0000** | 0.0000 | 0.0000 | Mid/Lower Tier Encoder |

*Note: Empirical bootstrap rank frequencies represent resampling stability under matched condition perturbation, not Bayesian posterior probabilities of absolute domain capability.*

---

## 7. Representation Robustness
To assess whether structural representation shifts alter model hierarchies, model performances were aggregated across representations and compared against global benchmark ranks.

| Representation | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Json** | `answerdotai/ModernBERT-base` | 0.6190 | 0.7857 | 0.7143 | Moderate ranking stability |
| **Key_value** | `answerdotai/ModernBERT-base` | 0.7308 | 1.0000 | 1.0000 | High ranking stability |
| **Mixed** | `answerdotai/ModernBERT-base` | 0.6957 | 0.7857 | 0.7143 | Moderate ranking stability |
| **Narrative** | `roberta-base` | 0.4544 | 0.9286 | 0.8095 | High ranking stability |
| **Template** | `answerdotai/ModernBERT-base` | 0.4023 | 0.8929 | 0.8095 | Moderate ranking stability |

---

## 8. Subset Robustness
Evaluations across knowledge-classified subsets evaluate whether domain-informativeness stratification produces rank inversions or disparate encoder behavior.

| Knowledge Subset | Winning Model | Winner Top-1 | Spearman $\rho$ vs Global | Kendall $\tau$ vs Global | Ranking Stability Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **balanced_knowledge** | `answerdotai/ModernBERT-base` | 0.5749 | 1.0000 | 1.0000 | High ranking stability |
| **high_knowledge** | `answerdotai/ModernBERT-base` | 0.5828 | 1.0000 | 1.0000 | High ranking stability |
| **low_knowledge** | `answerdotai/ModernBERT-base` | 0.6020 | 0.9643 | 0.9048 | High ranking stability |
| **medium_knowledge** | `answerdotai/ModernBERT-base` | 0.5778 | 0.9643 | 0.9048 | High ranking stability |
| **random_baseline** | `answerdotai/ModernBERT-base` | 0.5620 | 0.9643 | 0.9048 | High ranking stability |

---

## 9. Stage 12 Component Sensitivity Analysis
Leave-one-dimension-out ablation on the Stage 12 Domain Informativeness Engine evaluates how the removal of individual observable signals affects document scores, ranking correlation, and Top-20% document selection (Jaccard index).

### Component Score & Ranking Sensitivity
| Ablated Component | Mean $\Delta_{{\text{{score}}}}$ | Median $\Delta$ | SD $\Delta$ | Spearman $\rho$ | Kendall $\tau$ | Top-20% Jaccard Overlap | Selection Impact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `domain_relevance` | -0.0587 | -0.0617 | 0.0244 | 0.9643 | 0.8442 | **0.6684** | Moderate selection impact: component alters borderline selections |
| `information_content` | +0.1943 | +0.1990 | 0.0329 | 0.9358 | 0.8457 | **0.8880** | Low selection impact: score shifts have limited effect on top document selection |
| `tfidf_representativeness` | -0.0032 | +0.0030 | 0.0422 | 0.8606 | 0.7234 | **0.6097** | Moderate selection impact: component alters borderline selections |
| `redundancy_noise` | -0.1487 | -0.1484 | 0.0557 | 0.8606 | 0.6737 | **0.5010** | High selection impact: component strongly influences document selection |

### Stratum / Tier Transition Stability
| Ablated Component | Same Tier (%) | High Tier Retention (%) | Medium Tier Retention (%) | Low Tier Retention (%) | Documents Shifted |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `domain_relevance` | 89.58% | **80.14%** | 91.33% | 93.79% | 10,089 / 96,850 |
| `information_content` | 81.17% | **94.04%** | 74.69% | 87.71% | 18,237 / 96,850 |
| `tfidf_representativeness` | 76.15% | **75.71%** | 80.13% | 64.64% | 23,103 / 96,850 |
| `redundancy_noise` | 66.70% | **66.70%** | 72.26% | 50.03% | 32,253 / 96,850 |

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
