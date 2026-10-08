# Report & Metric Consistency Audit across Pipeline Stages
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**  
**Date**: October 8, 2026  
**Scope**: Systematic cross-check of numerical consistency, rankings, statistical symbols, units, formulas, and definitions across Stage 14, Stage 15, Stage 16, Stage 17, and downstream reports.

---

## 1. Executive Summary

A comprehensive automated consistency audit was conducted across all generated artifacts:
- **Zero Unexplained Contradictions**: All JSON, CSV, and Markdown summaries match across stages.
- **Rankings Alignment**:
  - Stage 14 raw averages, Stage 15 leaderboard, Stage 16 rank stability, and Stage 17 decision engine all agree on the ranking of the 7 models:
    $$\text{ModernBERT} > \text{RoBERTa} > \text{SciBERT} > \text{BERT-base} > \text{Legal-BERT} > \text{BioBERT} > \text{PubMedBERT}$$
- **Statistical Invariants Upheld**:
  - Friedman test degrees of freedom: strictly $df = 6$ ($k = 7$ models).
  - ANOVA F-statistics match authoritative CSV: Model $F = 804.16$, Representation $F = 552.03$, Subset $F = 15.47$, Interaction $F = 55.40$.
  - Paired comparisons: exactly 25 matched conditions ($5 \times 5$).
  - Win counts: ModernBERT vs RoBERTa is strictly 20 wins / 5 losses ($20/25 = 80.0\%$), never misrepresented as 100%.
  - Unpaired Cliff's $\delta$ across 625 comparisons is reported separately from matched-pair win rates.
- **MECS Formula Alignment**:
  - Authoritative baseline configuration: $\text{Top1} = 0.40, \text{Rare Top1} = 0.15, \text{Loss} = 0.25, \text{Frag} = 0.20$ (no Top5 in baseline).
  - Loss transform clipping: Fixed $[3.0, 8.0]$ bounded linear transform correctly clips ModernBERT (loss 2.1937) to $1.0$, producing baseline score = **87.40**.
- **PPPL Reconciliation**:
  - Primary Intrinsic MLM Perplexity: $\exp(\bar{L}_{\text{MLM}}) = 8.97$ (cell-level macro-average = 9.32).
  - Secondary Sampled PLL Pseudo-Perplexity: $13.72$ from `pll_results.json`.

---

## 2. Multi-Stage Metric Alignment Table

| Metric | Stage 14 Raw Aggregator | Stage 15 Leaderboard (`leaderboard.csv`) | Stage 16 Statistics (`stage16_pairwise_tests.csv`) | Stage 17 Decision (`stage17_model_selection.csv`) | Consistency Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ModernBERT Maritime Top-1** | 0.5799 (57.99%) | 57.99% | 0.5799 | 57.99% | EXACT MATCH |
| **ModernBERT Maritime Loss** | 2.1937 | 2.1937 | 2.1937 | 2.1937 | EXACT MATCH |
| **ModernBERT Overall Top-1** | 0.6742 (67.42%) | 67.42% | 0.6742 | 67.42% | EXACT MATCH |
| **RoBERTa Maritime Top-1** | 0.4612 (46.12%) | 46.12% | 0.4612 | 46.12% | EXACT MATCH |
| **SciBERT Maritime Top-1** | 0.3239 (32.39%) | 32.39% | 0.3239 | 32.39% | EXACT MATCH |
| **BERT-base Maritime Top-1** | 0.2808 (28.08%) | 28.08% | 0.2808 | 28.08% | EXACT MATCH |
| **Legal-BERT Maritime Top-1** | 0.2796 (27.96%) | 27.96% | 0.2796 | 27.96% | EXACT MATCH |
| **BioBERT Maritime Top-1** | 0.2355 (23.55%) | 23.55% | 0.2355 | 23.55% | EXACT MATCH |
| **PubMedBERT Maritime Top-1** | 0.2029 (20.29%) | 20.29% | 0.2029 | 20.29% | EXACT MATCH |
| **ModernBERT vs RoBERTa Wins** | 20 / 25 | N/A | 20 Wins, 5 Losses (80.0%) | Cited as 20/25 (80.0%) | EXACT MATCH |
| **Baseline MECS Score** | 87.40 | 87.40 | N/A | 87.40 | EXACT MATCH |
| **Recommended Model** | ModernBERT | ModernBERT | ModernBERT | ModernBERT | UNANIMOUS |

---

## 3. MECS Formula & Normalization Audit

### Authoritative Definition (`stage15_mecs_config.json`)
The baseline configuration is:
$$ \text{MECS}_{\text{baseline}} = 100 \times \left[ 0.40 \cdot S_{\text{top1}} + 0.15 \cdot S_{\text{rare}} + 0.25 \cdot S_{\text{loss}} + 0.20 \cdot S_{\text{frag}} \right] $$

Where fixed transforms are defined as:
* $S_{\text{top1}} = \text{clip}\left(\frac{\text{Top1}}{0.35}, 0.0, 1.0\right)$ (Ceiling = 35.0%)
* $S_{\text{rare}} = \text{clip}\left(\frac{\text{Rare Top1}}{0.25}, 0.0, 1.0\right)$ (Ceiling = 25.0%)
* $S_{\text{loss}} = \text{clip}\left(\frac{8.0 - \text{Loss}}{8.0 - 3.0}, 0.0, 1.0\right)$ (Fixed interval $[3.0, 8.0]$)
* $S_{\text{frag}} = \text{clip}\left(1.0 - \frac{\text{Fragmentation\%}}{100.0}, 0.0, 1.0\right)$

### ModernBERT Calculation Step-by-Step
Using authoritative model-level macro-averages over all 25 conditions (`stage15_model_profiles.csv` / `leaderboard.csv`):
1. $\text{Top1} = 0.5799 \text{ (57.99%)} \implies S_{\text{top1}} = \min(1.0, 0.579872 / 0.35) = \mathbf{1.0}$ *(Note: single-cell row 0 in comparison.csv exhibits 0.5814, but model-level macro-average is strictly 0.5799)*
2. $\text{Rare Top1} = 0.7734 \text{ (77.34%)} \implies S_{\text{rare}} = \min(1.0, 0.773416 / 0.25) = \mathbf{1.0}$
3. $\text{Loss} = 2.1937 \implies S_{\text{loss}} = \text{clip}\left(\frac{8.0 - 2.193664}{5.0}, 0.0, 1.0\right) = \text{clip}(1.1613, 0.0, 1.0) = \mathbf{1.0}$
4. $\text{Frag\%} = 62.99\% \implies S_{\text{frag}} = 1.0 - 0.6299 = \mathbf{0.3701}$

$$\text{MECS} = 100 \times [0.40(1.0) + 0.15(1.0) + 0.25(1.0) + 0.20(0.3701)] = 100 \times [0.40 + 0.15 + 0.25 + 0.07402] = \mathbf{87.40}$$

---

## 4. PPPL Dual-Lineage Reconciliation

| Model Name | Primary MLM Loss ($\bar{L}_{\text{MLM}}$) | Primary MLM Perplexity ($\exp(\bar{L}_{\text{MLM}})$) | Cell Macro-Mean PPPL ($\frac{1}{25}\sum \exp(L_i)$) | Auxiliary Sampled PLL Pseudo-Perplexity (`pll_results.json`) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ModernBERT** | 2.1937 | **8.97** | 9.32 | **13.72** | RECONCILED |
| **RoBERTa** | 2.8720 | **17.67** | 18.24 | **16.02** | RECONCILED |
| **SciBERT** | 4.2804 | **72.27** | 74.15 | **36.58** | RECONCILED |
| **BERT-base** | 4.7544 | **116.09** | 119.82 | **25.50** | RECONCILED |
| **Legal-BERT** | 4.4779 | **88.05** | 90.41 | **42.10** | RECONCILED |
| **BioBERT** | 5.1486 | **172.19** | 178.60 | **91.85** | RECONCILED |
| **PubMedBERT** | 5.8753 | **356.14** | 368.52 | **91.56** | RECONCILED |

* **Methodological Distinction**:
  - `PPL_MLM`: Intrinsic sequence-level masked token perplexity derived from the primary 175-cell benchmark cross-entropy loss.
  - `PPPL_PLL`: Diagnostic pseudo-perplexity evaluated on a sampled document set using token-by-token pseudo-log-likelihood.

---

## 5. Statistical ANOVA & Pairwise Test Audit

### Factorial Crossed Repeated-Measures ANOVA (`stage16_crossed_anova.csv`)
- **Model Effect**: $F(6, 96) = \mathbf{804.1615}$, $p = 1.11 \times 10^{-16}$, Variance Contribution = **55.66%**, $\eta_p^2 = 0.9805$.
- **Representation Effect**: $F(4, 96) = \mathbf{552.0302}$, $p = 1.11 \times 10^{-16}$, Variance Contribution = **25.47%**, $\eta_p^2 = 0.9583$.
- **Subset Effect**: $F(4, 96) = \mathbf{15.4679}$, $p = 8.46 \times 10^{-10}$, Variance Contribution = **0.71%**, $\eta_p^2 = 0.3919$.
- **Model $\times$ Representation**: $F(24, 96) = \mathbf{55.3965}$, $p = 1.11 \times 10^{-16}$, Variance Contribution = **15.34%**, $\eta_p^2 = 0.9327$.
- **Model $\times$ Subset**: $F(24, 96) = \mathbf{2.1247}$, $p = 0.00533$, Variance Contribution = **0.59%**, $\eta_p^2 = 0.3469$.
- **Representation $\times$ Subset**: $F(16, 96) = \mathbf{6.0638}$, $p = 5.25 \times 10^{-9}$, Variance Contribution = **1.12%**, $\eta_p^2 = 0.5026$.
- **Residual Variance**: **1.11%**.

### Friedman Omnibus Reference Test
- Statistic: $\chi^2 = \mathbf{121.1657}$
- Degrees of Freedom: $df = \mathbf{6}$ ($7 \text{ models} - 1$)
- P-value: $p = 9.27 \times 10^{-24}$
- Kendall's $W$: $W = 0.8078$

### Pairwise Dominance Separation
- **ModernBERT vs RoBERTa**:
  - Matched conditions: exactly 25.
  - Paired wins: **20**, losses: **5**, ties: **0** $\implies$ **80.0% win rate**.
  - Wilcoxon statistic: $W = 33.0$, $p_{\text{raw}} = 0.000109$, $p_{\text{holm}} = \mathbf{0.000229}$.
  - Paired rank-biserial effect size: $r_{\text{prb}} = \mathbf{0.8585}$.
  - Unpaired Cliff's delta (across 625 cell pairs): $\delta = \mathbf{0.5104}$.

---

## 6. Discrepancy Scrutiny Checklist
1. **No Stale Terminology**: `MUI` replaced everywhere with `MECS` (Maritime Encoder Composite Score).
2. **No Conflated Statistics**: Paired cell dominance (25 matched conditions) and unpaired Cliff's delta (625 pairs) are strictly segregated.
3. **No Fabricated Performance**: Zero-count slices remain `None` / `null` without zero-fill distortion.
4. **No Circular Normalization**: Loss bounds $[3.0, 8.0]$ are cohort-independent reference constants.
5. **No False Win Rates**: ModernBERT vs RoBERTa win rate is documented as 20/25 (80.0%), never 100%.

**Auditor Sign-Off**: **ZERO UNEXPLAINED CONTRADICTIONS**.
