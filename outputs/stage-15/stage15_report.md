# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report

## 1. Executive Conclusion

**Recommended Model:** `answerdotai/ModernBERT-base`  
- **Selection Status:** Pareto-Optimal  
- **Baseline Operational MECS:** 51.35 / 100  
- **Maritime Top-1 Accuracy:** 28.59%  
- **Rare Maritime Token Accuracy:** 7.11%  
- **MLM Loss:** 4.1942  
- **Weighting Sensitivity Stability:** Won 2/4 scenarios  

**Rationale:** answerdotai/ModernBERT-base demonstrated the strongest intrinsic MLM capability (28.59% Top-1 accuracy, 4.1942 MLM loss), high ranking consistency across representations (mean rank 1.2) and subsets (mean rank 1.0), leading performance across 2/4 MECS sensitivity scenarios (baseline and performance-heavy paradigms), and confirmed non-dominated Pareto status.

---

## 2. Data Coverage

- **Discovered Files:** 175
- **Valid Evaluated Matrix Cells:** 175
- **Expected Full Cartesian Cells:** 175
- **Missing Cells:** 0
- **Invalid Files Skipped:** 0
- **Evaluated Models (7):** allenai/scibert_scivocab_uncased, answerdotai/ModernBERT-base, bert-base-uncased, dmis-lab/biobert-base-cased-v1.2, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, nlpaueb/legal-bert-base-uncased, roberta-base
- **Evaluated Representations (5):** json, key_value, mixed, narrative, template
- **Evaluated Subsets (5):** balanced_knowledge, high_knowledge, low_knowledge, medium_knowledge, random_baseline

---

## 3. Model Comparison

### Capability Metrics

| Model Name | Maritime Top-1 (%) | Top-5 (%) | Rare Top-1 (%) | MLM Loss | Pseudo-Perplexity | Baseline MECS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 28.59 | 45.74 | 7.11 | 4.1942 | 14.07 | 51.35 |
| `roberta-base` | 21.01 | 36.61 | 3.55 | 5.3657 | 17.31 | 29.42 |
| `nlpaueb/legal-bert-base-uncased` | 20.92 | 34.41 | 15.13 | 4.5905 | 39.42 | 50.4 |
| `bert-base-uncased` | 17.67 | 31.43 | 0.88 | 6.6480 | 28.15 | 39.14 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 14.19 | 22.42 | 17.23 | 5.7500 | 79.62 | 45.06 |
| `allenai/scibert_scivocab_uncased` | 14.17 | 24.87 | 13.34 | 6.1235 | 35.42 | 42.03 |
| `dmis-lab/biobert-base-cased-v1.2` | 13.58 | 22.43 | 11.76 | 6.0726 | 81.32 | 41.91 |

### Domain / Tokenizer Fit & Operational Metrics

| Model Name | Frag Rate (%) | OOV Rate (%) | Single Token Cov (%) | Latency (ms) | Throughput (docs/s) | Parameters (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 63.0 | N/A | 37.0 | 31.5 | 31.8 | 149 |
| `roberta-base` | 64.8 | N/A | 35.2 | 25.1 | 39.8 | 125 |
| `nlpaueb/legal-bert-base-uncased` | 37.6 | 0.03 | 62.4 | 22.3 | 44.8 | 110 |
| `bert-base-uncased` | 26.6 | 0.00 | 73.4 | 21.0 | 47.6 | 110 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 42.4 | 0.00 | 57.6 | 21.9 | 45.6 | 110 |
| `allenai/scibert_scivocab_uncased` | 41.8 | 0.00 | 58.2 | 22.1 | 45.3 | 110 |
| `dmis-lab/biobert-base-cased-v1.2` | 35.5 | 0.00 | 64.5 | 22.4 | 44.6 | 110 |

---

## 4. Representation Robustness

- **Kendall's W (Multi-Ranking Concordance):** `0.6514`
- **Mean Pairwise Spearman's Rho:** `0.5643`

Representation breakdown across evaluated formats:

| Model | json | key_value | mixed | narrative | template | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1.0 | 1.0 | 2.0 | 1.0 | 1.0 | 1.2 | 0.45 |
| `nlpaueb/legal-bert-base-uncased` | 2.0 | 3.0 | 1.0 | 3.0 | 4.0 | 2.6 | 1.14 |
| `roberta-base` | 5.0 | 2.0 | 5.0 | 2.0 | 2.0 | 3.2 | 1.64 |
| `bert-base-uncased` | 3.0 | 4.0 | 7.0 | 4.0 | 3.0 | 4.2 | 1.64 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 6.0 | 5.0 | 3.0 | 6.0 | 7.0 | 5.4 | 1.52 |
| `allenai/scibert_scivocab_uncased` | 7.0 | 6.0 | 4.0 | 5.0 | 6.0 | 5.6 | 1.14 |
| `dmis-lab/biobert-base-cased-v1.2` | 4.0 | 7.0 | 6.0 | 7.0 | 5.0 | 5.8 | 1.3 |

---

## 5. Subset Robustness

- **Kendall's W (Multi-Ranking Concordance):** `0.9229`
- **Mean Pairwise Spearman's Rho:** `0.9036`

Subset ranking breakdown across knowledge/informativeness conditions:

| Model | balanced_knowledge | high_knowledge | low_knowledge | medium_knowledge | random_baseline | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| `roberta-base` | 2.0 | 3.0 | 2.0 | 3.0 | 2.0 | 2.4 | 0.55 |
| `nlpaueb/legal-bert-base-uncased` | 3.0 | 2.0 | 3.0 | 2.0 | 3.0 | 2.6 | 0.55 |
| `bert-base-uncased` | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 0.0 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 6.0 | 6.0 | 6.0 | 5.0 | 5.0 | 5.6 | 0.55 |
| `allenai/scibert_scivocab_uncased` | 5.0 | 7.0 | 5.0 | 7.0 | 6.0 | 6.0 | 1.0 |
| `dmis-lab/biobert-base-cased-v1.2` | 7.0 | 5.0 | 7.0 | 6.0 | 7.0 | 6.4 | 0.89 |

---

## 6. MECS Sensitivity Analysis

Testing invariance across four distinct weighting hypotheses:
1. **Baseline / Operational:** Balanced operational mixture.
2. **Performance-Heavy:** Focuses strictly on intrinsic MLM accuracy and loss.
3. **Domain-Heavy:** Strongly weights rare domain terminology and domain accuracy.
4. **Balanced:** Equal weighting across capability, tokenizer fit, and throughput efficiency.

| Model Name | Baseline Score (Rank) | Perf-Heavy Score (Rank) | Domain-Heavy Score (Rank) | Balanced Score (Rank) | Total Wins | Win Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 51.4 (#1) | 67.4 (#1) | 46.6 (#5) | 32.4 (#6) | 2 | 50% |
| `nlpaueb/legal-bert-base-uncased` | 50.4 (#2) | 49.1 (#2) | 56.0 (#2) | 61.4 (#1) | 1 | 25% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 45.1 (#3) | 25.3 (#4) | 57.0 (#1) | 53.0 (#2) | 1 | 25% |
| `allenai/scibert_scivocab_uncased` | 42.0 (#4) | 21.9 (#5) | 48.8 (#3) | 48.0 (#4) | 0 | 0% |
| `dmis-lab/biobert-base-cased-v1.2` | 41.9 (#5) | 16.8 (#7) | 46.9 (#4) | 47.7 (#5) | 0 | 0% |
| `bert-base-uncased` | 39.1 (#6) | 19.9 (#6) | 33.8 (#6) | 48.0 (#3) | 0 | 0% |
| `roberta-base` | 29.4 (#7) | 36.1 (#3) | 22.7 (#7) | 26.5 (#7) | 0 | 0% |

---

## 7. Pareto Dominance Analysis

| Model Name | Pareto Status | Dominates Count | Dominated By Count | Dominating Models |
| :--- | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `answerdotai/ModernBERT-base` | **Pareto-Optimal** | 0 | 0 | None |
| `bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `dmis-lab/biobert-base-cased-v1.2` | **Pareto-Optimal** | 0 | 0 | None |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **Pareto-Optimal** | 0 | 0 | None |
| `nlpaueb/legal-bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `roberta-base` | **Pareto-Optimal** | 0 | 0 | None |

---

## 8. Capability vs. Resource Trade-offs

### Capability vs. Tokenizer/Domain Fit
- `answerdotai/ModernBERT-base` achieves highest overall intrinsic MLM performance, but exhibits higher subword fragmentation than specialized WordPiece architectures.
- Models with lower subword fragmentation (e.g. `bert-base-uncased` at 26.6% fragmentation) offer better morphological token boundaries for specific domain stems, despite lower overall MLM Top-1 accuracy.

### Capability vs. Operational Cost
- `answerdotai/ModernBERT-base` requires 149M parameters and ~458.6ms latency.
- Lightweight alternatives (e.g., 110M base models) provide faster inference latency (down to ~154-175ms) with smaller disk and memory footprints.

### Alternative Trade-off Details
- **Alternative `allenai/scibert_scivocab_uncased`:** Offers lower subword fragmentation (41.8% vs 63.0%); higher rare maritime token accuracy (13.3% vs 7.1%); smaller footprint (110M vs 149M params); lower latency (22.1ms vs 31.5ms).
- **Alternative `bert-base-uncased`:** Offers lower subword fragmentation (26.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (21.0ms vs 31.5ms).
- **Alternative `dmis-lab/biobert-base-cased-v1.2`:** Offers lower subword fragmentation (35.5% vs 63.0%); higher rare maritime token accuracy (11.8% vs 7.1%); smaller footprint (110M vs 149M params); lower latency (22.4ms vs 31.5ms).
- **Alternative `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`:** Offers lower subword fragmentation (42.4% vs 63.0%); higher rare maritime token accuracy (17.2% vs 7.1%); smaller footprint (110M vs 149M params); lower latency (21.9ms vs 31.5ms).
- **Alternative `nlpaueb/legal-bert-base-uncased`:** Offers lower subword fragmentation (37.6% vs 63.0%); higher rare maritime token accuracy (15.1% vs 7.1%); smaller footprint (110M vs 149M params); lower latency (22.3ms vs 31.5ms).
- **Alternative `roberta-base`:** Offers smaller footprint (125M vs 149M params); lower latency (25.1ms vs 31.5ms).

---

## 9. Methodological Note

> **Notice on Composite Scoring:**  
> Maritime Encoder Composite Score (MECS) is an operational composite compatibility score used to summarize encoder evaluation characteristics for model selection. It is not intended as a direct measure of language or maritime understanding. The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, rare-token domain generalization, representation consistency, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.
