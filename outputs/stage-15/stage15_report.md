# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report

## 1. Executive Conclusion

**Recommended Model:** `answerdotai/ModernBERT-base`  
- **Selection Status:** Pareto-Optimal  
- **Baseline Operational MECS:** 87.40 / 100  
- **Maritime Top-1 Accuracy:** 57.99%  
- **Rare Maritime Token Accuracy:** 77.34%  
- **MLM Loss:** 2.1937  
- **Weighting Sensitivity Stability:** Won 3/4 scenarios  

**Rationale:** answerdotai/ModernBERT-base demonstrated the strongest intrinsic MLM capability (57.99% Top-1 accuracy, 2.1937 MLM loss), high ranking consistency across representations (mean rank 1.2) and subsets (mean rank 1.0), consistent leader across 3/4 MECS sensitivity scenarios, and confirmed non-dominated Pareto status.

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
| `answerdotai/ModernBERT-base` | 57.99 | 73.65 | 77.34 | 2.1937 | 13.72 | 87.4 |
| `roberta-base` | 46.12 | 69.53 | 80.70 | 2.8720 | 16.02 | 87.04 |
| `allenai/scibert_scivocab_uncased` | 32.39 | 48.46 | 52.02 | 4.2804 | 36.58 | 82.19 |
| `bert-base-uncased` | 28.08 | 45.71 | 46.48 | 4.7544 | 25.50 | 78.0 |
| `nlpaueb/legal-bert-base-uncased` | 27.96 | 43.21 | 38.48 | 4.4779 | 42.10 | 77.05 |
| `dmis-lab/biobert-base-cased-v1.2` | 23.55 | 35.01 | 28.06 | 5.1486 | 91.85 | 69.01 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 20.29 | 29.73 | 22.52 | 5.8753 | 91.56 | 58.78 |

### Domain / Tokenizer Fit & Operational Metrics

| Model Name | Frag Rate (%) | OOV Rate (%) | Single Token Cov (%) | Latency (ms) | Throughput (docs/s) | Parameters (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 63.0 | N/A | 37.0 | 28.7 | 34.8 | 149 |
| `roberta-base` | 64.8 | N/A | 35.2 | 23.4 | 42.8 | 125 |
| `allenai/scibert_scivocab_uncased` | 42.1 | 0.00 | 57.9 | 20.4 | 49.0 | 110 |
| `bert-base-uncased` | 26.6 | 0.00 | 73.4 | 19.5 | 51.2 | 110 |
| `nlpaueb/legal-bert-base-uncased` | 37.6 | 0.03 | 62.4 | 20.8 | 48.2 | 110 |
| `dmis-lab/biobert-base-cased-v1.2` | 35.8 | 0.00 | 64.2 | 20.8 | 48.2 | 110 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 42.7 | 0.00 | 57.3 | 20.6 | 48.6 | 110 |

---

## 4. Representation Robustness

- **Kendall's W (Multi-Ranking Concordance):** `0.7943`
- **Mean Pairwise Spearman's Rho:** `0.7429`

Representation breakdown across evaluated formats:

| Model | json | key_value | mixed | narrative | template | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1.0 | 1.0 | 1.0 | 2.0 | 1.0 | 1.2 | 0.45 |
| `roberta-base` | 3.0 | 2.0 | 2.0 | 1.0 | 2.0 | 2.0 | 0.71 |
| `allenai/scibert_scivocab_uncased` | 4.0 | 3.0 | 3.0 | 4.0 | 3.0 | 3.4 | 0.55 |
| `bert-base-uncased` | 5.0 | 4.0 | 7.0 | 3.0 | 4.0 | 4.6 | 1.52 |
| `nlpaueb/legal-bert-base-uncased` | 2.0 | 5.0 | 4.0 | 5.0 | 7.0 | 4.6 | 1.82 |
| `dmis-lab/biobert-base-cased-v1.2` | 6.0 | 6.0 | 5.0 | 6.0 | 5.0 | 5.6 | 0.55 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7.0 | 7.0 | 6.0 | 7.0 | 6.0 | 6.6 | 0.55 |

---

## 5. Subset Robustness

- **Kendall's W (Multi-Ranking Concordance):** `0.9829`
- **Mean Pairwise Spearman's Rho:** `0.9786`

Subset ranking breakdown across knowledge/informativeness conditions:

| Model | balanced_knowledge | high_knowledge | low_knowledge | medium_knowledge | random_baseline | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| `roberta-base` | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 0.0 |
| `allenai/scibert_scivocab_uncased` | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 0.0 |
| `nlpaueb/legal-bert-base-uncased` | 5.0 | 5.0 | 4.0 | 4.0 | 4.0 | 4.4 | 0.55 |
| `bert-base-uncased` | 4.0 | 4.0 | 5.0 | 5.0 | 5.0 | 4.6 | 0.55 |
| `dmis-lab/biobert-base-cased-v1.2` | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 0.0 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 | 0.0 |

---

## 6. MECS Sensitivity Analysis

Testing invariance across four distinct weighting hypotheses:
1. **Baseline / Operational:** Balanced operational mixture.
2. **Performance-Heavy:** Focuses strictly on intrinsic MLM accuracy and loss.
3. **Domain-Heavy:** Strongly weights rare domain terminology and domain accuracy.
4. **Balanced:** Equal weighting across capability, tokenizer fit, and throughput efficiency.

| Model Name | Baseline Score (Rank) | Perf-Heavy Score (Rank) | Domain-Heavy Score (Rank) | Balanced Score (Rank) | Total Wins | Win Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 87.4 (#1) | 100.0 (#1) | 84.2 (#1) | 78.8 (#2) | 3 | 75% |
| `roberta-base` | 87.0 (#2) | 100.0 (#1) | 83.8 (#4) | 79.5 (#1) | 2 | 50% |
| `allenai/scibert_scivocab_uncased` | 82.2 (#3) | 87.7 (#3) | 84.1 (#2) | 76.6 (#3) | 0 | 0% |
| `bert-base-uncased` | 78.0 (#4) | 78.8 (#5) | 84.1 (#2) | 74.2 (#4) | 0 | 0% |
| `nlpaueb/legal-bert-base-uncased` | 77.0 (#5) | 80.6 (#4) | 82.1 (#5) | 73.2 (#5) | 0 | 0% |
| `dmis-lab/biobert-base-cased-v1.2` | 69.0 (#6) | 70.2 (#6) | 78.1 (#6) | 67.7 (#6) | 0 | 0% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 58.8 (#7) | 59.0 (#7) | 68.3 (#7) | 59.5 (#7) | 0 | 0% |

---

## 7. Pareto Dominance Analysis

| Model Name | Pareto Status | Dominates Count | Dominated By Count | Dominating Models |
| :--- | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | **Pareto-Optimal** | 1 | 0 | None |
| `answerdotai/ModernBERT-base` | **Pareto-Optimal** | 0 | 0 | None |
| `bert-base-uncased` | **Pareto-Optimal** | 2 | 0 | None |
| `nlpaueb/legal-bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `roberta-base` | **Pareto-Optimal** | 0 | 0 | None |
| `dmis-lab/biobert-base-cased-v1.2` | **Dominated** | 0 | 1 | bert-base-uncased |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **Dominated** | 0 | 2 | allenai/scibert_scivocab_uncased, bert-base-uncased |

---

## 8. Capability vs. Resource Trade-offs

### Capability vs. Tokenizer/Domain Fit
- `answerdotai/ModernBERT-base` achieves highest overall intrinsic MLM performance, but exhibits higher subword fragmentation than specialized WordPiece architectures.
- Models with lower subword fragmentation (e.g. `bert-base-uncased` at 26.6% fragmentation) offer better morphological token boundaries for specific domain stems, despite lower overall MLM Top-1 accuracy.

### Capability vs. Operational Cost (Environment-Specific)
- `answerdotai/ModernBERT-base` requires 149M parameters with an operational inference latency of ~31.5ms per document (~31.8 docs/sec) on the benchmark GPU harness.
- Lightweight alternatives (e.g., 110M base models like `bert-base-uncased` and `nlpaueb/legal-bert-base-uncased`) provide lower inference latency (~21.0-22.4ms per document, ~44.7-47.6 docs/sec) with smaller disk footprints (~440MB vs ~590MB).
- Latency and throughput depend on benchmark execution hardware and batch configuration and represent operational considerations rather than intrinsic linguistic capabilities.

### Alternative Trade-off Details
- **Alternative `allenai/scibert_scivocab_uncased`:** Offers lower subword fragmentation (42.1% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (20.4ms vs 28.7ms).
- **Alternative `bert-base-uncased`:** Offers lower subword fragmentation (26.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (19.5ms vs 28.7ms).
- **Alternative `nlpaueb/legal-bert-base-uncased`:** Offers lower subword fragmentation (37.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (20.8ms vs 28.7ms).
- **Alternative `roberta-base`:** Offers higher rare maritime token accuracy (80.7% vs 77.3%); smaller footprint (125M vs 149M params); lower latency (23.4ms vs 28.7ms).

---

## 9. Methodological Note

> **Notice on Composite Scoring:**  
> Maritime Encoder Composite Score (MECS) is an operational composite compatibility score used to summarize encoder evaluation characteristics for model selection. It is not intended as a direct measure of language or maritime understanding. The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, rare-token domain generalization, representation consistency, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.
