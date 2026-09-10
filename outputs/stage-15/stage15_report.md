# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report

## 1. Executive Conclusion

**Recommended Model:** `answerdotai/ModernBERT-base`  
- **Selection Status:** Pareto-Optimal  
- **Baseline Operational MUI:** 78.56 / 100  
- **Maritime Top-1 Accuracy:** 73.05%  
- **Rare Maritime Token Accuracy:** 70.51%  
- **MLM Loss:** 1.4386  
- **Weighting Sensitivity Stability:** Won 4/4 scenarios  

**Rationale:** answerdotai/ModernBERT-base demonstrated the strongest intrinsic MLM capability (73.05% Top-1 accuracy, 1.4386 MLM loss), high ranking consistency across representations (mean rank 7.0) and subsets (mean rank 1.0), unanimous leader across all 4 MUI sensitivity scenarios, and confirmed non-dominated Pareto status.

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

| Model Name | Maritime Top-1 (%) | Top-5 (%) | Rare Top-1 (%) | MLM Loss | Pseudo-Perplexity | Baseline MUI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 73.05 | 85.42 | 70.51 | 1.4386 | 15.45 | 78.56 |
| `roberta-base` | 63.32 | 81.13 | 61.85 | 1.9490 | 17.14 | 65.78 |
| `nlpaueb/legal-bert-base-uncased` | 33.05 | 49.17 | 38.71 | 4.0853 | 40.99 | 35.33 |
| `allenai/scibert_scivocab_uncased` | 32.62 | 49.76 | 51.13 | 4.2064 | 33.27 | 47.63 |
| `bert-base-uncased` | 29.73 | 46.69 | 50.44 | 4.6164 | 24.67 | 49.97 |
| `dmis-lab/biobert-base-cased-v1.2` | 25.78 | 39.64 | 30.82 | 4.7867 | 83.67 | 35.53 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 21.20 | 31.30 | 23.55 | 5.7246 | 81.53 | 23.79 |

### Domain / Tokenizer Fit & Operational Metrics

| Model Name | Frag Rate (%) | OOV Rate (%) | Single Token Cov (%) | Latency (ms) | Throughput (docs/s) | Parameters (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 63.0 | N/A | 37.0 | 35.0 | 28.5 | 149 |
| `roberta-base` | 64.8 | N/A | 35.2 | 27.1 | 36.9 | 125 |
| `nlpaueb/legal-bert-base-uncased` | 37.6 | 0.03 | 62.4 | 24.4 | 40.9 | 110 |
| `allenai/scibert_scivocab_uncased` | 41.8 | 0.00 | 58.2 | 23.8 | 42.0 | 110 |
| `bert-base-uncased` | 26.6 | 0.00 | 73.4 | 21.4 | 46.7 | 110 |
| `dmis-lab/biobert-base-cased-v1.2` | 35.5 | 0.00 | 64.5 | 24.7 | 40.4 | 110 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 42.4 | 0.00 | 57.6 | 23.5 | 42.5 | 110 |

---

## 4. Representation Robustness

- **Mean Pairwise Kendall's Tau (Ranking Agreement):** `0.7333`
- **Mean Pairwise Spearman's Rho:** `0.8500`

Representation breakdown across evaluated formats:

| Model | json | key_value | mixed | narrative | template | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 2 | 1.2 | 0.45 |
| `roberta-base` | 3 | 2 | 2 | 2 | 1 | 2.0 | 0.71 |
| `nlpaueb/legal-bert-base-uncased` | 2 | 4 | 3 | 4 | 5 | 3.6 | 1.14 |
| `allenai/scibert_scivocab_uncased` | 4 | 3 | 4 | 5 | 3 | 3.8 | 0.84 |
| `bert-base-uncased` | 5 | 5 | 6 | 3 | 4 | 4.6 | 1.14 |
| `dmis-lab/biobert-base-cased-v1.2` | 6 | 6 | 5 | 6 | 6 | 5.8 | 0.45 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7 | 7 | 7 | 7 | 7 | 7.0 | 0.0 |

---

## 5. Subset Robustness

- **Mean Pairwise Kendall's Tau (Ranking Agreement):** `0.9048`
- **Mean Pairwise Spearman's Rho:** `0.9571`

Subset ranking breakdown across knowledge/informativeness conditions:

| Model | balanced_knowledge | high_knowledge | low_knowledge | medium_knowledge | random_baseline | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 1 | 1.0 | 0.0 |
| `roberta-base` | 2 | 2 | 2 | 2 | 2 | 2.0 | 0.0 |
| `allenai/scibert_scivocab_uncased` | 3 | 3 | 3 | 4 | 4 | 3.4 | 0.55 |
| `nlpaueb/legal-bert-base-uncased` | 5 | 4 | 4 | 3 | 3 | 3.8 | 0.84 |
| `bert-base-uncased` | 4 | 5 | 5 | 5 | 5 | 4.8 | 0.45 |
| `dmis-lab/biobert-base-cased-v1.2` | 6 | 6 | 6 | 6 | 6 | 6.0 | 0.0 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7 | 7 | 7 | 7 | 7 | 7.0 | 0.0 |

---

## 6. MUI Sensitivity Analysis

Testing invariance across four distinct weighting hypotheses:
1. **Baseline / Operational:** Balanced operational mixture.
2. **Performance-Heavy:** Focuses strictly on intrinsic MLM accuracy and loss.
3. **Domain-Heavy:** Strongly weights rare domain terminology and domain accuracy.
4. **Balanced:** Equal weighting across capability, tokenizer fit, and throughput efficiency.

| Model Name | Baseline Score (Rank) | Perf-Heavy Score (Rank) | Domain-Heavy Score (Rank) | Balanced Score (Rank) | Total Wins | Win Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 78.6 (#1) | 100.0 (#1) | 84.1 (#1) | 60.9 (#1) | 4 | 100% |
| `roberta-base` | 65.8 (#2) | 84.6 (#2) | 69.0 (#2) | 59.4 (#3) | 0 | 0% |
| `bert-base-uncased` | 50.0 (#3) | 28.8 (#5) | 53.0 (#3) | 59.9 (#2) | 0 | 0% |
| `allenai/scibert_scivocab_uncased` | 47.6 (#4) | 34.5 (#3) | 50.4 (#4) | 50.1 (#4) | 0 | 0% |
| `dmis-lab/biobert-base-cased-v1.2` | 35.5 (#5) | 14.4 (#6) | 32.4 (#6) | 37.7 (#6) | 0 | 0% |
| `nlpaueb/legal-bert-base-uncased` | 35.3 (#6) | 30.1 (#4) | 33.4 (#5) | 46.6 (#5) | 0 | 0% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 23.8 (#7) | 0.0 (#7) | 18.8 (#7) | 27.1 (#7) | 0 | 0% |

---

## 7. Pareto Dominance Analysis

| Model Name | Pareto Status | Dominates Count | Dominated By Count | Dominating Models |
| :--- | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `answerdotai/ModernBERT-base` | **Pareto-Optimal** | 0 | 0 | None |
| `bert-base-uncased` | **Pareto-Optimal** | 2 | 0 | None |
| `nlpaueb/legal-bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `roberta-base` | **Pareto-Optimal** | 0 | 0 | None |
| `dmis-lab/biobert-base-cased-v1.2` | **Dominated** | 0 | 1 | bert-base-uncased |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **Dominated** | 0 | 1 | bert-base-uncased |

---

## 8. Capability vs. Resource Trade-offs

### Capability vs. Tokenizer/Domain Fit
- `answerdotai/ModernBERT-base` achieves highest overall intrinsic MLM performance, but exhibits higher subword fragmentation than specialized WordPiece architectures.
- Models with lower subword fragmentation (e.g. `bert-base-uncased` at 26.6% fragmentation) offer better morphological token boundaries for specific domain stems, despite lower overall MLM Top-1 accuracy.

### Capability vs. Operational Cost
- `answerdotai/ModernBERT-base` requires 149M parameters and ~458.6ms latency.
- Lightweight alternatives (e.g., 110M base models) provide faster inference latency (down to ~154-175ms) with smaller disk and memory footprints.

### Alternative Trade-off Details
- **Alternative `allenai/scibert_scivocab_uncased`:** Offers lower subword fragmentation (41.8% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (23.8ms vs 35.0ms).
- **Alternative `bert-base-uncased`:** Offers lower subword fragmentation (26.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (21.4ms vs 35.0ms).
- **Alternative `nlpaueb/legal-bert-base-uncased`:** Offers lower subword fragmentation (37.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (24.4ms vs 35.0ms).
- **Alternative `roberta-base`:** Offers smaller footprint (125M vs 149M params); lower latency (27.1ms vs 35.0ms).

---

## 9. Methodological Note

> **Notice on Composite Scoring:**  
> The Maritime Understanding Index (MUI) is an operational composite compatibility score rather than a human-validated measure of innate maritime understanding. The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, rare-token domain generalization, representation consistency, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.
