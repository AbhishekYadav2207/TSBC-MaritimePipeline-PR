# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report

## 1. Executive Conclusion

**Recommended Model:** `answerdotai/ModernBERT-base`  
- **Selection Status:** Pareto-Optimal  
- **Baseline Operational MUI:** 78.46 / 100  
- **Maritime Top-1 Accuracy:** 72.82%  
- **Rare Maritime Token Accuracy:** 69.61%  
- **MLM Loss:** 1.4740  
- **Weighting Sensitivity Stability:** Won 3/4 scenarios  

**Rationale:** answerdotai/ModernBERT-base demonstrated the strongest intrinsic MLM capability (72.82% Top-1 accuracy, 1.4740 MLM loss), high ranking consistency across representations (mean rank 7.0) and subsets (mean rank 1.0), consistent leader across 3/4 MUI sensitivity scenarios, and confirmed non-dominated Pareto status.

---

## 2. Data Coverage

- **Discovered Files:** 200
- **Valid Evaluated Matrix Cells:** 200
- **Expected Full Cartesian Cells:** 200
- **Missing Cells:** 0
- **Invalid Files Skipped:** 0
- **Evaluated Models (8):** allenai/scibert_scivocab_uncased, answerdotai/ModernBERT-base, bert-base-uncased, dmis-lab/biobert-base-cased-v1.2, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext, microsoft/deberta-v3-base, nlpaueb/legal-bert-base-uncased, roberta-base
- **Evaluated Representations (5):** json, key_value, mixed, narrative, template
- **Evaluated Subsets (5):** balanced_knowledge, high_knowledge, low_knowledge, medium_knowledge, random_baseline

---

## 3. Model Comparison

### Capability Metrics

| Model Name | Maritime Top-1 (%) | Top-5 (%) | Rare Top-1 (%) | MLM Loss | Pseudo-Perplexity | Baseline MUI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 72.82 | 85.18 | 69.61 | 1.4740 | 14.34 | 78.46 |
| `roberta-base` | 62.03 | 80.06 | 68.04 | 2.0423 | 18.13 | 71.75 |
| `nlpaueb/legal-bert-base-uncased` | 33.02 | 49.15 | 38.92 | 4.0722 | 40.22 | 52.13 |
| `allenai/scibert_scivocab_uncased` | 32.12 | 48.78 | 49.18 | 4.2310 | 32.10 | 62.53 |
| `bert-base-uncased` | 29.18 | 46.72 | 50.14 | 4.6544 | 24.33 | 66.48 |
| `dmis-lab/biobert-base-cased-v1.2` | 25.16 | 38.92 | 29.54 | 4.8530 | 80.88 | 55.74 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 20.14 | 30.56 | 21.52 | 5.7853 | 86.88 | 47.69 |
| `microsoft/deberta-v3-base` | 0.00 | 0.00 | 0.00 | 14.4891 | 5435378.97 | 30.0 |

### Domain / Tokenizer Fit & Operational Metrics

| Model Name | Frag Rate (%) | OOV Rate (%) | Single Token Cov (%) | Latency (ms) | Throughput (docs/s) | Parameters (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 63.3 | N/A | 36.7 | 34.8 | 28.7 | 149 |
| `roberta-base` | 65.1 | N/A | 34.9 | 27.4 | 36.5 | 125 |
| `nlpaueb/legal-bert-base-uncased` | 37.9 | 0.03 | 62.1 | 23.8 | 42.0 | 110 |
| `allenai/scibert_scivocab_uncased` | 42.1 | 0.00 | 57.9 | 23.3 | 43.0 | 110 |
| `bert-base-uncased` | 26.6 | 0.00 | 73.4 | 20.3 | 49.4 | 110 |
| `dmis-lab/biobert-base-cased-v1.2` | 35.5 | 0.00 | 64.5 | 23.0 | 43.6 | 110 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 42.7 | 0.00 | 57.3 | 23.2 | 43.1 | 110 |
| `microsoft/deberta-v3-base` | 21.5 | 0.00 | 78.5 | 17.5 | 57.1 | 86 |

---

## 4. Representation Robustness

- **Mean Pairwise Kendall's Tau (Ranking Agreement):** `0.8000`
- **Mean Pairwise Spearman's Rho:** `0.8929`

Representation breakdown across evaluated formats:

| Model | json | key_value | mixed | narrative | template | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 2 | 1.2 | 0.45 |
| `roberta-base` | 3 | 2 | 2 | 2 | 1 | 2.0 | 0.71 |
| `allenai/scibert_scivocab_uncased` | 4 | 3 | 3 | 5 | 3 | 3.6 | 0.89 |
| `nlpaueb/legal-bert-base-uncased` | 2 | 5 | 4 | 4 | 5 | 4.0 | 1.22 |
| `bert-base-uncased` | 5 | 4 | 6 | 3 | 4 | 4.4 | 1.14 |
| `dmis-lab/biobert-base-cased-v1.2` | 6 | 6 | 5 | 6 | 6 | 5.8 | 0.45 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7 | 7 | 7 | 7 | 7 | 7.0 | 0.0 |
| `microsoft/deberta-v3-base` | 8 | 8 | 8 | 8 | 8 | 8.0 | 0.0 |

---

## 5. Subset Robustness

- **Mean Pairwise Kendall's Tau (Ranking Agreement):** `0.9714`
- **Mean Pairwise Spearman's Rho:** `0.9905`

Subset ranking breakdown across knowledge/informativeness conditions:

| Model | balanced_knowledge | high_knowledge | low_knowledge | medium_knowledge | random_baseline | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 1 | 1.0 | 0.0 |
| `roberta-base` | 2 | 2 | 2 | 2 | 2 | 2.0 | 0.0 |
| `nlpaueb/legal-bert-base-uncased` | 3 | 3 | 4 | 3 | 3 | 3.2 | 0.45 |
| `allenai/scibert_scivocab_uncased` | 4 | 4 | 3 | 4 | 4 | 3.8 | 0.45 |
| `bert-base-uncased` | 5 | 5 | 5 | 5 | 5 | 5.0 | 0.0 |
| `dmis-lab/biobert-base-cased-v1.2` | 6 | 6 | 6 | 6 | 6 | 6.0 | 0.0 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7 | 7 | 7 | 7 | 7 | 7.0 | 0.0 |
| `microsoft/deberta-v3-base` | 8 | 8 | 8 | 8 | 8 | 8.0 | 0.0 |

---

## 6. MUI Sensitivity Analysis

Testing invariance across four distinct weighting hypotheses:
1. **Baseline / Operational:** Balanced operational mixture.
2. **Performance-Heavy:** Focuses strictly on intrinsic MLM accuracy and loss.
3. **Domain-Heavy:** Strongly weights rare domain terminology and domain accuracy.
4. **Balanced:** Equal weighting across capability, tokenizer fit, and throughput efficiency.

| Model Name | Baseline Score (Rank) | Perf-Heavy Score (Rank) | Domain-Heavy Score (Rank) | Balanced Score (Rank) | Total Wins | Win Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 78.5 (#1) | 100.0 (#1) | 84.0 (#1) | 60.8 (#3) | 3 | 75% |
| `bert-base-uncased` | 66.5 (#3) | 57.6 (#5) | 69.8 (#3) | 69.8 (#1) | 1 | 25% |
| `roberta-base` | 71.8 (#2) | 91.6 (#2) | 77.6 (#2) | 61.2 (#2) | 0 | 0% |
| `allenai/scibert_scivocab_uncased` | 62.5 (#4) | 60.1 (#3) | 65.5 (#4) | 59.3 (#4) | 0 | 0% |
| `dmis-lab/biobert-base-cased-v1.2` | 55.7 (#5) | 47.7 (#6) | 54.8 (#5) | 54.2 (#6) | 0 | 0% |
| `nlpaueb/legal-bert-base-uncased` | 52.1 (#6) | 58.0 (#4) | 52.3 (#6) | 58.1 (#5) | 0 | 0% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 47.7 (#7) | 39.4 (#7) | 45.5 (#7) | 45.5 (#7) | 0 | 0% |
| `microsoft/deberta-v3-base` | 30.0 (#8) | 0.0 (#8) | 25.0 (#8) | 40.0 (#8) | 0 | 0% |

---

## 7. Pareto Dominance Analysis

| Model Name | Pareto Status | Dominates Count | Dominated By Count | Dominating Models |
| :--- | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `answerdotai/ModernBERT-base` | **Pareto-Optimal** | 0 | 0 | None |
| `bert-base-uncased` | **Pareto-Optimal** | 2 | 0 | None |
| `microsoft/deberta-v3-base` | **Pareto-Optimal** | 0 | 0 | None |
| `nlpaueb/legal-bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `roberta-base` | **Pareto-Optimal** | 0 | 0 | None |
| `dmis-lab/biobert-base-cased-v1.2` | **Dominated** | 1 | 1 | bert-base-uncased |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **Dominated** | 0 | 2 | bert-base-uncased, dmis-lab/biobert-base-cased-v1.2 |

---

## 8. Capability vs. Resource Trade-offs

### Capability vs. Tokenizer/Domain Fit
- `answerdotai/ModernBERT-base` achieves highest overall intrinsic MLM performance, but exhibits higher subword fragmentation than specialized WordPiece architectures.
- Models with lower subword fragmentation (e.g. `bert-base-uncased` at 26.6% fragmentation) offer better morphological token boundaries for specific domain stems, despite lower overall MLM Top-1 accuracy.

### Capability vs. Operational Cost
- `answerdotai/ModernBERT-base` requires 149M parameters and ~458.6ms latency.
- Lightweight alternatives (e.g., 110M base models) provide faster inference latency (down to ~154-175ms) with smaller disk and memory footprints.

### Alternative Trade-off Details
- **Alternative `allenai/scibert_scivocab_uncased`:** Offers lower subword fragmentation (42.1% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (23.3ms vs 34.8ms).
- **Alternative `bert-base-uncased`:** Offers lower subword fragmentation (26.6% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (20.3ms vs 34.8ms).
- **Alternative `microsoft/deberta-v3-base`:** Offers lower subword fragmentation (21.5% vs 63.3%); smaller footprint (86M vs 149M params); lower latency (17.5ms vs 34.8ms).
- **Alternative `nlpaueb/legal-bert-base-uncased`:** Offers lower subword fragmentation (37.9% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (23.8ms vs 34.8ms).
- **Alternative `roberta-base`:** Offers smaller footprint (125M vs 149M params); lower latency (27.4ms vs 34.8ms).

---

## 9. Methodological Note

> **Notice on Composite Scoring:**  
> The Maritime Understanding Index (MUI) is an operational composite compatibility score rather than a human-validated measure of innate maritime understanding. The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, rare-token domain generalization, representation consistency, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.
