# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report

## 1. Executive Conclusion

**Recommended Model:** `answerdotai/ModernBERT-base`  
- **Selection Status:** Pareto-Optimal  
- **Baseline Operational MUI:** 78.56 / 100  
- **Maritime Top-1 Accuracy:** 73.17%  
- **Rare Maritime Token Accuracy:** 73.03%  
- **MLM Loss:** 1.4269  
- **Weighting Sensitivity Stability:** Won 4/4 scenarios  

**Rationale:** answerdotai/ModernBERT-base demonstrated the strongest intrinsic MLM capability (73.17% Top-1 accuracy, 1.4269 MLM loss), high ranking consistency across representations (mean rank 1.2) and subsets (mean rank 1.0), unanimous leader across all 4 MUI sensitivity scenarios, and confirmed non-dominated Pareto status.

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
| `answerdotai/ModernBERT-base` | 73.17 | 85.58 | 73.03 | 1.4269 | 18.33 | 78.56 |
| `roberta-base` | 61.49 | 79.64 | 64.40 | 2.0526 | 18.72 | 63.84 |
| `nlpaueb/legal-bert-base-uncased` | 33.14 | 49.08 | 39.11 | 4.1012 | 48.00 | 35.93 |
| `allenai/scibert_scivocab_uncased` | 32.18 | 49.11 | 49.89 | 4.2238 | 31.16 | 46.89 |
| `bert-base-uncased` | 28.91 | 46.53 | 47.76 | 4.6808 | 22.25 | 49.01 |
| `dmis-lab/biobert-base-cased-v1.2` | 25.46 | 38.83 | 30.23 | 4.8310 | 86.14 | 35.81 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 20.50 | 30.22 | 22.81 | 5.8097 | 64.62 | 23.79 |

### Domain / Tokenizer Fit & Operational Metrics

| Model Name | Frag Rate (%) | OOV Rate (%) | Single Token Cov (%) | Latency (ms) | Throughput (docs/s) | Parameters (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 63.0 | N/A | 37.0 | 34.1 | 29.3 | 149 |
| `roberta-base` | 64.8 | N/A | 35.2 | 26.9 | 37.2 | 125 |
| `nlpaueb/legal-bert-base-uncased` | 37.6 | 0.03 | 62.4 | 24.3 | 41.1 | 110 |
| `allenai/scibert_scivocab_uncased` | 41.8 | 0.00 | 58.2 | 23.9 | 41.8 | 110 |
| `bert-base-uncased` | 26.6 | 0.00 | 73.4 | 22.4 | 44.7 | 110 |
| `dmis-lab/biobert-base-cased-v1.2` | 35.5 | 0.00 | 64.5 | 24.4 | 40.9 | 110 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 42.4 | 0.00 | 57.6 | 23.4 | 42.8 | 110 |

---

## 4. Representation Robustness

- **Mean Pairwise Kendall's Tau (Ranking Agreement):** `0.6952`
- **Mean Pairwise Spearman's Rho:** `0.8107`

Representation breakdown across evaluated formats:

| Model | json | key_value | mixed | narrative | template | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 2 | 1.2 | 0.45 |
| `roberta-base` | 3 | 2 | 2 | 2 | 1 | 2.0 | 0.71 |
| `nlpaueb/legal-bert-base-uncased` | 2 | 5 | 3 | 4 | 4 | 3.6 | 1.14 |
| `allenai/scibert_scivocab_uncased` | 4 | 3 | 4 | 5 | 3 | 3.8 | 0.84 |
| `bert-base-uncased` | 5 | 4 | 7 | 3 | 5 | 4.8 | 1.48 |
| `dmis-lab/biobert-base-cased-v1.2` | 6 | 6 | 5 | 6 | 6 | 5.8 | 0.45 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7 | 7 | 6 | 7 | 7 | 6.8 | 0.45 |

---

## 5. Subset Robustness

- **Mean Pairwise Kendall's Tau (Ranking Agreement):** `0.9619`
- **Mean Pairwise Spearman's Rho:** `0.9857`

Subset ranking breakdown across knowledge/informativeness conditions:

| Model | balanced_knowledge | high_knowledge | low_knowledge | medium_knowledge | random_baseline | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 1 | 1.0 | 0.0 |
| `roberta-base` | 2 | 2 | 2 | 2 | 2 | 2.0 | 0.0 |
| `nlpaueb/legal-bert-base-uncased` | 4 | 3 | 3 | 3 | 3 | 3.2 | 0.45 |
| `allenai/scibert_scivocab_uncased` | 3 | 4 | 4 | 4 | 4 | 3.8 | 0.45 |
| `bert-base-uncased` | 5 | 5 | 5 | 5 | 5 | 5.0 | 0.0 |
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
| `roberta-base` | 63.8 (#2) | 82.5 (#2) | 68.1 (#2) | 59.6 (#2) | 0 | 0% |
| `bert-base-uncased` | 49.0 (#3) | 27.2 (#5) | 50.2 (#3) | 58.3 (#3) | 0 | 0% |
| `allenai/scibert_scivocab_uncased` | 46.9 (#4) | 33.8 (#3) | 48.9 (#4) | 50.7 (#4) | 0 | 0% |
| `nlpaueb/legal-bert-base-uncased` | 35.9 (#5) | 30.9 (#4) | 33.9 (#5) | 48.6 (#5) | 0 | 0% |
| `dmis-lab/biobert-base-cased-v1.2` | 35.8 (#6) | 14.6 (#6) | 32.4 (#6) | 39.7 (#6) | 0 | 0% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 23.8 (#7) | 0.0 (#7) | 18.8 (#7) | 29.2 (#7) | 0 | 0% |

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
- **Alternative `allenai/scibert_scivocab_uncased`:** Offers lower subword fragmentation (41.8% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (23.9ms vs 34.1ms).
- **Alternative `bert-base-uncased`:** Offers lower subword fragmentation (26.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (22.4ms vs 34.1ms).
- **Alternative `nlpaueb/legal-bert-base-uncased`:** Offers lower subword fragmentation (37.6% vs 63.0%); smaller footprint (110M vs 149M params); lower latency (24.3ms vs 34.1ms).
- **Alternative `roberta-base`:** Offers smaller footprint (125M vs 149M params); lower latency (26.9ms vs 34.1ms).

---

## 9. Methodological Note

> **Notice on Composite Scoring:**  
> The Maritime Understanding Index (MUI) is an operational composite compatibility score rather than a human-validated measure of innate maritime understanding. The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, rare-token domain generalization, representation consistency, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.
