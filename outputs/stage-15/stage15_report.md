# Stage 15: Cross-Model Benchmarking & Defensible Model Selection Report

## 1. Executive Conclusion

**Recommended Model:** `answerdotai/ModernBERT-base`  
- **Selection Status:** Pareto-Optimal  
- **Baseline Operational MUI:** 71.46 / 100  
- **Maritime Top-1 Accuracy:** 56.04%  
- **Rare Maritime Token Accuracy:** 29.09%  
- **MLM Loss:** 2.3063  
- **Weighting Sensitivity Stability:** Won 2/4 scenarios  

**Rationale:** answerdotai/ModernBERT-base demonstrated the strongest overall intrinsic MLM capability (56.04% Top-1 accuracy, 2.3063 MLM loss), unbroken ranking invariance across all evaluated representations and subsets, leading performance across 2/4 MUI sensitivity scenarios (baseline and performance-heavy paradigms), and confirmed membership in the non-dominated Pareto front.

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

## 3. Model Comparison (Raw & Direction-Normalized Metrics)

| Model Name | Maritime Top-1 (%) | Rare Top-1 (%) | MLM Loss | Frag Rate (%) | Latency (ms) | Throughput (docs/s) | Baseline MUI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 56.04 | 29.09 | 2.3063 | 63.3 | 458.6 | 2.2 | 71.46 |
| `roberta-base` | 47.03 | 30.07 | 2.7948 | 65.1 | 382.7 | 2.6 | 61.03 |
| `allenai/scibert_scivocab_uncased` | 31.24 | 6.44 | 4.2628 | 42.1 | 323.4 | 3.1 | 37.09 |
| `bert-base-uncased` | 29.08 | 54.35 | 4.6237 | 26.6 | 174.9 | 5.7 | 59.25 |
| `nlpaueb/legal-bert-base-uncased` | 28.66 | 4.49 | 4.4541 | 37.9 | 249.3 | 4.0 | 27.08 |
| `dmis-lab/biobert-base-cased-v1.2` | 24.65 | 32.35 | 4.9969 | 35.5 | 154.1 | 6.5 | 42.24 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 20.59 | 3.40 | 5.7259 | 42.7 | 326.8 | 3.1 | 23.72 |

---

## 4. Representation Robustness

- **Mean Pairwise Kendall's Tau (Rank Stability):** `0.7143`
- **Mean Pairwise Spearman's Rho:** `0.8071`

Representation breakdown across evaluated formats:

| Model | json | key_value | mixed | narrative | template | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 1 | 1.0 | 0.0 |
| `roberta-base` | 2 | 2 | 2 | 2 | 2 | 2.0 | 0.0 |
| `allenai/scibert_scivocab_uncased` | 5 | 3 | 3 | 4 | 3 | 3.6 | 0.89 |
| `nlpaueb/legal-bert-base-uncased` | 3 | 4 | 4 | 5 | 6 | 4.4 | 1.14 |
| `bert-base-uncased` | 6 | 5 | 5 | 3 | 4 | 4.6 | 1.14 |
| `dmis-lab/biobert-base-cased-v1.2` | 4 | 6 | 6 | 6 | 7 | 5.8 | 1.1 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 7 | 7 | 7 | 7 | 5 | 6.6 | 0.89 |

---

## 5. Subset Robustness

- **Mean Pairwise Kendall's Tau (Rank Stability):** `0.9238`
- **Mean Pairwise Spearman's Rho:** `0.9571`

Subset ranking breakdown across knowledge/informativeness conditions:

| Model | balanced_knowledge | high_knowledge | low_knowledge | medium_knowledge | random_baseline | Mean Rank | Rank Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `answerdotai/ModernBERT-base` | 1 | 1 | 1 | 1 | 1 | 1.0 | 0.0 |
| `roberta-base` | 2 | 2 | 2 | 2 | 2 | 2.0 | 0.0 |
| `allenai/scibert_scivocab_uncased` | 3 | 3 | 3 | 4 | 3 | 3.2 | 0.45 |
| `bert-base-uncased` | 4 | 4 | 4 | 5 | 4 | 4.2 | 0.45 |
| `nlpaueb/legal-bert-base-uncased` | 5 | 5 | 5 | 3 | 5 | 4.6 | 0.89 |
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
| `answerdotai/ModernBERT-base` | 71.5 (#1) | 90.1 (#1) | 68.3 (#2) | 51.0 (#3) | 2 | 50% |
| `bert-base-uncased` | 59.2 (#3) | 43.4 (#3) | 70.8 (#1) | 67.7 (#1) | 2 | 50% |
| `roberta-base` | 61.0 (#2) | 74.7 (#2) | 59.8 (#3) | 44.5 (#4) | 0 | 0% |
| `dmis-lab/biobert-base-cased-v1.2` | 42.2 (#4) | 23.5 (#6) | 47.5 (#4) | 53.3 (#2) | 0 | 0% |
| `allenai/scibert_scivocab_uncased` | 37.1 (#5) | 29.9 (#4) | 35.0 (#5) | 31.9 (#6) | 0 | 0% |
| `nlpaueb/legal-bert-base-uncased` | 27.1 (#6) | 24.0 (#5) | 22.6 (#6) | 35.0 (#5) | 0 | 0% |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 23.7 (#7) | 0.0 (#7) | 18.7 (#7) | 15.7 (#7) | 0 | 0% |

---

## 7. Pareto Dominance Analysis

| Model Name | Pareto Status | Dominates Count | Dominated By Count | Dominating Models |
| :--- | :---: | :---: | :---: | :--- |
| `allenai/scibert_scivocab_uncased` | **Pareto-Optimal** | 1 | 0 | None |
| `answerdotai/ModernBERT-base` | **Pareto-Optimal** | 0 | 0 | None |
| `bert-base-uncased` | **Pareto-Optimal** | 1 | 0 | None |
| `dmis-lab/biobert-base-cased-v1.2` | **Pareto-Optimal** | 1 | 0 | None |
| `nlpaueb/legal-bert-base-uncased` | **Pareto-Optimal** | 0 | 0 | None |
| `roberta-base` | **Pareto-Optimal** | 0 | 0 | None |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | **Dominated** | 0 | 3 | allenai/scibert_scivocab_uncased, bert-base-uncased, dmis-lab/biobert-base-cased-v1.2 |

---

## 8. Trade-offs and Architectural Considerations

- **Alternative `allenai/scibert_scivocab_uncased`:** Offers lower subword fragmentation (42.1% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (323.4ms vs 458.6ms).
- **Alternative `bert-base-uncased`:** Offers lower subword fragmentation (26.6% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (174.9ms vs 458.6ms).
- **Alternative `dmis-lab/biobert-base-cased-v1.2`:** Offers lower subword fragmentation (35.5% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (154.1ms vs 458.6ms).
- **Alternative `nlpaueb/legal-bert-base-uncased`:** Offers lower subword fragmentation (37.9% vs 63.3%); smaller footprint (110M vs 149M params); lower latency (249.3ms vs 458.6ms).
- **Alternative `roberta-base`:** Offers smaller footprint (125M vs 149M params); lower latency (382.7ms vs 458.6ms).

---

## 9. Methodological Note

> **Notice on Composite Scoring:**  
> The Maritime Understanding Index (MUI) is an operational composite compatibility score rather than a human-validated measure of innate maritime understanding. The selection of the final model is founded on a defensible, multi-criteria evidence hierarchy comprising direction-normalized intrinsic MLM accuracy, rare-token domain generalization, representation invariance, knowledge subset robustness, sensitivity analysis invariance, and non-dominated Pareto status.
