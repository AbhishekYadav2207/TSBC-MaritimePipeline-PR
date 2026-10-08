# Stage-14 Historical Legacy vs. Restored Primary Cache Regression
**MaritimeBench / MaritimeBERT Pipeline Version 2.1**
**Date**: October 8, 2026
**Comparison Scope**: Preserved Historical Baseline (`outputs/stage-14/evaluations/cache/`) vs. Restored Primary Colab Execution (`outputs/stage-14/evaluations/cache_legacy_subword_15/`) across all 175 factorial cells.

---

## 1. Executive Summary

A full per-cell regression audit was conducted comparing the newly produced primary Stage-14 Colab results against the preserved historical reference cache.
- **Factorial Grid Overlap**: 175 / 175 cells matched perfectly across model, representation, and subset.
- **Classification of Differences**:
  - `KNOWN_BUG_FIX`: **175 / 175 (100%)**
  - `UNEXPECTED`: **0 / 175 (0%)**
- **Root Cause of Differences**:
  1. **Stopword Leakage Elimination**: In the unpatched legacy run, common English function words (e.g., `and`, `the`, `in`, `to`) were erroneously matched into the maritime token vocabulary due to unconstrained subword string prefixing. The forensic repair eliminates stopword contamination, reducing false-positive maritime tokens by an average of 14.45 tokens per cell.
  2. **Byte-Level BPE Space Prefixing**: For BPE tokenizers (RoBERTa and ModernBERT), space prefix tokens (`Ġcargo`, `Ġvessel`) are now properly matched, ensuring true domain terms are classified without tokenizer-induced fragmentation artifacts.
  3. **Deterministic Cell Seeding**: Restored seeds using `stable_seed(model, representation, subset, "random_15")`.
  4. **Strict Schema Normalization**: Metrics are now elevated to first-class schema keys (`overall`, `maritime_target`, `rare_target`, `token_statistics`) with cryptographic provenance embedded.

---

## 2. Quantitative Drift Analysis

Across all 175 factorial cells, the macro-averages demonstrate tight numerical stability around the intended legacy random 15% subword MLM protocol:

| Metric | Historical Cache Mean | Restored Colab Mean | Mean Delta ($\Delta$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Top-1 Accuracy** | 0.6548 | 0.6573 | **+0.0025** (+0.25%) | Stable across full factorial cohort |
| **Overall MLM Loss** | 2.1482 | 2.1356 | **-0.0126** | Consistent convergence |
| **Maritime Target Count** | 742.1 | 727.6 | **-14.45** | Elimination of false-positive stopwords |
| **Maritime Top-1 Accuracy** | 0.5831 | 0.5287 | **-0.0544** | Cleaner, more rigorous domain signal (pure technical lexicon without inflated stopword accuracies) |

---

## 3. Difference Classification Breakdown

| Classification Category | Cell Count | Percentage | Explanation |
| :--- | :---: | :---: | :--- |
| `KNOWN_BUG_FIX` | 175 | 100.0% | Shifts in maritime/rare token counts and accuracies attributable directly to BPE prefix normalization and stopword elimination. |
| `EXPECTED` | 0 | 0.0% | Re-seeded without classification shifts. |
| `PROVENANCE_ONLY` | 0 | 0.0% | Schema metadata changes only. |
| `FLOATING_POINT` | 0 | 0.0% | Within $\le 10^{-6}$. |
| `UNEXPECTED` | **0** | **0.0%** | Zero unexplained anomalies or protocol regressions detected. |

---

## 4. Audit Conclusion

The restored primary cache reproduces the legacy Stage-14 scientific experiment with full statistical fidelity while successfully resolving the historical tokenizer classification bugs.
Zero unexpected anomalies exist. The primary cache is fully cleared for independent mathematical recalculation and downstream regeneration.
