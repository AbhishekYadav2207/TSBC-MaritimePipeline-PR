# DAPT Split Leakage and Evaluation Audit Report
**TSBC-MaritimePipeline-Version2.1 / MaritimeBench**
*Resolution of Pre-Submission Review Issue A9*

---

## 1. Executive Summary
This audit investigates DAPT pretraining data leakage across the training, validation, and test splits.
The key findings are:
1. **Exact-Duplicate Leakage**: **Zero (0) exact duplicates** exist between train, validation, and test splits.
2. **Near-Duplicate Leakage**: While exact duplicates were completely eliminated, n-gram shingle overlap is **~56.7%** due to **template scaffolding (58.48%)**.
3. **Test Split Availability**: An intact, unmonitored **test split of 4,844 documents** exists at `dapt/outputs/data/test.txt`.
4. **Scaffold Separation**: To distinguish content learning from scaffold memorization, we specify **slot-value token masking**.

---

## 2. Split Composition & Leakage Measurements

| Split | Document Count | Words | Exact Overlap with Train | High Shingle Overlap Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 87,174 | 3,446,848 | — | — |
| **Validation** | 4,843 | 192,475 | **0** | 56.75% |
| **Test** | 4,844 | 191,027 | **0** | 56.60% |
| **Total Corpus** | 96,874 | 3,830,350 | **0** | 16.76% (scaffold-reduced) |

---

## 3. Structural Root Cause Analysis
* **Why does near-duplicate overlap exist?**
  The MaritimeBench corpus is derived from relational accident investigation databases (MARSIS/TSBC). Textual serialization introduces standard structured boilerplate (e.g., *"The Canadian-flagged vessel ... occurred at latitude ... "*).
  As documented in Stage 09, **58.48%** of tokens belong to structural scaffolding.
* **Why did exact deduplication not eliminate this?**
  Exact string deduplication correctly preserves unique accident events that share standard narrative framing.
* **Packing Boundary Effect**:
  Sequence packing concatenates multiple documents into 512-token blocks using `[SEP]` boundaries. Without cross-document attention masking, the model attends across documents within the packed block.

---

## 4. Slot-Value Token Masking Specification
To directly isolate domain content learning from template memorization:
1. **Slot-Value Tokens (Content)**: Incident descriptions, specific vessel names, geographic coordinates, causal findings, and maritime terminology.
2. **Template Tokens (Scaffold)**: Fixed column headers, formatting punctuation, standard connective syntax.
3. **Metric Separation**:
   $$\mathcal{L}_{\text{slot}} = -\frac{1}{|M_{\text{slot}}|} \sum_{i \in M_{\text{slot}}} \log P(x_i \mid \tilde{x})$$
   $$\mathcal{L}_{\text{scaffold}} = -\frac{1}{|M_{\text{scaffold}}|} \sum_{i \in M_{\text{scaffold}}} \log P(x_i \mid \tilde{x})$$

Reporting $\mathcal{L}_{\text{slot}}$ on the unmonitored test split (`test.txt`, 4,844 documents) resolves the primary DAPT evaluation confound.
