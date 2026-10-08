# Comprehensive Maritime NLP Corpus Quality Report

This report evaluates the scale, document length, structural diversity, scaffolding influence, BERT tokenizer compatibility, Maritime Information Density (MID), and pretraining readiness of the maritime corpus.

---

## 1. Corpus Scale & Information Density
* **Total Documents**: 96,874
* **Total Words (Tokens)**: 3,734,709
* **Total Characters**: 24,357,090
* **Unique Vocabulary**: 42,780 terms
* **Maritime Information Density (MID)**: **3.67** concepts / 100 words

---

## 2. Document Length Distribution
* **Mean Length**: 38.55 words
* **Median Length**: 37.00 words
* **Standard Deviation**: 20.17 words
* **Percentiles**: P10=10, P25=25, P50=37, P75=51, P90=67, P95=73
* **Min / Max**: 4 / 513 words

### Length Buckets
* `<20 words`: 18,260 (18.8%)
* `20–50 words`: 53,286 (55.0%)
* `50–100 words`: 25,030 (25.8%)
* `100–200 words`: 274 (0.3%)
* `200–512 words`: 23 (0.0%)
* `>512 words`: 1 (0.0%)

---

## 3. Linguistic Diversity
* **Type-Token Ratio (TTR)**: 0.01145
* **Shannon Entropy**: 8.3172 bits
* **Unique Sentences**: 145,691
* **Unique Paragraphs**: 96,868

---

## 4. Duplication & Near-Duplicate Analysis
* **Sentence Duplicate Ratio**: 9.44%
* **Paragraph Duplicate Ratio**: 0.01%
* **Scaffold-Reduced Near-Duplicate Rate (MinHash LSH)**: 16.72%
* **Template Pattern Concentration**: 41.82% (Top pattern: `raw_tsb_summary`)

---

## 5. Maritime Domain Coverage
* **Top Domain Bigrams**: 'navigation equipment', 'magnetic compass', 'equipment included', 'vhf radio', 'equipment reported'
* **Top Domain Trigrams**: 'navigation equipment reported', 'equipment reported inactive', 'reported inactive included', 'active navigation equipment', 'navigation equipment included'
* **Top Domain 4-Grams**: 'navigation equipment reported inactive', 'equipment reported inactive included', 'active navigation equipment included', 'under clear weather and', 'weather and calm glassy'

---

## 6. Template Influence
* **Template Scaffolding Token Ratio**: 58.47%
* **Domain-Derived Token Ratio**: 41.53%

---

## 7. BERT Tokenizer Baseline Compatibility
* **BERT Model**: `bert-base-uncased`
* **Tokenizer Fertility (Subwords/Word)**: 1.4132
* **Maritime Fragmentation Rate**: 9.09%
* **OOV / [UNK] Rate**: 0.0000%
* *Cross-Stage Note*: Stage 09 computes an independent corpus-level BERT baseline diagnostic. Stage 13 provides the authoritative multi-candidate comparative tokenizer benchmark across the candidate pool.

---

## 8. Cross-Model MLM Evaluation Reference
* **Status**: Evaluated in Stage 14
* *Cross-Stage Note*: Systematic Masked Language Modeling (MLM) benchmarking across candidate models, whole-word masking, and structural representations is conducted authoritatively in Stage 14 (14_mlm_evaluation.py). Stage 09 remains strictly confined to corpus-level statistics.

---

## 9. Multi-Dimensional Readiness Dimensions
* ✅ **Relational Integrity**: PASS
* ✅ **Linguistic Quality**: PASS
* ⚠️ **Semantic Density**: WARN
* ✅ **Duplication**: PASS
* ⚠️ **Template Influence**: WARN
* ✅ **Domain Coverage**: PASS
* ✅ **Bert Compatibility**: PASS

---

## 10. Pretraining Readiness Assessment

# Status: **READY WITH WARNINGS**

* **Assessment Summary**: Corpus evaluation across 7 quality dimensions.
