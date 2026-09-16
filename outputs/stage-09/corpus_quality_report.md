# Comprehensive Maritime NLP Corpus Quality Report

This report evaluates the scale, document length, structural diversity, scaffolding influence, BERT tokenizer compatibility, Maritime Information Density (MID), and pretraining readiness of the maritime corpus.

---

## 1. Corpus Scale & Information Density
* **Total Documents**: 96,861
* **Total Words (Tokens)**: 3,734,580
* **Total Characters**: 24,356,921
* **Unique Vocabulary**: 42,780 terms
* **Maritime Information Density (MID)**: **3.66** concepts / 100 words

---

## 2. Document Length Distribution
* **Mean Length**: 38.56 words
* **Median Length**: 37.00 words
* **Standard Deviation**: 20.17 words
* **Percentiles**: P10=10, P25=25, P50=37, P75=51, P90=67, P95=73
* **Min / Max**: 4 / 513 words

### Length Buckets
* `<20 words`: 18,206 (18.8%)
* `20–50 words`: 53,365 (55.1%)
* `50–100 words`: 24,995 (25.8%)
* `100–200 words`: 271 (0.3%)
* `200–512 words`: 23 (0.0%)
* `>512 words`: 1 (0.0%)

---

## 3. Linguistic Diversity
* **Type-Token Ratio (TTR)**: 0.01146
* **Shannon Entropy**: 8.3170 bits
* **Unique Sentences**: 145,691
* **Unique Paragraphs**: 96,856

---

## 4. Duplication & Near-Duplicate Analysis
* **Sentence Duplicate Ratio**: 9.42%
* **Paragraph Duplicate Ratio**: 0.01%
* **Scaffold-Reduced Near-Duplicate Rate (MinHash LSH)**: 16.76%
* **Template Pattern Concentration**: 41.82% (Top pattern: `raw_tsb_summary`)

---

## 5. Maritime Domain Coverage
* **Top Domain Bigrams**: 'navigation equipment', 'magnetic compass', 'equipment included', 'vhf radio', 'equipment reported'
* **Top Domain Trigrams**: 'navigation equipment reported', 'equipment reported inactive', 'reported inactive included', 'active navigation equipment', 'navigation equipment included'
* **Top Domain 4-Grams**: 'navigation equipment reported inactive', 'equipment reported inactive included', 'active navigation equipment included', 'under clear weather and', 'weather and calm glassy'

---

## 6. Template Influence
* **Template Scaffolding Token Ratio**: 58.48%
* **Domain-Derived Token Ratio**: 41.52%

---

## 7. BERT Tokenizer Compatibility
* **BERT Model**: `bert-base-uncased`
* **Tokenizer Fertility (Subwords/Word)**: 1.3989
* **Maritime Fragmentation Rate**: 26.57%
* **OOV / [UNK] Rate**: 0.0000%

---

## 8. BERT MLM Baseline Diagnostic
* **MLM Evaluation Model**: `bert-base-uncased`
* **General Tokens Top-1 Accuracy**: 0.00%
* **Maritime Tokens Top-1 Accuracy**: 0.00%
* **Performance Gap**: 0.00%

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
