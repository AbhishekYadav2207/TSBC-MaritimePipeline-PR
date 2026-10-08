# Forensic Code Diff Report: Stage-14 Historical vs Current Implementation
**Authoritative Legacy Commit Reference**: `3602aad`  
**Current Working Commit**: `37cd3fd`  
**Target File**: `scripts/14_mlm_evaluation.py`  
**Audit Date**: 2026-10-08  
**Audit Objective**: Identify all divergences between the historical primary Stage-14 protocol and subsequent modifications, ensuring the restoration preserves the legacy scientific protocol while retaining validated bug fixes.

---

## 1. Executive Summary
A comprehensive line-by-line AST and unified diff analysis was conducted between commit `3602aad` and the current `scripts/14_mlm_evaluation.py`.

The analysis reveals that while recent commits introduced Whole-Word Masking (WWM) capabilities to address Review Issue A1 (subword sibling piece leakage), the script's CLI argument defaults were inadvertently switched from the primary legacy protocol (`subword`) to WWM (`whole_word` / `word`). Furthermore, several critical bug fixes (such as BPE space-prefix matching and stopword filtering) were introduced alongside WWM that must be preserved within the restored legacy subword protocol.

---

## 2. Forensic Categorization of Changes

### A. Masking Protocol & CLI Defaults
* **Commit `3602aad`**:
  * Evaluated strictly with `masking_strategy="random_15"`, `masking_mode="subword"`, `evaluation_unit="subword"`.
  * No WWM functions existed.
* **Current Script (`37cd3fd`)**:
  * Introduced arguments: `--masking_mode` (choices: `subword`, `whole_word`, `wwm_subword`, `wwm_word`) and `--evaluation_unit` (choices: `subword`, `word`).
  * **Divergence / Issue**: Default values were set to `--masking_mode whole_word` and `--evaluation_unit word`, thereby silently replacing the primary benchmark with the secondary WWM analysis.
* **Restoration Target (Phase 3C)**:
  * Default MUST be restored to:
    * `masking_mode = "subword"`
    * `evaluation_unit = "subword"`
    * `masking_strategy = "random_15"`
  * WWM becomes an explicit, opt-in secondary robustness analysis.

---

### B. Target Classification Logic (Preserving Validated Bug Fixes)
* **Commit `3602aad` (Known Bugs in Target Identification)**:
  * In `build_vocabulary_token_sets`:
    ```python
    # Legacy 3602aad
    sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(term))
    maritime_token_ids.update(sub_ids)
    ```
    * *Defect 1 (BPE prefix failure)*: Did not generate leading space variations (e.g., `Ġcargo` vs `cargo` in RoBERTa / ModernBERT BPE).
    * *Defect 2 (Stopword leakage)*: Multi-word domain phrases like `"search and rescue"` inadvertently added common function words like `"and"` to `maritime_token_ids`.
* **Current Script (`37cd3fd`) (Validated Forensic Fixes)**:
  * Added `EXCLUDED_STOPWORDS = {"and", "the", "of", "in", "is", "for", "to", "with", "at", "by", "from", "on", "a", "an", "or", "as", "into", "through"}`.
  * Generates both un-spaced and leading-space tokenizations: `sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w))` and `sub_ids_spaced = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + w))`.
  * Decodes token IDs and verifies that single-character pieces or decoded stopwords are excluded from maritime token sets.
* **Restoration Target (Phase 3D)**:
  * RETAIN the validated BPE prefix and stopword filtering logic.
  * Do NOT revert to the naive `3602aad` tokenizer matching.

---

### C. Seed Protocol & Determinism
* **Commit `3602aad`**:
  * Used `stable_seed(*parts)` with SHA-256:
    ```python
    def stable_seed(*parts) -> int:
        key = "||".join(str(p) for p in parts)
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
    ```
  * Cell seed: `stable_seed(model_name, rep, sub, "random_15")`.
* **Current Script (`37cd3fd`)**:
  * Retains the identical `stable_seed` implementation.
  * Seed protocol is 100% consistent with legacy.

---

### D. Scoring & Evaluation Unit
* **Commit `3602aad`**:
  * Evaluated position-level cross-entropy loss, Top-1, Top-5, Top-10 at the subword token level.
  * Tracked overall summary, general tokens summary, maritime tokens summary, rare maritime summary, and category breakdowns.
* **Current Script (`37cd3fd`)**:
  * In `subword` mode, evaluates identical subword metrics.
  * In `word` mode, added `word_eval_stats` (strict multi-piece reconstruction).
* **Restoration Target (Phase 3E)**:
  * Primary evaluation unit is strictly `subword`.
  * Format output with explicit separated fields: `overall`, `maritime_target`, `rare_target`, and `token_statistics`.
  * Empty or undefined slices MUST emit `null`/`NA`, never `0.0` or `NaN`.

---

### E. Cache Isolation & Namespace Management
* **Commit `3602aad`**:
  * All cell records were written to `outputs/stage-14/evaluations/cache/`.
* **Current Script (`37cd3fd`)**:
  * When in WWM mode, wrote to `outputs/stage-14/evaluations/cache_wwm_word/`.
* **Restoration Target (Phase 3F)**:
  * Regenerate primary benchmark into `outputs/stage-14/evaluations/cache_legacy_subword_15/`.
  * Preserve historical `outputs/stage-14/evaluations/cache/` intact for regression comparison.
  * Preserve `outputs/stage-14/evaluations/cache_wwm_word/` as secondary robustness analysis.
  * Tag every output record with `execution_type = "PRIMARY"` or `"SECONDARY"`.

---

## 3. Forensic Code Change Summary Table

| Functional Component | Legacy `3602aad` State | Current `37cd3fd` State | Planned Restored State |
| :--- | :--- | :--- | :--- |
| **Masking Strategy Default** | `random_15` | `whole_word` | `random_15` (PRIMARY default) |
| **Masking Mode Default** | `subword` | `whole_word` | `subword` (PRIMARY default) |
| **Evaluation Unit Default** | `subword` | `word` | `subword` (PRIMARY default) |
| **WWM Mode Availability** | None | Active as default | Retained as secondary opt-in flag |
| **BPE Space Prefixes (`Ġ`)** | Not handled | Handled via `" " + w` | Handled via `" " + w` (retained fix) |
| **Stopword Filtering** | Leaked into domain | Excluded via stopword set | Excluded via stopword set (retained fix) |
| **Cell Seed Protocol** | SHA-256 stable seed | SHA-256 stable seed | SHA-256 stable seed (identical) |
| **Output Schema** | Implicit nested dict | Modified for WWM words | Strict contract with separated scopes & NA |
| **Cache Directory** | `evaluations/cache` | `evaluations/cache_wwm_word` | `evaluations/cache_legacy_subword_15` |
| **Execution Type Tag** | Missing | Missing | Explicit `"PRIMARY"` tag |

---

## 4. Conclusion & Directive
The target script must reconcile the historical `3602aad` scientific protocol with the validated bug fixes. Whole-Word Masking remains available for secondary sensitivity analysis, but the default execution path of `scripts/14_mlm_evaluation.py` will unconditionally execute the primary legacy subword MLM benchmark.
