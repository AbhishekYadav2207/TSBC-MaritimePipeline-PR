# MaritimeBench: Representation Provenance & Cryptographic Traceability Manifest

**Generated:** `2026-10-07T15:10:04.725242+00:00` | **Git Commit:** `b1160bd92836234f0e56b2e5d8c2c66d8bf7a49c` | **Pipeline Version:** `2.1`  
**Target Corpus:** Frozen Corpus A (MARSIS / TSB Canadian Maritime Corpus)  

---

## 1. Frozen Source Corpus Lineage

| Artifact Role | File Path | SHA-256 Checksum | Size |
|---|---|---|---|
| **Text Corpus** | `outputs/stage-08/maritime_corpus.txt` | `51f7a9cd897ccda060fd10c9661a8f8129ebe7a6c4e19846cb6c931bd9c099cd` | 24,549,202 bytes |
| **Metadata Corpus** | `outputs/stage-08/maritime_corpus.jsonl` | `ea95122199de9ca831a90b14b35962e33accb05ae83893442a9613d72aefcb32` | 774,173,719 bytes |
| **Stage 07 Clean Input** | `outputs/stage-07/clean_documents.jsonl` | `a29e3d7db30b7e52c2f1d35812cd17e1f4f070da73a5f3665b7399938a73d56c` | 785,395,491 bytes (96,844 docs) |

## 2. Multi-Format Corpus Representations (Stage 11)

All representations derive deterministically from the frozen source via `scripts/11_corpus_representations.py`.

| Representation | Path | SHA-256 Checksum | Docs | Size |
|---|---|---|---|---|
| **json** | `outputs/stage-11/corpus_representations/json.jsonl` | `e0cb3fd7a6ebbcaa4a5df7f7a90554d1c4508974c609083bc50efe3920ad6220` | 96,844 | 759,801,357 bytes |
| **key_value** | `outputs/stage-11/corpus_representations/key_value.jsonl` | `1b0b0e991505f5589024096be5a47cb5873e2cbedba88878ef703c4c86e55f6c` | 96,844 | 65,167,543 bytes |
| **mixed** | `outputs/stage-11/corpus_representations/mixed.jsonl` | `c57b92cd15ef0ca841fbdb88c7b0863445505d65e2469fcf5cbb77b34c9ef272` | 96,844 | 91,828,343 bytes |
| **narrative** | `outputs/stage-11/corpus_representations/narrative.jsonl` | `d9fe116532303ced6174985fa092c351f1232833f5aa6904eb2a484ce0193da7` | 96,844 | 31,485,928 bytes |
| **template** | `outputs/stage-11/corpus_representations/template.jsonl` | `8e7d3312c5fbed9db5f46343db9afe4236383c0fe179e303b78a706f638e0ca2` | 96,844 | 51,747,609 bytes |

## 3. Cryptographic Verification & Invariance Guarantees

- **Zero Lineage Invention**: Lineage records the actual execution chain: `clean_documents.jsonl` (Stage 07) -> `corpus_representations/` (Stage 11) -> `maritime_corpus.*` (Stage 08 export).
- **Unique Serialization**: All 5 representations have distinct SHA-256 digests, verifying format heterogeneity.
- **Ordering Invariance**: All representations preserve identical `occurrence_id` sequence order.
- **Generator Script Digest**: `a6f54957b4ed7eaccc0f873fb85f3cdda2216f2876c2a19fe532f462523568e4` (`scripts/11_corpus_representations.py`).

---
*Automated verification test: `tests/test_representation_provenance.py`.*
