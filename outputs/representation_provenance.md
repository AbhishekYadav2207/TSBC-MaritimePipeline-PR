# MaritimeBench: Representation Provenance & Cryptographic Traceability Manifest

**Generated:** `2026-10-07T19:56:26.621531+00:00` | **Git Commit:** `919a3ffa06872764ee6a0e9d4713144da5a5287e` | **Pipeline Version:** `2.1`  
**Target Corpus:** Frozen Corpus A (MARSIS / TSB Canadian Maritime Corpus)  

---

## 1. Frozen Source Corpus Lineage

| Artifact Role | File Path | SHA-256 Checksum | Size |
|---|---|---|---|
| **Text Corpus** | `outputs/stage-08/maritime_corpus.txt` | `602a848485750b465f560b1ca68ba29de60b4adbb28b3a5c5d9a1ebaf270eaec` | 24,556,413 bytes |
| **Metadata Corpus** | `outputs/stage-08/maritime_corpus.jsonl` | `3a2031f741f7a99cbff73f1d8993eee894e232d5271f292f45bbe3c1b767bc5c` | 774,410,772 bytes |
| **Stage 07 Clean Input** | `outputs/stage-07/clean_documents.jsonl` | `4f05da2fbdac924f5d4171b2329f8d41cd87b40eea8691a73efba46ca5931ecd` | 785,636,298 bytes (96,874 docs) |

## 2. Multi-Format Corpus Representations (Stage 11)

All representations derive deterministically from the frozen source via `scripts/11_corpus_representations.py`.

| Representation | Path | SHA-256 Checksum | Docs | Size |
|---|---|---|---|---|
| **json** | `outputs/stage-11/corpus_representations/json.jsonl` | `14122f5bd95b401346f84ed5c724c65caf4353524ed1eedf55dd84d13b4b5028` | 96,874 | 759,980,595 bytes |
| **key_value** | `outputs/stage-11/corpus_representations/key_value.jsonl` | `0bde74c8bf4b9db52f455593577756fd09860f0f4a1f39ce51eda823a583500b` | 96,874 | 65,092,615 bytes |
| **mixed** | `outputs/stage-11/corpus_representations/mixed.jsonl` | `11643d5d7118da3b4fceb471204ce87d8a8559b9e922f2592c0953cc4b6e4c28` | 96,874 | 91,761,238 bytes |
| **narrative** | `outputs/stage-11/corpus_representations/narrative.jsonl` | `b63a6be92121a7e1ee0150b609ae00fc7f39c80e83ac8a0bf0520ccff67f6b9f` | 96,874 | 31,398,377 bytes |
| **template** | `outputs/stage-11/corpus_representations/template.jsonl` | `aef2c13fb338b4dae41b7d4b2929b4bdc22c0e0ad780bf4cbc33c1644e3e0ab0` | 96,874 | 51,667,477 bytes |

## 3. Cryptographic Verification & Invariance Guarantees

- **Zero Lineage Invention**: Lineage records the actual execution chain: `clean_documents.jsonl` (Stage 07) -> `corpus_representations/` (Stage 11) -> `maritime_corpus.*` (Stage 08 export).
- **Unique Serialization**: All 5 representations have distinct SHA-256 digests, verifying format heterogeneity.
- **Ordering Invariance**: All representations preserve identical `occurrence_id` sequence order.
- **Generator Script Digest**: `a6f54957b4ed7eaccc0f873fb85f3cdda2216f2876c2a19fe532f462523568e4` (`scripts/11_corpus_representations.py`).

---
*Automated verification test: `tests/test_representation_provenance.py`.*
