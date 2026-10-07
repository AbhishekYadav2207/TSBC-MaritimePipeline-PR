# MaritimeBench: Representation Provenance & Cryptographic Traceability Manifest

**Generated:** `2026-10-07T16:49:13.757689+00:00` | **Git Commit:** `dda55b0776c47e018e5a3ea7675ba1a9f1b85793` | **Pipeline Version:** `2.1`  
**Target Corpus:** Frozen Corpus A (MARSIS / TSB Canadian Maritime Corpus)  

---

## 1. Frozen Source Corpus Lineage

| Artifact Role | File Path | SHA-256 Checksum | Size |
|---|---|---|---|
| **Text Corpus** | `outputs/stage-08/maritime_corpus.txt` | `c69534d3c4f893da6b93c5d4a9353092bf8e98a6809247b07b86f72f39a47c0f` | 24,553,933 bytes |
| **Metadata Corpus** | `outputs/stage-08/maritime_corpus.jsonl` | `353fbc18aa471c28669f65cdf4c94efc014864b316fa025114bc24d81bee25a0` | 774,365,691 bytes |
| **Stage 07 Clean Input** | `outputs/stage-07/clean_documents.jsonl` | `4e0b642be5e03f8ebb0f9d77fc5aec694f3c241e406d77dc4201894571f332d8` | 785,587,917 bytes (96,848 docs) |

## 2. Multi-Format Corpus Representations (Stage 11)

All representations derive deterministically from the frozen source via `scripts/11_corpus_representations.py`.

| Representation | Path | SHA-256 Checksum | Docs | Size |
|---|---|---|---|---|
| **json** | `outputs/stage-11/corpus_representations/json.jsonl` | `19ddfa5f9f8c9aea5112297b63c3454ebd6a1b6b03b3f73447319356aedd447a` | 96,848 | 759,911,186 bytes |
| **key_value** | `outputs/stage-11/corpus_representations/key_value.jsonl` | `47b79a48420139c7a61722accdb9e1f4692c5984d52ff44f1093454488a26cca` | 96,848 | 65,173,231 bytes |
| **mixed** | `outputs/stage-11/corpus_representations/mixed.jsonl` | `60a4e3989110c297caef81f9dee2a8be830fa0d5f48fa672cf2681faa756c1b3` | 96,848 | 91,838,853 bytes |
| **narrative** | `outputs/stage-11/corpus_representations/narrative.jsonl` | `fdbffb7846637cdb013d48afc98fcdf5e3dae5c9177542cd3c398253ecf73b31` | 96,848 | 31,490,946 bytes |
| **template** | `outputs/stage-11/corpus_representations/template.jsonl` | `ece9aad029adbff1bbc43707e6a36f5ba758338e23603169f8a5a5c8c532b2aa` | 96,848 | 51,749,263 bytes |

## 3. Cryptographic Verification & Invariance Guarantees

- **Zero Lineage Invention**: Lineage records the actual execution chain: `clean_documents.jsonl` (Stage 07) -> `corpus_representations/` (Stage 11) -> `maritime_corpus.*` (Stage 08 export).
- **Unique Serialization**: All 5 representations have distinct SHA-256 digests, verifying format heterogeneity.
- **Ordering Invariance**: All representations preserve identical `occurrence_id` sequence order.
- **Generator Script Digest**: `a6f54957b4ed7eaccc0f873fb85f3cdda2216f2876c2a19fe532f462523568e4` (`scripts/11_corpus_representations.py`).

---
*Automated verification test: `tests/test_representation_provenance.py`.*
