# MaritimeBench: Representation Provenance & Cryptographic Traceability Manifest

**Generated:** `2026-10-07T08:10:47.979234+00:00` | **Git Commit:** `e36464b67e8971752073835f4a31075fc6282957` | **Pipeline Version:** `2.1`  
**Target Corpus:** Frozen Corpus A (MARSIS / TSB Canadian Maritime Corpus)  

---

## 1. Frozen Source Corpus Lineage

| Artifact Role | File Path | SHA-256 Checksum | Size |
|---|---|---|---|
| **Text Corpus** | `outputs/stage-08/maritime_corpus.txt` | `6a4d69587fbe05fc76fc26843b2af48d960c62f7807386361ea95eae03d470ff` | 24,552,501 bytes |
| **Metadata Corpus** | `outputs/stage-08/maritime_corpus.jsonl` | `19013423d5fd7b160570e79381af8c269423856920f64d14c474642085965fb3` | 774,216,739 bytes |
| **Stage 07 Clean Input** | `outputs/stage-07/clean_documents.jsonl` | `aaf0c6c3dffefff9f62c6ddf3fa2813f0554d88e6ddf29b1d0ec1e7d50d1c253` | 785,440,601 bytes (96,860 docs) |

## 2. Multi-Format Corpus Representations (Stage 11)

All representations derive deterministically from the frozen source via `scripts/11_corpus_representations.py`.

| Representation | Path | SHA-256 Checksum | Docs | Size |
|---|---|---|---|---|
| **json** | `outputs/stage-11/corpus_representations/json.jsonl` | `ec36551206d62b3b2eb24c6d0eeb6bfac33c1425521f568183bfa1a06c7d6ed0` | 96,860 | 759,785,016 bytes |
| **key_value** | `outputs/stage-11/corpus_representations/key_value.jsonl` | `ef950b87c3242cd0021ecee2ac973576aefb0ef25a9763e16ac2baed41adde37` | 96,860 | 65,083,031 bytes |
| **mixed** | `outputs/stage-11/corpus_representations/mixed.jsonl` | `6299d094966104588ddc3410f4ea1f0ef74fb25eb590ce8798b20826d6f418fc` | 96,860 | 91,747,491 bytes |
| **narrative** | `outputs/stage-11/corpus_representations/narrative.jsonl` | `a72abfe37bc5157cc723d4e5dbf04bbcbac2527a9066e5b862d5bea50a08b310` | 96,860 | 31,393,532 bytes |
| **template** | `outputs/stage-11/corpus_representations/template.jsonl` | `938492c53a867793939686489d8a27b26ed0885a7a337e572e74004281c0444e` | 96,860 | 51,662,229 bytes |

## 3. Cryptographic Verification & Invariance Guarantees

- **Zero Lineage Invention**: Lineage records the actual execution chain: `clean_documents.jsonl` (Stage 07) -> `corpus_representations/` (Stage 11) -> `maritime_corpus.*` (Stage 08 export).
- **Unique Serialization**: All 5 representations have distinct SHA-256 digests, verifying format heterogeneity.
- **Ordering Invariance**: All representations preserve identical `occurrence_id` sequence order.
- **Generator Script Digest**: `a6f54957b4ed7eaccc0f873fb85f3cdda2216f2876c2a19fe532f462523568e4` (`scripts/11_corpus_representations.py`).

---
*Automated verification test: `tests/test_representation_provenance.py`.*
