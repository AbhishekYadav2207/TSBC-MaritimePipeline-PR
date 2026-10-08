# MaritimeBench: Representation Provenance & Cryptographic Traceability Manifest

**Generated:** `2026-10-08T08:29:33.611227+00:00` | **Git Commit:** `038925fd89336fe2029e5efbff27163de3878b7b` | **Pipeline Version:** `2.1`  
**Target Corpus:** Frozen Corpus A (MARSIS / TSB Canadian Maritime Corpus)  

---

## 1. Frozen Source Corpus Lineage

| Artifact Role | File Path | SHA-256 Checksum | Size |
|---|---|---|---|
| **Text Corpus** | `outputs/stage-08/maritime_corpus.txt` | `bf3ec45892c194e98310a74ae870851d55df50ba37dfb968cc731e525c6edb07` | 24,551,773 bytes |
| **Metadata Corpus** | `outputs/stage-08/maritime_corpus.jsonl` | `83635ab0d85d34d82d760bca351ff88facf77d1562d23166b010173623a39c76` | 774,155,686 bytes |
| **Stage 07 Clean Input** | `outputs/stage-07/clean_documents.jsonl` | `a3bc0299b0745c24f5379a656e60e9ce3863dda2082612c69dbcee9124c3d468` | 785,378,292 bytes (96,850 docs) |

## 2. Multi-Format Corpus Representations (Stage 11)

All representations derive deterministically from the frozen source via `scripts/11_corpus_representations.py`.

| Representation | Path | SHA-256 Checksum | Docs | Size |
|---|---|---|---|---|
| **json** | `outputs/stage-11/corpus_representations/json.jsonl` | `5e2631bab5181bb71ef59672b9c294fc8a01cbd9b382f5d1af1e61b41fd597ae` | 96,850 | 759,724,331 bytes |
| **key_value** | `outputs/stage-11/corpus_representations/key_value.jsonl` | `13157e455bfa684b79467bd91b1bdbba91c08d7d3f6419e5ef352b0cc54c38ea` | 96,850 | 65,077,206 bytes |
| **mixed** | `outputs/stage-11/corpus_representations/mixed.jsonl` | `11fa4cfed02094d14325e2abdb59ee3198bfe33bf1f1ff6741c2320816d35012` | 96,850 | 91,740,730 bytes |
| **narrative** | `outputs/stage-11/corpus_representations/narrative.jsonl` | `e32b1e4870877338f94a543db8fde99b26f7abe02154bd22e11ed274aa4d0eef` | 96,850 | 31,392,111 bytes |
| **template** | `outputs/stage-11/corpus_representations/template.jsonl` | `3fba5483db6c4d32d8df6c7692342998f522e14f64ce22f139497da58ceeaad2` | 96,850 | 51,656,361 bytes |

## 3. Cryptographic Verification & Invariance Guarantees

- **Zero Lineage Invention**: Lineage records the actual execution chain: `clean_documents.jsonl` (Stage 07) -> `corpus_representations/` (Stage 11) -> `maritime_corpus.*` (Stage 08 export).
- **Unique Serialization**: All 5 representations have distinct SHA-256 digests, verifying format heterogeneity.
- **Ordering Invariance**: All representations preserve identical `occurrence_id` sequence order.
- **Generator Script Digest**: `a6f54957b4ed7eaccc0f873fb85f3cdda2216f2876c2a19fe532f462523568e4` (`scripts/11_corpus_representations.py`).

---
*Automated verification test: `tests/test_representation_provenance.py`.*
