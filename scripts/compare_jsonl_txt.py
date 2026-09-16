import csv
import json
import re
import sys
from pathlib import Path

# Import the EXACT same function used by Stage 08.
# Adjust this import if your project structure differs.
from text_sanitizer import strip_administrative_noise


def normalize_stage08_text(text: str) -> str:
    """
    Reproduce the text transformation performed by Stage 08 exactly.
    """
    text_clean = strip_administrative_noise(text)

    # This assumes the intended Stage-08 regex is r'\n+'
    # If your actual Stage-08 source literally contains a different regex,
    # replace this line with the exact implementation from that file.
    text_clean = re.sub(
        r"\n+",
        "\n",
        text_clean.replace("\r\n", "\n").replace("\r", "\n"),
    )

    return text_clean.strip()


def load_jsonl_documents(jsonl_path: Path):
    """
    Load document fields from clean_documents.jsonl.

    Returns:
        all_records:
            All JSONL records with their original document.
        expected_docs:
            Non-empty documents after applying Stage-08 transformation.
    """
    all_records = []
    expected_docs = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                continue

            record = json.loads(line)

            original = record.get("document", "")
            transformed = normalize_stage08_text(original)

            entry = {
                "line_number": line_number,
                "occurrence_id": record.get("occurrence_id"),
                "original": original,
                "transformed": transformed,
                "original_chars": len(original),
                "transformed_chars": len(transformed),
                "original_words": len(original.split()),
                "transformed_words": len(transformed.split()),
                "empty_after_transform": not bool(transformed),
            }

            all_records.append(entry)

            if transformed:
                expected_docs.append(entry)

    return all_records, expected_docs


def load_txt_documents(txt_path: Path):
    """
    Parse maritime_corpus.txt using the same blank-line document
    separation used by DAPT CorpusLoader.
    """
    documents = []

    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        current_lines = []

        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()

            if stripped:
                current_lines.append(stripped)
            else:
                if current_lines:
                    documents.append({
                        "document_index": len(documents) + 1,
                        "start_line": line_number - len(current_lines),
                        "text": " ".join(current_lines),
                    })
                    current_lines = []

        if current_lines:
            documents.append({
                "document_index": len(documents) + 1,
                "start_line": None,
                "text": " ".join(current_lines),
            })

    return documents

def canonicalize_for_dapt(text: str) -> str:
    """
    Convert text to the canonical representation used by DAPT CorpusLoader.

    DAPT:
        - strips each non-empty line
        - joins lines using one space
    """
    lines = [
        line.strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]

    return " ".join(lines)

def compare_documents(expected_docs, txt_docs, report_csv):
    """
    Compare documents positionally and write detailed CSV results.
    """

    total_expected = len(expected_docs)
    total_txt = len(txt_docs)
    total_compare = min(total_expected, total_txt)

    exact_matches = 0
    mismatches = 0

    transformed_char_total = 0
    txt_char_total = 0
    transformed_word_total = 0
    txt_word_total = 0

    mismatch_rows = []

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "occurrence_id",
                "jsonl_line",
                "txt_start_line",
                "status",
                "jsonl_chars",
                "txt_chars",
                "char_difference",
                "jsonl_words",
                "txt_words",
                "word_difference",
                "jsonl_text",
                "txt_text",
            ],
        )
        writer.writeheader()

        for i in range(total_compare):
            json_entry = expected_docs[i]
            txt_entry = txt_docs[i]

            raw_expected_text = json_entry["transformed"]
            raw_actual_text = txt_entry["text"]

            expected_text = canonicalize_for_dapt(raw_expected_text)
            actual_text = canonicalize_for_dapt(raw_actual_text)

            expected_chars = len(expected_text)
            actual_chars = len(actual_text)

            expected_words = len(expected_text.split())
            actual_words = len(actual_text.split())

            if expected_text == actual_text:
                status = "MATCH"
                exact_matches += 1
            else:
                status = "MISMATCH"
                mismatches += 1

            row = {
                "index": i + 1,
                "occurrence_id": json_entry["occurrence_id"],
                "jsonl_line": json_entry["line_number"],
                "txt_start_line": txt_entry["start_line"],
                "status": status,
                "jsonl_chars": expected_chars,
                "txt_chars": actual_chars,
                "char_difference": actual_chars - expected_chars,
                "jsonl_words": expected_words,
                "txt_words": actual_words,
                "word_difference": actual_words - expected_words,
                "jsonl_text": expected_text,
                "txt_text": actual_text,
            }

            # Writing every document to CSV can make the file enormous.
            # We still write every record because this is intended as
            # a forensic comparison.
            writer.writerow(row)

            if status == "MISMATCH" and len(mismatch_rows) < 20:
                mismatch_rows.append(row)

    return {
        "total_expected": total_expected,
        "total_txt": total_txt,
        "total_compared": total_compare,
        "exact_matches": exact_matches,
        "mismatches": mismatches,
        "missing_in_txt": max(0, total_expected - total_txt),
        "extra_in_txt": max(0, total_txt - total_expected),
        "expected_chars": transformed_char_total,
        "txt_chars": txt_char_total,
        "character_difference": txt_char_total - transformed_char_total,
        "expected_words": transformed_word_total,
        "txt_words": txt_word_total,
        "word_difference": txt_word_total - transformed_word_total,
        "sample_mismatches": mismatch_rows,
    }


def main():
    if len(sys.argv) < 3:
        print(
            "\nUsage:\n"
            "  python compare_jsonl_txt.py "
            "<clean_documents.jsonl> <maritime_corpus.txt> "
            "[output_directory]\n"
        )
        sys.exit(1)

    jsonl_path = Path(sys.argv[1]).resolve()
    txt_path = Path(sys.argv[2]).resolve()

    output_dir = (
        Path(sys.argv[3]).resolve()
        if len(sys.argv) >= 4
        else Path("corpus_comparison_report").resolve()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_csv = output_dir / "document_comparison.csv"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    if not txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    print("=" * 80)
    print("MARITIME CORPUS JSONL ↔ TXT COMPARISON")
    print("=" * 80)

    print(f"\nJSONL: {jsonl_path}")
    print(f"TXT  : {txt_path}")

    # ------------------------------------------------------------------
    # Load source JSONL
    # ------------------------------------------------------------------
    print("\n[1/3] Reading clean_documents.jsonl...")
    all_records, expected_docs = load_jsonl_documents(jsonl_path)

    print(f"Total JSONL records          : {len(all_records):,}")
    print(
        f"Empty after Stage-08 transform: "
        f"{sum(x['empty_after_transform'] for x in all_records):,}"
    )
    print(f"Expected TXT documents       : {len(expected_docs):,}")

    # ------------------------------------------------------------------
    # Load TXT
    # ------------------------------------------------------------------
    print("\n[2/3] Reading maritime_corpus.txt...")
    txt_docs = load_txt_documents(txt_path)

    print(f"Actual TXT documents         : {len(txt_docs):,}")

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("\n[3/3] Comparing documents record-by-record...")
    result = compare_documents(expected_docs, txt_docs, report_csv)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"JSONL records                : {len(all_records):,}")
    print(f"Expected TXT documents       : {result['total_expected']:,}")
    print(f"Actual TXT documents         : {result['total_txt']:,}")
    print(f"Documents compared           : {result['total_compared']:,}")
    print(f"Exact matches                : {result['exact_matches']:,}")
    print(f"Mismatches                   : {result['mismatches']:,}")
    print(f"Missing in TXT               : {result['missing_in_txt']:,}")
    print(f"Extra in TXT                 : {result['extra_in_txt']:,}")

    print("\nCharacter comparison")
    print("-" * 40)
    print(f"Expected characters          : {result['expected_chars']:,}")
    print(f"TXT characters               : {result['txt_chars']:,}")
    print(f"Difference                   : {result['character_difference']:+,}")

    print("\nWord comparison")
    print("-" * 40)
    print(f"Expected whitespace words    : {result['expected_words']:,}")
    print(f"TXT whitespace words         : {result['txt_words']:,}")
    print(f"Difference                   : {result['word_difference']:+,}")

    print("\nDetailed report")
    print("-" * 40)
    print(f"CSV report                   : {report_csv}")

    if result["sample_mismatches"]:
        print("\n" + "=" * 80)
        print("FIRST 20 MISMATCHES")
        print("=" * 80)

        for row in result["sample_mismatches"]:
            print(f"\nDocument #{row['index']}")
            print(f"Occurrence ID : {row['occurrence_id']}")
            print(f"JSONL line    : {row['jsonl_line']}")
            print(f"TXT line      : {row['txt_start_line']}")
            print(
                f"Characters    : "
                f"{row['jsonl_chars']} → {row['txt_chars']} "
                f"({row['char_difference']:+d})"
            )
            print(
                f"Words         : "
                f"{row['jsonl_words']} → {row['txt_words']} "
                f"({row['word_difference']:+d})"
            )

            print("\nEXPECTED (Stage-08 transformed JSONL document):")
            print("-" * 60)
            print(row["jsonl_text"])

            print("\nACTUAL (TXT document):")
            print("-" * 60)
            print(row["txt_text"])

    if (
        result["total_expected"] == result["total_txt"]
        and result["mismatches"] == 0
    ):
        print("\n✅ PERFECT MATCH")
        print(
            "Every non-empty JSONL document matches the corresponding "
            "TXT document exactly."
        )
    else:
        print("\n⚠️ DIFFERENCES FOUND")
        print(
            "The JSONL-derived corpus and TXT corpus are not identical. "
            "Use document_comparison.csv to inspect every record."
        )


if __name__ == "__main__":
    main()