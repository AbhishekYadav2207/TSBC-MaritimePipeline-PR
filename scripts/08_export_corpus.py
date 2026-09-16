import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from pipeline_utils import setup_logging, load_config, get_project_root
from text_sanitizer import strip_administrative_noise


logger = setup_logging("08_export_corpus")


def get_git_commit() -> str:
    """
    Gets the current git commit hash if available,
    otherwise returns 'unknown'.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown"


def normalize_document_for_txt(text: str) -> str:
    """
    Converts one generated document into the canonical plain-text
    representation used by the DAPT corpus.

    Rules:
    1. Remove administrative noise.
    2. Normalize line endings.
    3. Remove trailing whitespace from lines.
    4. Collapse blank/whitespace-only lines inside a document.
    5. Preserve single newlines between lines/paragraphs.
    6. Strip leading/trailing whitespace.
    """
    if not text:
        return ""

    # 1. Remove administrative noise.
    cleaned = strip_administrative_noise(text)

    # 2. Normalize line endings.
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove trailing spaces/tabs before newlines.
    #
    #    Example:
    #       "The occurrence \n"
    #    becomes:
    #       "The occurrence\n"
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

    # 4. Collapse blank/whitespace-only lines.
    #
    #    This handles:
    #       \n\n
    #       \n \n
    #       \n    \n
    #
    #    and converts all of them to a single newline.
    cleaned = re.sub(r"\n[ \t]*\n+", "\n", cleaned)

    # 5. Strip outer whitespace.
    cleaned = cleaned.strip()

    return cleaned


def main():
    root = get_project_root()
    config = load_config()

    output_dir = root / config.get("output_dir", "outputs")

    clean_path = output_dir / "stage-07" / "clean_documents.jsonl"

    if not clean_path.exists():
        logger.error(
            f"Clean documents not found at {clean_path}! "
            f"Run Step 7 first."
        )
        return

    stage_dir = output_dir / "stage-08"
    stage_dir.mkdir(parents=True, exist_ok=True)

    corpus_txt_path = stage_dir / "maritime_corpus.txt"
    corpus_jsonl_path = stage_dir / "maritime_corpus.jsonl"
    manifest_path = stage_dir / "manifest.json"

    logger.info("Exporting clean documents to final corpus files...")
    logger.info(f"Source JSONL : {clean_path}")
    logger.info(f"Text corpus  : {corpus_txt_path}")
    logger.info(f"Metadata JSONL: {corpus_jsonl_path}")

    source_records = 0
    exported_docs = 0
    skipped_empty = 0
    documents_with_internal_newlines = 0

    with (
        open(clean_path, "r", encoding="utf-8") as fin,
        open(corpus_txt_path, "w", encoding="utf-8", newline="\n") as ftxt,
        open(corpus_jsonl_path, "w", encoding="utf-8") as fjsonl,
    ):
        for line_number, line in enumerate(
            tqdm(fin, desc="Exporting Corpus"),
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            source_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error(
                    f"Invalid JSON on line {line_number}: {exc}"
                )
                raise

            doc_text = record.get("document", "")
            occurrence_id = record.get("occurrence_id")
            structured = record.get("structured")

            # Track whether the source document contained blank-line
            # boundaries before normalization.
            normalized_source = (
                doc_text.replace("\r\n", "\n").replace("\r", "\n")
                if doc_text
                else ""
            )

            if re.search(r"\n\s*\n", normalized_source):
                documents_with_internal_newlines += 1

            # ----------------------------------------------------------
            # 1. Export canonical plain-text corpus
            # ----------------------------------------------------------
            doc_text_clean = normalize_document_for_txt(doc_text)

            if not doc_text_clean:
                skipped_empty += 1
                continue

            # Exactly ONE blank line separates documents.
            ftxt.write(doc_text_clean)
            ftxt.write("\n\n")

            # ----------------------------------------------------------
            # 2. Export metadata-preserving JSONL
            # ----------------------------------------------------------
            output_obj = {
                "occurrence_id": occurrence_id,
                "document": doc_text,
                "provenance": record.get("provenance"),
                "structured": structured,
            }

            fjsonl.write(
                json.dumps(
                    output_obj,
                    ensure_ascii=False,
                )
                + "\n"
            )

            exported_docs += 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if source_records != exported_docs + skipped_empty:
        logger.error(
            "Export accounting mismatch: "
            f"source_records={source_records}, "
            f"exported_docs={exported_docs}, "
            f"skipped_empty={skipped_empty}"
        )
        raise RuntimeError(
            "Stage-08 export accounting validation failed."
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPORT VALIDATION")
    logger.info("=" * 80)

    logger.info(
        f"Source JSONL records             : {source_records:,}"
    )
    logger.info(
        f"Exported TXT/JSONL documents    : {exported_docs:,}"
    )
    logger.info(
        f"Skipped empty documents          : {skipped_empty:,}"
    )
    logger.info(
        "Source documents containing "
        f"internal blank lines            : "
        f"{documents_with_internal_newlines:,}"
    )

    # For your corpus, the expected condition is:
    #
    # source_records == exported_docs
    #
    # because your comparison already showed that no records are empty
    # after transformation.
    if skipped_empty > 0:
        logger.warning(
            f"{skipped_empty:,} source document(s) were skipped "
            "because they became empty after sanitization."
        )

    # ------------------------------------------------------------------
    # Create manifest
    # ------------------------------------------------------------------

    manifest = {
        "version": "1.1",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "source": "MARSIS",
        "language": "English",
        "pipeline_version": "1.1",
        "git_commit": get_git_commit(),
        "source_jsonl": str(clean_path.resolve()),
        "text_corpus": str(corpus_txt_path.resolve()),
        "metadata_jsonl": str(corpus_jsonl_path.resolve()),
        "source_records": source_records,
        "exported_documents": exported_docs,
        "skipped_empty_documents": skipped_empty,
        "source_documents_with_internal_blank_lines": (
            documents_with_internal_newlines
        ),
        "document_separator": "double_newline",
        "internal_newline_policy": "collapse_consecutive_newlines",
    }

    with open(manifest_path, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Final log
    # ------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("CORPUS EXPORTED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(
        f"Final text file       : {corpus_txt_path}"
    )
    logger.info(
        f"Final metadata JSONL  : {corpus_jsonl_path}"
    )
    logger.info(
        f"Manifest              : {manifest_path}"
    )
    logger.info(
        f"Total source records  : {source_records:,}"
    )
    logger.info(
        f"Total exported docs   : {exported_docs:,}"
    )
    logger.info(
        f"Skipped empty docs    : {skipped_empty:,}"
    )

    if source_records == exported_docs:
        logger.info(
            "Document-count validation: PASS"
        )
    else:
        logger.warning(
            "Document-count validation: WARNING "
            f"({source_records:,} source vs "
            f"{exported_docs:,} exported)"
        )


if __name__ == "__main__":
    main()