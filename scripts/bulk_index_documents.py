#!/usr/bin/env python3
"""
Bulk document indexing script for the Multi-modal Enterprise KM Agent.
This script processes multiple do except (OSError, ValueError, RuntimeError) as e:
                logger.error("✗ Error indexing %s: %s", doc_path, e)
                total_failed += 1

    # Summary
    logger.info("=" * 50)
    logger.info("BULK INDEXING COMPLETE")
    logger.info("Total documents found: %d", len(documents))
    logger.info("Successfully processed: %d", total_processed)
    logger.info("Failed to process: %d", total_failed)
    logger.info("=" * 50)

    if total_failed > 0:
        logger.warning("Some documents failed to index. Check the log for details.")
        sys.exit(1)directory and indexes them into the vector database.

Usage:
    python scripts/bulk_index_documents.py /path/to/documents \
        --recursive --batch-size 10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.rag.components import get_indexing_pipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bulk_indexing.log"),
        ],
    )


def find_documents(
    directory: Path, recursive: bool = True, extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Find all documents in the given directory.

    Args:
        directory: Directory to search in
        recursive: Whether to search recursively
        extensions: List of file extensions to include (default: pdf, txt, md)

    Returns:
        List of document paths
    """
    if extensions is None:
        extensions = [".pdf", ".txt", ".md", ".docx", ".html"]

    pattern = "**/*" if recursive else "*"
    documents: List[Path] = []

    for ext in extensions:
        documents.extend(directory.glob(f"{pattern}{ext}"))

    return sorted(documents)


def batch_process_documents(documents: List[Path], batch_size: int = 5):
    """Process documents in batches."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        yield batch


def main():
    parser = argparse.ArgumentParser(
        description="Bulk index documents into the KM Agent"
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing documents to index"
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search recursively in subdirectories",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5,
        help="Number of documents to process per batch",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        nargs="+",
        default=[".pdf", ".txt", ".md"],
        help="File extensions to include (default: .pdf .txt .md)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually indexing",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        logger.error("Directory does not exist: %s", directory)
        sys.exit(1)

    if not directory.is_dir():
        logger.error("Path is not a directory: %s", directory)
        sys.exit(1)

    logger.info("Searching for documents in: %s", directory)
    logger.info("Recursive search: %s", args.recursive)
    logger.info("File extensions: %s", args.extensions)

    # Find documents
    documents = find_documents(directory, args.recursive, args.extensions)
    logger.info("Found %d documents to process", len(documents))

    if not documents:
        logger.warning("No documents found. Exiting.")
        return

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - Documents that would be processed:")
        for i, doc in enumerate(documents, 1):
            logger.info("  %3d. %s", i, doc)
        logger.info("Total: %d documents", len(documents))
        return

    # Initialize indexing pipeline
    logger.info("Initializing indexing pipeline...")
    try:
        indexing_pipeline = get_indexing_pipeline()
    except (ImportError, RuntimeError, ConnectionError) as e:
        logger.error("Failed to initialize indexing pipeline: %s", e)
        sys.exit(1)

    # Process documents in batches
    total_processed = 0
    total_failed = 0

    for batch_num, batch in enumerate(
        batch_process_documents(documents, args.batch_size), 1
    ):
        logger.info("Processing batch %d (%d documents)", batch_num, len(batch))

        for doc_path in batch:
            try:
                logger.info("Indexing: %s", doc_path)
                indexing_pipeline.index_document(str(doc_path))
                logger.info("✓ Successfully indexed: %s", doc_path.name)
                total_processed += 1

            except Exception as e:
                logger.error("✗ Error indexing %s: %s", doc_path, e)
                total_failed += 1

    # Summary
    logger.info("=" * 50)
    logger.info("BULK INDEXING COMPLETE")
    logger.info("Total documents found: %d", len(documents))
    logger.info("Successfully processed: %d", total_processed)
    logger.info("Failed to process: %d", total_failed)
    logger.info("=" * 50)

    if total_failed > 0:
        logger.warning("Some documents failed to index. Check the log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
