#!/usr/bin/env python3
"""
Index Knowledge Base QA Pairs

Indexes the organized QA pairs from the knowledge base into ChromaDB
for use in RAG retrieval.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.rag.components.knowledge_base_indexer import index_knowledge_base

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Index QA pairs from knowledge base into ChromaDB"
    )

    parser.add_argument(
        "--qa_pairs_file",
        type=str,
        default="data/processed/kb_v1_c627d49b/train.jsonl",
        help="Path to QA pairs JSONL file",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="question_only",
        choices=["question_only", "answer_only", "qa_combined", "qa_separate"],
        help="Indexing strategy for QA pairs",
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default="knowledge_base",
        help="ChromaDB collection name",
    )

    parser.add_argument(
        "--version",
        type=str,
        help="Dataset version (auto-detected from path if not provided)",
    )

    parser.add_argument(
        "--db_path",
        type=str,
        help="ChromaDB persistence directory (uses default if not specified)",
    )

    args = parser.parse_args()

    # Auto-detect version from path if not provided
    if args.version is None and "kb_" in args.qa_pairs_file:
        # Extract version from path like "kb_v1_c627d49b"
        path_parts = Path(args.qa_pairs_file).parts
        for part in path_parts:
            if part.startswith("kb_v"):
                args.version = part
                break

    # Check input file exists
    qa_pairs_path = Path(args.qa_pairs_file)
    if not qa_pairs_path.exists():
        logger.error("QA pairs file does not exist: %s", qa_pairs_path)
        sys.exit(1)

    try:
        logger.info("Starting knowledge base indexing...")
        logger.info("  File: %s", args.qa_pairs_file)
        logger.info("  Strategy: %s", args.strategy)
        logger.info("  Collection: %s", args.collection_name)
        logger.info("  Version: %s", args.version)

        # Index the knowledge base
        stats = index_knowledge_base(
            qa_pairs_file=str(qa_pairs_path),
            strategy=args.strategy,
            collection_name=args.collection_name,
            db_path=args.db_path,
            version=args.version,
        )

        # Print results
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE INDEXING COMPLETED")
        print("=" * 60)

        print(f"Dataset Version: {stats['version']}")
        print(f"Total QA Pairs: {stats['total_qa_pairs']}")
        print(f"Indexed Items: {stats['total_indexed_items']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Strategy: {stats['strategy']}")

        print("\nStatistics:")
        print(f"  Average Question Length: {stats['avg_question_length']:.1f} words")
        print(f"  Average Answer Length: {stats['avg_answer_length']:.1f} words")

        print("\nChromaDB Status:")
        print(f"  Collection: {args.collection_name}")
        print(f"  Items indexed: {stats['total_indexed_items']}")

        print("\n" + "=" * 60)

        # Save indexing results
        results_file = Path(args.qa_pairs_file).parent / "indexing_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info("Indexing results saved to: %s", results_file)

    except Exception as e:
        logger.error("Knowledge base indexing failed: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
