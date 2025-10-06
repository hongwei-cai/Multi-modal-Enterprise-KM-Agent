#!/usr/bin/env python3
"""
Interactive RAG Pipeline Testing Script.
This script allows you to test the RAG pipeline with custom queries and documents.

Usage:
    python scripts/test_rag_pipeline.py
    # Then enter queries interactively

Or for batch testing:
    python scripts/test_rag_pipeline.py --queries "What is AI?" \
        "How does RAG work?" --output results.txt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.rag.indexing_pipeline import get_indexing_pipeline
from src.rag.rag_pipeline import get_rag_pipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("rag_test.log"),
        ],
    )


def load_documents_from_directory(directory: Path) -> List[str]:
    """Load all document paths from a directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    documents: List[Path] = []
    for ext in [".pdf", ".txt", ".md", ".docx"]:
        documents.extend(directory.glob(f"**/*{ext}"))

    return [str(doc) for doc in documents]


def interactive_mode(rag_pipeline, indexing_pipeline):
    """Run interactive testing mode."""
    print("=" * 60)
    print("RAG Pipeline Interactive Testing")
    print("=" * 60)
    print("Commands:")
    print("  /index <path>    - Index documents from directory")
    print("  /list            - List indexed documents (not implemented)")
    print("  /clear           - Clear index (not implemented)")
    print("  /quit            - Exit")
    print("  <query>          - Ask a question")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nQuery> ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                # Handle commands
                parts = user_input.split()
                command = parts[0].lower()

                if command == "/quit":
                    print("Goodbye!")
                    break
                elif command == "/index" and len(parts) > 1:
                    doc_path = parts[1]
                    try:
                        if Path(doc_path).is_dir():
                            documents = load_documents_from_directory(Path(doc_path))
                            print(f"Found {len(documents)} documents to index")
                            for doc in documents[:5]:  # Show first 5
                                indexing_pipeline.index_document(doc)
                            if len(documents) > 5:
                                print(f"... and {len(documents) - 5} more")
                        else:
                            indexing_pipeline.index_document(doc_path)
                        print("Documents indexed successfully!")
                    except (OSError, ValueError, RuntimeError) as e:
                        print(f"Error indexing documents: {e}")
                elif command == "/list":
                    print("Document listing not implemented yet")
                elif command == "/clear":
                    print("Index clearing not implemented yet")
                else:
                    print(f"Unknown command: {command}")
            else:
                # Handle query
                print(f"Query: {user_input}")
                print("Thinking...")

                try:
                    response = rag_pipeline.answer_question(user_input)
                    print("\nResponse:")
                    print("-" * 40)

                    if isinstance(response, dict):
                        answer = response.get("answer", "No answer provided")
                        context = response.get("context", [])
                        print(answer)

                        if context:
                            print(f"\nRetrieved {len(context)} document chunks")
                            print("Context preview:")
                            for i, ctx in enumerate(
                                context[:2]
                            ):  # Show first 2 contexts
                                preview = (
                                    ctx.get("text", "")[:200] + "..."
                                    if len(ctx.get("text", "")) > 200
                                    else ctx.get("text", "")
                                )
                                print(f"  [{i+1}] {preview}")
                    else:
                        print(str(response))

                except (RuntimeError, ValueError, TypeError) as e:
                    print(f"Error processing query: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def batch_mode(
    rag_pipeline, indexing_pipeline, queries: List[str], output_file: Optional[str]
):
    """Run batch testing mode."""
    logger = logging.getLogger(__name__)

    results = []

    for i, query in enumerate(queries, 1):
        logger.info("Processing query %d/%d: %s", i, len(queries), query)

        try:
            response = rag_pipeline.answer_question(query)

            if isinstance(response, dict):
                answer = response.get("answer", "No answer provided")
                context_count = len(response.get("context", []))
            else:
                answer = str(response)
                context_count = 0

            result = {
                "query": query,
                "answer": answer,
                "context_count": context_count,
                "success": True,
            }

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Error processing query '%s': %s", query, e)
            result = {
                "query": query,
                "answer": f"Error: {e}",
                "context_count": 0,
                "success": False,
            }

        results.append(result)

    # Print results
    print("\n" + "=" * 80)
    print("BATCH TEST RESULTS")
    print("=" * 80)

    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"{status} Query: {result['query']}")
        print(
            f"   Answer: {result['answer'][:100]}\
                {'...' if len(result['answer']) > 100 else ''}"
        )
        print(f"   Context chunks: {result['context_count']}")
        print("-" * 80)

    success_count = sum(1 for r in results if r["success"])
    print(f"\nSummary: {success_count}/{len(results)} queries processed successfully")

    # Save to file if requested
    if output_file:
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG Pipeline interactively or in batch mode"
    )
    parser.add_argument(
        "--queries", "-q", nargs="+", help="Queries to test in batch mode"
    )
    parser.add_argument(
        "--documents", "-d", type=str, help="Directory with documents to index first"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for batch results (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = get_rag_pipeline()
        indexing_pipeline = get_indexing_pipeline()

        # Index documents if specified
        if args.documents:
            logger.info("Indexing documents from: %s", args.documents)
            try:
                documents = load_documents_from_directory(Path(args.documents))
                logger.info("Found %d documents", len(documents))

                for doc_path in documents:
                    try:
                        indexing_pipeline.index_document(doc_path)
                        logger.debug("Indexed: %s", doc_path)
                    except (OSError, ValueError, RuntimeError) as e:
                        logger.error("Failed to index %s: %s", doc_path, e)

                logger.info("Document indexing complete")
            except (OSError, ValueError) as e:
                logger.error("Error loading documents: %s", e)
                sys.exit(1)

        # Run in appropriate mode
        if args.queries:
            # Batch mode
            batch_mode(rag_pipeline, indexing_pipeline, args.queries, args.output)
        else:
            # Interactive mode
            interactive_mode(rag_pipeline, indexing_pipeline)

    except (RuntimeError, ImportError, ConnectionError) as e:
        logger.error("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
