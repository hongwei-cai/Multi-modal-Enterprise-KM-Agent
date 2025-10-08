#!/usr/bin/env python3
"""
Performance benchmarking script for the Multi-modal Enterprise KM Agent.
This script runs comprehensive benchmarks on the RAG pipeline to measure speed,\
    accuracy, and resource usage.

Usage:
    python scripts/benchmark_rag_performance.py --queries 50 --documents 10\
        --output results.json
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional

from src.rag.components import get_indexing_pipeline, get_rag_pipeline
from src.rag.experiment_tracker import (
    ExperimentConfig,
    MLflowExperimentTracker,
    PerformanceMetrics,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    query: str
    response_time: float
    response_length: int
    retrieved_docs: int
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics from benchmark runs."""

    total_queries: int
    avg_response_time: float
    median_response_time: float
    min_response_time: float
    max_response_time: float
    std_response_time: float
    avg_response_length: int
    avg_retrieved_docs: float
    total_memory_usage_mb: Optional[float] = None
    avg_cpu_usage_percent: Optional[float] = None
    timestamp: str = ""


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("benchmark.log"),
        ],
    )


def get_sample_queries() -> List[str]:
    """Get a list of sample queries for benchmarking."""
    return [
        "What is machine learning?",
        "How does natural language processing work?",
        "What are the benefits of using RAG systems?",
        "Explain the concept of embeddings in AI",
        "How do vector databases work?",
        "What is the difference between supervised and unsupervised learning?",
        "How can I improve document retrieval accuracy?",
        "What are the main components of a RAG pipeline?",
        "How does chunking affect retrieval performance?",
        "What metrics should I use to evaluate a QA system?",
        "How do I handle long documents in RAG systems?",
        "What are the advantages of using transformers for NLP?",
        "How can I reduce hallucinations in LLM responses?",
        "What is the role of prompt engineering in RAG?",
        "How do I optimize vector search performance?",
    ]


def get_sample_documents() -> List[str]:
    """Get paths to sample documents for indexing."""
    docs_dir = Path("tests/data/pdfs")
    if not docs_dir.exists():
        return []

    return [str(f) for f in docs_dir.glob("*.pdf") if f.is_file()]


def benchmark_single_query(rag_pipeline, query: str, tracker) -> BenchmarkResult:
    """Run benchmark for a single query."""
    start_time = time.time()

    # Track system resources before query
    tracker.log_system_resources("pre_query")

    # Execute query
    try:
        response = rag_pipeline.answer_question(query)
        response_text = (
            response.get("answer", "") if isinstance(response, dict) else str(response)
        )
        retrieved_docs = (
            len(response.get("context", [])) if isinstance(response, dict) else 0
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logging.error("Error during query execution: %s", e)
        response_text = ""
        retrieved_docs = 0

    response_time = time.time() - start_time

    # Track system resources after query
    tracker.log_system_resources("post_query")

    return BenchmarkResult(
        query=query,
        response_time=response_time,
        response_length=len(response_text),
        retrieved_docs=retrieved_docs,
    )


def calculate_summary(results: List[BenchmarkResult]) -> BenchmarkSummary:
    """Calculate summary statistics from benchmark results."""
    if not results:
        return BenchmarkSummary(
            total_queries=0,
            avg_response_time=0.0,
            median_response_time=0.0,
            min_response_time=0.0,
            max_response_time=0.0,
            std_response_time=0.0,
            avg_response_length=0,
            avg_retrieved_docs=0.0,
        )

    response_times = [r.response_time for r in results]
    response_lengths = [r.response_length for r in results]
    retrieved_docs = [r.retrieved_docs for r in results]

    return BenchmarkSummary(
        total_queries=len(results),
        avg_response_time=mean(response_times),
        median_response_time=median(response_times),
        min_response_time=min(response_times),
        max_response_time=max(response_times),
        std_response_time=stdev(response_times) if len(response_times) > 1 else 0.0,
        avg_response_length=int(mean(response_lengths)),
        avg_retrieved_docs=mean(retrieved_docs),
    )


def save_results(
    results: List[BenchmarkResult], summary: BenchmarkSummary, output_file: str
):
    """Save benchmark results to JSON file."""
    data = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
        "metadata": {
            "timestamp": summary.timestamp,
            "script_version": "1.0.0",
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info("Results saved to: %s", output_file)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG pipeline performance")
    parser.add_argument(
        "--queries",
        "-q",
        type=int,
        default=10,
        help="Number of queries to run (default: 10)",
    )
    parser.add_argument(
        "--documents",
        "-d",
        type=int,
        default=5,
        help="Number of documents to index before benchmarking (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rag_performance_benchmark",
        help="Name for the MLflow experiment",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting RAG Performance Benchmark")
    logger.info("Queries to run: %d", args.queries)
    logger.info("Documents to index: %d", args.documents)

    try:
        # Initialize components
        logger.info("Initializing components...")
        rag_pipeline = get_rag_pipeline()
        indexing_pipeline = get_indexing_pipeline()
        tracker = MLflowExperimentTracker()

        # Start experiment tracking
        experiment_config = ExperimentConfig(
            experiment_name=args.experiment_name,
            run_name=f"benchmark_{int(time.time())}",
            model_name="google/flan-t5-base",  # Default model
            model_version="1.0.0",
            parameters={
                "num_queries": args.queries,
                "num_documents": args.documents,
            },
        )
        run_id = tracker.start_experiment(experiment_config)

        # Index sample documents
        if args.documents > 0:
            logger.info("Indexing sample documents...")
            sample_docs = get_sample_documents()[: args.documents]

            for doc_path in sample_docs:
                try:
                    logger.info("Indexing: %s", doc_path)
                    indexing_pipeline.index_document(doc_path)
                except (RuntimeError, ValueError, ConnectionError) as e:
                    logger.error("Failed to index %s: %s", doc_path, e)

            logger.info("Document indexing complete")

        # Get queries to benchmark
        queries = get_sample_queries()[: args.queries]
        logger.info("Running %d benchmark queries...", len(queries))

        # Run benchmarks
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(
                "Running query %d/%d: %s",
                i,
                len(queries),
                query[:50] + "..." if len(query) > 50 else query,
            )

            result = benchmark_single_query(rag_pipeline, query, tracker)
            results.append(result)

            logger.info(
                "  Response time: %.2fs, Length: %d chars, Retrieved docs: %d",
                result.response_time,
                result.response_length,
                result.retrieved_docs,
            )

        # Calculate summary
        summary = calculate_summary(results)
        summary.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Log summary to experiment tracker
        metrics = PerformanceMetrics(
            latency_ms=summary.avg_response_time * 1000,  # Convert to ms
            memory_usage_mb=summary.total_memory_usage_mb or 0.0,
            cpu_usage_percent=summary.avg_cpu_usage_percent or 0.0,
        )
        tracker.log_metrics(run_id, metrics)

        # Save results
        save_results(results, summary, args.output)

        # Print summary
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info("Total queries: %d", summary.total_queries)
        logger.info("Average response time: %.2f seconds", summary.avg_response_time)
        logger.info("Median response time: %.2f seconds", summary.median_response_time)
        logger.info(
            "Response time range: %.2f - %.2f seconds",
            summary.min_response_time,
            summary.max_response_time,
        )
        logger.info(
            "Average response length: %d characters", summary.avg_response_length
        )
        logger.info("Average retrieved documents: %.1f", summary.avg_retrieved_docs)
        logger.info("=" * 60)

        # End experiment
        tracker.end_experiment(run_id)

        logger.info("Benchmark complete! Results saved to: %s", args.output)

    except (RuntimeError, ValueError, ImportError) as e:
        logger.error("Benchmark failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
