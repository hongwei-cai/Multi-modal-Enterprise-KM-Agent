#!/usr/bin/env python3
"""
Knowledge Base Organization CLI

Organizes QA pairs into topic-clustered datasets with train/validation/test splits.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.rag.components.knowledge_base_organizer import KnowledgeBaseOrganizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Organize knowledge base with topic clustering and dataset splits"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/qa_pairs.jsonl",
        help="Path to QA pairs JSONL file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for organized datasets",
    )

    parser.add_argument(
        "--n_topics",
        type=int,
        default=None,
        help="Number of topics for clustering (auto-determined if not specified)",
    )

    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )

    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )

    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")

    parser.add_argument(
        "--min_cluster_size", type=int, default=5, help="Minimum samples per cluster"
    )

    parser.add_argument(
        "--version_name",
        type=str,
        default=None,
        help="Custom version name for the dataset",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        logger.error(f"Split ratios must sum to 1.0, got {total_ratio}")
        sys.exit(1)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    try:
        # Initialize organizer
        organizer = KnowledgeBaseOrganizer(args.output_dir)

        # Run organization pipeline
        logger.info("Starting knowledge base organization...")
        results = organizer.organize_knowledge_base(
            qa_pairs_file=str(input_path),
            n_topics=args.n_topics,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_cluster_size=args.min_cluster_size,
            version_name=args.version_name,
        )

        # Print results
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE ORGANIZATION COMPLETED")
        print("=" * 60)

        print(f"Version: {results['version']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Topics Created: {results['clusters']}")

        print("\nDataset Paths:")
        for split, path in results["dataset_paths"].items():
            print(f"  {split.capitalize()}: {path}")

        print("\nSplit Distribution:")
        metadata = results["metadata"]
        for split, ratio in metadata.split_ratios.items():
            count = (
                len(results["dataset_paths"][split])
                if split in results["dataset_paths"]
                else 0
            )
            print(f"  {split.capitalize()}: {count} samples ({ratio:.1%})")

        print("\nQuality Metrics:")
        quality = metadata.quality_metrics
        if "overall" in quality:
            overall = quality["overall"]
            print(
                f"  Average Relevance Score: {overall.get('avg_relevance_score', 0):.3f}"
            )
            print(
                f"  Answer in Passage Ratio: {overall.get('answer_in_passage_ratio', 0):.1%}"
            )
            print(
                f"  Average Question Length: {overall.get('avg_question_length', 0):.1f} words"
            )

        print("\nClustering Analysis:")
        clustering = metadata.clustering_info
        print(f"  Number of Clusters: {clustering.get('n_clusters', 0)}")
        print(f"  Average Cluster Size: {clustering.get('avg_cluster_size', 0):.1f}")
        print(f"  Cluster Sizes: {clustering.get('cluster_sizes', [])}")

        print("\nData Leakage Check:")
        leakage = results["quality_report"].get("data_leakage_check", {})
        if leakage.get("no_leakage", False):
            print("  ✓ No data leakage detected between splits")
        else:
            print("  ⚠ Potential data leakage detected:")
            if leakage.get("train_val_overlap", 0) > 0:
                print(f"    Train/Val overlap: {leakage['train_val_overlap']} files")
            if leakage.get("train_test_overlap", 0) > 0:
                print(f"    Train/Test overlap: {leakage['train_test_overlap']} files")
            if leakage.get("val_test_overlap", 0) > 0:
                print(f"    Val/Test overlap: {leakage['val_test_overlap']} files")

        print("\nRecommendations:")
        recommendations = results["quality_report"].get("recommendations", [])
        for rec in recommendations:
            print(f"  • {rec}")

        print("\nMetadata and reports saved to:")
        print(f"  {args.output_dir}/{results['version']}/")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Knowledge base organization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
