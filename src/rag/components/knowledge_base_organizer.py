"""
Knowledge Base Organization System for RAG Dataset Management.

This module provides functionality for:
- Topic-based document clustering using embeddings
- Stratified train/validation/test splits (80/10/10)
- Dataset versioning and metadata management
- Quality statistics and reporting
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..embedding import EmbeddingModel
from ..experiment_tracker import MLflowExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for a dataset split."""

    version: str
    created_at: str
    source_file: str
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    topics_count: int
    split_ratios: Dict[str, float]
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    clustering_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicCluster:
    """Represents a topic cluster with its QA pairs."""

    topic_id: int
    centroid: np.ndarray
    qa_pairs: List[Dict[str, Any]]
    topic_keywords: List[str] = field(default_factory=list)


class TopicClustering:
    """
    Topic-based document clustering using embeddings.

    Groups QA pairs into semantic topics using K-means clustering
    on question embeddings.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize topic clustering.

        Args:
            embedding_model: Embedding model to use. If None, creates default.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.clusters: List[TopicCluster] = []

    def cluster_qa_pairs(
        self,
        qa_pairs: List[Dict[str, Any]],
        n_topics: Optional[int] = None,
        min_cluster_size: int = 5,
    ) -> List[TopicCluster]:
        """
        Cluster QA pairs into topics using embeddings.

        Args:
            qa_pairs: List of QA pair dictionaries
            n_topics: Number of topics. If None, auto-determine optimal.
            min_cluster_size: Minimum samples per cluster

        Returns:
            List of TopicCluster objects
        """
        if not qa_pairs:
            logger.warning("No QA pairs provided for clustering")
            return []

        # Extract questions for embedding
        questions = [pair["question"] for pair in qa_pairs]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(questions)} questions")
        embeddings = np.array(
            [self.embedding_model.encode(question) for question in questions]
        )

        # Determine optimal number of clusters if not provided
        if n_topics is None:
            n_topics = self._find_optimal_clusters(embeddings, max_clusters=20)

        logger.info(f"Clustering into {n_topics} topics")

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group QA pairs by cluster
        cluster_groups = defaultdict(list)
        for pair, label in zip(qa_pairs, cluster_labels):
            cluster_groups[label].append(pair)

        # Create TopicCluster objects
        clusters = []
        for topic_id in range(n_topics):
            cluster_pairs = cluster_groups[topic_id]

            # Skip small clusters
            if len(cluster_pairs) < min_cluster_size:
                logger.warning(
                    f"Skipping cluster {topic_id} with only {len(cluster_pairs)} samples"
                )
                continue

            # Calculate centroid
            cluster_embeddings = embeddings[cluster_labels == topic_id]
            centroid = np.mean(cluster_embeddings, axis=0)

            # Extract topic keywords (simplified - could use TF-IDF)
            topic_keywords = self._extract_topic_keywords(cluster_pairs)

            cluster = TopicCluster(
                topic_id=topic_id,
                centroid=centroid,
                qa_pairs=cluster_pairs,
                topic_keywords=topic_keywords,
            )
            clusters.append(cluster)

        self.clusters = clusters
        logger.info(f"Created {len(clusters)} topic clusters")
        return clusters

    def _find_optimal_clusters(
        self, embeddings: np.ndarray, max_clusters: int = 20
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            embeddings: Embedding matrix
            max_clusters: Maximum number of clusters to try

        Returns:
            Optimal number of clusters
        """
        if len(embeddings) < 10:
            return max(2, len(embeddings) // 5)

        max_clusters = min(max_clusters, len(embeddings) // 2)

        best_score = -1
        best_n_clusters = 2

        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)

                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(
                    f"Failed to compute silhouette score for {n_clusters} clusters: {e}"
                )
                continue

        logger.info(
            f"Optimal number of clusters: {best_n_clusters} (silhouette score: {best_score:.3f})"
        )
        return best_n_clusters

    def _extract_topic_keywords(
        self, qa_pairs: List[Dict[str, Any]], top_n: int = 5
    ) -> List[str]:
        """
        Extract topic keywords from cluster QA pairs.

        Args:
            qa_pairs: QA pairs in cluster
            top_n: Number of keywords to extract

        Returns:
            List of topic keywords
        """
        # Simple keyword extraction based on question words
        all_words = []
        for pair in qa_pairs:
            question = pair["question"].lower()
            # Remove common question words
            words = [
                w
                for w in question.split()
                if w
                not in {
                    "what",
                    "who",
                    "when",
                    "where",
                    "why",
                    "how",
                    "which",
                    "is",
                    "are",
                    "was",
                    "were",
                    "the",
                    "a",
                    "an",
                }
            ]
            all_words.extend(words)

        # Count word frequencies
        word_counts: DefaultDict[str, int] = defaultdict(int)
        for word in all_words:
            if len(word) > 3:  # Only consider meaningful words
                word_counts[word] += 1

        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]


class KnowledgeBaseOrganizer:
    """
    Main class for organizing knowledge base with clustering, splitting, and versioning.
    """

    def __init__(self, base_dir: str = "data/processed"):
        """
        Initialize knowledge base organizer.

        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = EmbeddingModel()
        self.topic_clustering = TopicClustering(self.embedding_model)
        self.experiment_tracker = MLflowExperimentTracker()

        # Dataset versioning
        self.version_file = self.base_dir / "dataset_versions.json"
        self._load_versions()

    def _load_versions(self):
        """Load dataset version history."""
        if self.version_file.exists():
            with open(self.version_file, "r") as f:
                self.versions = json.load(f)
        else:
            self.versions = {}

    def _save_versions(self):
        """Save dataset version history."""
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def organize_knowledge_base(
        self,
        qa_pairs_file: str,
        n_topics: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_cluster_size: int = 5,
        version_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete knowledge base organization pipeline.

        Args:
            qa_pairs_file: Path to QA pairs JSONL file
            n_topics: Number of topics for clustering
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            min_cluster_size: Minimum samples per cluster
            version_name: Custom version name

        Returns:
            Organization results and metadata
        """
        # Load QA pairs
        qa_pairs = self._load_qa_pairs(qa_pairs_file)

        # Filter accepted pairs
        accepted_pairs = [pair for pair in qa_pairs if pair.get("accepted", False)]
        logger.info(
            f"Loaded {len(qa_pairs)} total QA pairs, {len(accepted_pairs)} accepted"
        )

        if len(accepted_pairs) < 10:
            raise ValueError(
                "Insufficient accepted QA pairs for meaningful organization"
            )

        # Perform topic clustering
        clusters = self.topic_clustering.cluster_qa_pairs(
            accepted_pairs, n_topics, min_cluster_size
        )

        # Create stratified splits
        splits = self._create_stratified_splits(
            clusters, train_ratio, val_ratio, test_ratio
        )

        # Generate version name
        if version_name is None:
            version_name = self._generate_version_name(qa_pairs_file, clusters)

        # Save organized datasets
        dataset_paths = self._save_datasets(splits, version_name)

        # Generate metadata and statistics
        metadata = self._generate_metadata(
            qa_pairs_file, accepted_pairs, clusters, splits, version_name
        )

        # Save metadata
        self._save_metadata(metadata, version_name)

        # Update version history
        self.versions[version_name] = {
            "created_at": datetime.now().isoformat(),
            "source_file": qa_pairs_file,
            "metadata": metadata.__dict__,
        }
        self._save_versions()

        # Generate quality report
        report = self._generate_quality_report(metadata, splits)

        results = {
            "version": version_name,
            "metadata": metadata,
            "dataset_paths": dataset_paths,
            "quality_report": report,
            "clusters": len(clusters),
            "total_samples": len(accepted_pairs),
        }

        logger.info(f"Knowledge base organization completed: {version_name}")
        return results

    def _load_qa_pairs(self, file_path: str) -> List[Dict[str, Any]]:
        """Load QA pairs from JSONL file."""
        qa_pairs = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
        return qa_pairs

    def _create_stratified_splits(
        self,
        clusters: List[TopicCluster],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create stratified splits ensuring no topic leakage.

        Args:
            clusters: Topic clusters
            train_ratio: Training ratio
            val_ratio: Validation ratio
            test_ratio: Test ratio

        Returns:
            Dictionary with train/val/test splits
        """
        train_pairs = []
        val_pairs = []
        test_pairs = []

        for cluster in clusters:
            pairs = cluster.qa_pairs
            np.random.shuffle(pairs)  # Shuffle within cluster

            n_total = len(pairs)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val

            # Ensure at least one sample per split if possible
            if n_train == 0 and n_total > 0:
                n_train = 1
                if n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

            train_pairs.extend(pairs[:n_train])
            val_pairs.extend(pairs[n_train : n_train + n_val])
            test_pairs.extend(pairs[n_train + n_val :])

        # Final shuffle to avoid any ordering bias
        np.random.shuffle(train_pairs)
        np.random.shuffle(val_pairs)
        np.random.shuffle(test_pairs)

        logger.info(
            f"Created splits: train={len(train_pairs)}, val={len(val_pairs)},\
                test={len(test_pairs)}"
        )

        return {"train": train_pairs, "validation": val_pairs, "test": test_pairs}

    def _generate_version_name(
        self, source_file: str, clusters: List[TopicCluster]
    ) -> str:
        """Generate unique version name based on content hash."""
        content_hash = hashlib.md5()
        content_hash.update(str(len(clusters)).encode())
        content_hash.update(Path(source_file).name.encode())
        content_hash.update(datetime.now().isoformat().encode())

        return f"kb_v{len(self.versions) + 1}_{content_hash.hexdigest()[:8]}"

    def _save_datasets(
        self, splits: Dict[str, List[Dict[str, Any]]], version_name: str
    ) -> Dict[str, str]:
        """Save dataset splits to files."""
        version_dir = self.base_dir / version_name
        version_dir.mkdir(exist_ok=True)

        paths = {}
        for split_name, pairs in splits.items():
            file_path = version_dir / f"{split_name}.jsonl"
            with open(file_path, "w") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
            paths[split_name] = str(file_path)

        logger.info(f"Saved datasets to {version_dir}")
        return paths

    def _generate_metadata(
        self,
        source_file: str,
        accepted_pairs: List[Dict[str, Any]],
        clusters: List[TopicCluster],
        splits: Dict[str, List[Dict[str, Any]]],
        version_name: str,
    ) -> DatasetMetadata:
        """Generate comprehensive metadata for the dataset."""

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(accepted_pairs, splits)

        # Clustering info
        clustering_info = {
            "n_clusters": len(clusters),
            "cluster_sizes": [len(c.qa_pairs) for c in clusters],
            "avg_cluster_size": np.mean([len(c.qa_pairs) for c in clusters]),
            "topic_keywords": {c.topic_id: c.topic_keywords for c in clusters},
        }

        metadata = DatasetMetadata(
            version=version_name,
            created_at=datetime.now().isoformat(),
            source_file=source_file,
            total_samples=len(accepted_pairs),
            accepted_samples=len(accepted_pairs),
            rejected_samples=0,  # Already filtered
            topics_count=len(clusters),
            split_ratios={
                "train": len(splits["train"]) / len(accepted_pairs),
                "validation": len(splits["validation"]) / len(accepted_pairs),
                "test": len(splits["test"]) / len(accepted_pairs),
            },
            quality_metrics=quality_metrics,
            clustering_info=clustering_info,
        )

        return metadata

    def _calculate_quality_metrics(
        self,
        accepted_pairs: List[Dict[str, Any]],
        splits: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset."""

        def get_metrics_for_split(split_pairs):
            if not split_pairs:
                return {}

            relevance_scores = [p["metrics"]["relevance_score"] for p in split_pairs]
            question_lengths = [p["metrics"]["question_len_words"] for p in split_pairs]
            answer_in_passage = [p["metrics"]["answer_in_passage"] for p in split_pairs]

            return {
                "avg_relevance_score": np.mean(relevance_scores),
                "min_relevance_score": np.min(relevance_scores),
                "max_relevance_score": np.max(relevance_scores),
                "avg_question_length": np.mean(question_lengths),
                "answer_in_passage_ratio": np.mean(answer_in_passage),
                "total_samples": len(split_pairs),
            }

        return {
            "overall": get_metrics_for_split(accepted_pairs),
            "train": get_metrics_for_split(splits["train"]),
            "validation": get_metrics_for_split(splits["validation"]),
            "test": get_metrics_for_split(splits["test"]),
        }

    def _save_metadata(self, metadata: DatasetMetadata, version_name: str):
        """Save metadata to file."""
        version_dir = self.base_dir / version_name
        metadata_file = version_dir / "metadata.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)

    def _generate_quality_report(
        self, metadata: DatasetMetadata, splits: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""

        report = {
            "dataset_overview": {
                "version": metadata.version,
                "created_at": metadata.created_at,
                "total_samples": metadata.total_samples,
                "topics_count": metadata.topics_count,
            },
            "split_distribution": metadata.split_ratios,
            "quality_metrics": metadata.quality_metrics,
            "clustering_analysis": {
                "cluster_sizes": metadata.clustering_info["cluster_sizes"],
                "avg_cluster_size": metadata.clustering_info["avg_cluster_size"],
                "topic_keywords": metadata.clustering_info["topic_keywords"],
            },
            "data_leakage_check": self._check_data_leakage(splits),
            "recommendations": self._generate_recommendations(metadata, splits),
        }

        # Save report
        version_dir = self.base_dir / metadata.version
        report_file = version_dir / "quality_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _check_data_leakage(
        self, splits: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Check for potential data leakage between splits."""

        # Check if same file_path appears in multiple splits
        train_files = set(p["file_path"] for p in splits["train"])
        val_files = set(p["file_path"] for p in splits["validation"])
        test_files = set(p["file_path"] for p in splits["test"])

        leakage = {
            "train_val_overlap": len(train_files & val_files),
            "train_test_overlap": len(train_files & test_files),
            "val_test_overlap": len(val_files & test_files),
            "no_leakage": len(train_files & val_files) == 0
            and len(train_files & test_files) == 0
            and len(val_files & test_files) == 0,
        }

        return leakage

    def _generate_recommendations(
        self, metadata: DatasetMetadata, splits: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate recommendations based on dataset analysis."""

        recommendations = []

        # Check split sizes
        min_samples = 10
        for split_name, pairs in splits.items():
            if len(pairs) < min_samples:
                recommendations.append(
                    f"Consider increasing {split_name} set size (currently {len(pairs)} samples)"
                )

        # Check quality metrics
        overall_metrics = metadata.quality_metrics["overall"]
        if overall_metrics.get("avg_relevance_score", 0) < 0.5:
            recommendations.append(
                "Low average relevance score - consider improving QA pair generation"
            )

        if overall_metrics.get("answer_in_passage_ratio", 0) < 0.8:
            recommendations.append(
                "Many answers not found in passages - review answer extraction logic"
            )

        # Check clustering
        if metadata.topics_count < 3:
            recommendations.append(
                "Few topics detected - consider adjusting clustering parameters"
            )

        cluster_sizes = metadata.clustering_info["cluster_sizes"]
        if max(cluster_sizes) / min(cluster_sizes) > 10:
            recommendations.append(
                "Large variation in cluster sizes - consider rebalancing topics"
            )

        return (
            recommendations if recommendations else ["Dataset looks good for training!"]
        )
