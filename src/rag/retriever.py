"""
Semantic search retriever using vector similarity.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from .embedding import get_embedding_model
from .experiment_tracker import RetrievalMetrics
from .vector_database import get_vector_db

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves top-k similar documents using vector similarity with\
        optional experiment tracking.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: str = "documents",
        top_k: int = 5,
        experiment_tracker=None,
        run_id: Optional[str] = None,
    ):
        self.db = get_vector_db(db_path=db_path)
        self.collection_name = collection_name
        self.embedding_model = get_embedding_model()
        self.top_k = top_k
        self.experiment_tracker = experiment_tracker
        self.run_id = run_id
        self.db.create_collection(collection_name)  # Ensure collection exists

        logger.info("Retriever initialized with collection: %s", collection_name)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar documents for\
            the query with optional experiment tracking.

        Args:
            query: Search query string.
            top_k: Number of results to return (overrides instance default).

        Returns:
            List of dicts with 'document', 'metadata', 'distance'.
        """
        if top_k is None:
            top_k = self.top_k

        if self.db.collection is None:
            raise ValueError(
                "Collection not initialized. Call create_collection first."
            )

        # Start timing
        start_time = time.time()

        # Embed the query
        query_embedding = self.embedding_model.encode(query)

        # Query the vector DB
        results = self.db.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        # Calculate retrieval latency
        retrieval_latency_ms = (time.time() - start_time) * 1000

        # Process results efficiently
        retrieved_docs = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_docs.append({"document": doc, "metadata": metadata})

        # Log retrieval metrics if tracker is available
        if self.experiment_tracker and self.run_id:
            retrieval_metrics = RetrievalMetrics(
                query=query,
                retrieved_docs_count=len(retrieved_docs),
                retrieval_latency_ms=retrieval_latency_ms,
            )
            self.experiment_tracker.log_retrieval_metrics(
                self.run_id, retrieval_metrics
            )

        logger.info(
            "Retrieved %d documents for query: %s... (latency: %.2fms)",
            len(retrieved_docs),
            query[:50],
            retrieval_latency_ms,
        )
        return retrieved_docs


# Convenience function
def get_retriever(
    db_path: Optional[str] = None,
    top_k: int = 5,
    experiment_tracker=None,
    run_id: Optional[str] = None,
) -> Retriever:
    """
    Get a Retriever instance with optional experiment tracking.

    Args:
        db_path: Path to vector DB.
        top_k: Default number of results.
        experiment_tracker: Optional MLflow experiment tracker.
        run_id: Optional run ID for logging.

    Returns:
        Retriever instance.
    """
    return Retriever(
        db_path=db_path,
        top_k=top_k,
        experiment_tracker=experiment_tracker,
        run_id=run_id,
    )
