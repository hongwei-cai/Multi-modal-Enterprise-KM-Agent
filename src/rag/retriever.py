"""
Semantic search retriever using vector similarity.
"""
import logging
from typing import Any, Dict, List, Optional

from .embedding import get_embedding_model
from .vector_database import get_vector_db

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves top-k similar documents using vector similarity.
    """

    def __init__(self, db_path: Optional[str] = None, top_k: int = 5):
        self.db = get_vector_db(db_path=db_path)
        self.embedding_model = get_embedding_model()
        self.top_k = top_k
        self.db.create_collection("documents")  # Ensure collection exists

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar documents for the query.

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

        # Embed the query
        query_embedding = self.embedding_model.encode(query)

        # Query the vector DB
        results = self.db.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        # Process results efficiently
        retrieved_docs = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_docs.append({"document": doc, "metadata": metadata})

        logger.info(
            f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}..."
        )
        return retrieved_docs


# Convenience function
def get_retriever(db_path: Optional[str] = None, top_k: int = 5) -> Retriever:
    """
    Get a Retriever instance.

    Args:
        db_path: Path to vector DB.
        top_k: Default number of results.

    Returns:
        Retriever instance.
    """
    return Retriever(db_path=db_path, top_k=top_k)
