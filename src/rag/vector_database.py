""""""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from configs.vector_db_config import persist_directory
from src.rag.embedding import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    ChromaDB client wrapper for vector storage and retrieval.
    """

    def __init__(self, persist_directory: str = persist_directory):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory to persist the database.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

        self.collection = None
        logger.info("ChromaDB client initialized.")

    def create_collection(
        self, name: str = "documents", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create or get a collection for storing document vectors.

        Args:
            name: Name of the collection.
            metadata: Optional metadata for the collection.
        """
        if metadata is None:
            metadata = {
                "description": "Collection for storing document \
                        chunks and embeddings"
            }

        self.collection = self.client.get_or_create_collection(
            name=name, metadata=metadata
        )
        logger.info("Collection '%s' created or retrieved.", name)

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        """
        Add documents with embeddings to the collection.

        Args:
            ids: List of unique IDs for each document chunk.
            embeddings: List of embedding vectors.
            metadatas: List of metadata dictionaries (e.g., source file, chunk index).
            documents: List of text documents/chunks.
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )
        logger.info("Added %d documents to collection.", len(ids))

    def add_documents_with_embeddings(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        """
        Add documents with auto-generated embeddings.

        Args:
            ids: Unique IDs.
            documents: Text documents.
            metadatas: Metadata.
            embedding_model: EmbeddingModel instance (defaults to all-MiniLM-L6-v2).
        """
        if embedding_model is None:
            embedding_model = get_embedding_model()
        embeddings = embedding_model.encode_batch(documents)
        self.add_documents(ids, embeddings, metadatas, documents)

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection for similar documents.

        Args:
            query_embedding: Embedding vector for the query.
            n_results: Number of results to return.

        Returns:
            Query results including IDs, distances, metadatas, and documents.
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )
        return results

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Name of the collection to delete.
        """
        self.client.delete_collection(name=name)
        logger.info("Collection '%s' deleted.", name)


# Convenience function
def get_vector_db(db_path: Optional[str] = None) -> VectorDatabase:
    """
    Get a VectorDatabase instance.

    Args:
        db_path: Path to vector DB. If None, uses config default.

    Returns:
        VectorDatabase instance.
    """
    if db_path is None:
        db_path = persist_directory  # Fallback to config
    return VectorDatabase(persist_directory=db_path)
