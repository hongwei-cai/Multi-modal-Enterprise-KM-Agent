import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from chromadb import EphemeralClient, HttpClient, PersistentClient

from src.rag.embedding import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    ChromaDB client wrapper for vector storage and retrieval.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory to persist the database. \
                If None, uses environment variables or fallback.
                - CHROMA_DB_PATH: HTTP URL for remote ChromaDB server
                - CHROMA_PERSIST_DIR: Local directory for persistence
        """
        if persist_directory is None:
            # Check for HTTP endpoint first
            http_url = os.getenv("CHROMA_DB_PATH")
            if http_url and http_url.startswith("http"):
                persist_directory = http_url
            else:
                # Check for local persistence directory
                persist_directory = os.getenv("CHROMA_PERSIST_DIR")
                if not persist_directory:
                    # Fallback: Project root chroma_db directory
                    persist_directory = str(
                        Path(__file__).parent.parent.parent / "chroma_db"
                    )
                    logger.warning(
                        "Neither CHROMA_DB_PATH nor CHROMA_PERSIST_DIR set, "
                        "using fallback: %s",
                        persist_directory,
                    )

        # Validate path (optional: ensure it's writable or exists)
        if not persist_directory.startswith("http") and persist_directory != ":memory:":
            try:
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    "Invalid persist_directory '%s': %s. Falling back to :memory:",
                    persist_directory,
                    e,
                )
                persist_directory = ":memory:"

        if persist_directory.startswith("http"):
            # Client mode: Connect to ChromaDB server
            host = persist_directory.split("://")[1].split(":")[0]
            port = int(persist_directory.split(":")[-1])
            self.client = HttpClient(host=host, port=port)
        else:
            if persist_directory == ":memory:" or os.getenv("PYTEST_CURRENT_TEST"):
                # Use ephemeral (in-memory) client during tests
                self.client = EphemeralClient()
            else:
                # Local persistent mode
                self.client = PersistentClient(path=persist_directory)
        self.collection = None

        logger.info(
            "ChromaDB client initialized with persist_directory: %s", persist_directory
        )

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

        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name, metadata=metadata
            )
            logger.info("Collection '%s' created or retrieved.", name)
        else:
            logger.info("Collection '%s' already exists.", name)

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
        Query the collection for similar vectors.

        Args:
            query_embedding: Query embedding vector.
            n_results: Number of results.

        Returns:
            Query results dict.
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        return self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results
        )

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Args:
            name: Name of the collection to delete.
        """
        self.client.delete_collection(name=name)
        logger.info("Collection '%s' deleted.", name)

    def get(self, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get all documents in the collection.

        Args:
            include: List of fields to include (e.g., ['metadatas']).

        Returns:
            Dict with included fields.
        """
        if self.collection is None:
            raise ValueError("Collection not created. Call create_collection() first.")

        return self.collection.get(include=include or [])


# Convenience function
def get_vector_db(db_path: Optional[str] = None) -> VectorDatabase:
    """
    Get a VectorDatabase instance.

    Args:
        db_path: Path to vector DB. Can be:
            - Local directory path for persistence
            - HTTP URL for remote ChromaDB server
            - If None, uses CHROMA_DB_PATH or CHROMA_PERSIST_DIR env vars

    Returns:
        VectorDatabase instance.
    """
    return VectorDatabase(persist_directory=db_path)
