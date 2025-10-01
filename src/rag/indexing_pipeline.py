"""
Document indexing pipeline: parse → chunk → embed → store.
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.rag.document_parser import parse_pdf
from src.rag.embedding import get_embedding_model
from src.rag.text_chunker import TextChunker
from src.rag.vector_database import get_vector_db

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    End-to-end document indexing pipeline.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = get_vector_db(db_path=db_path)
        self.embedding_model = get_embedding_model()
        self.db.create_collection("documents")

    def index_document(
        self,
        file_path: str,
        chunk_size: int = 512,
        overlap: int = 128,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a single document: parse, chunk, embed, store.

        Args:
            file_path: Path to the document.
            chunk_size: Size of text chunks.
            overlap: Overlap between chunks.
            metadata: Additional metadata (e.g., {"source": "user_upload"}).
        """
        try:
            # Step 1: Parse document
            logger.info("Parsing document: %s", file_path)
            text = parse_pdf(file_path)
            if not text:
                raise ValueError(f"No text extracted from {file_path}")

            # Step 2: Chunk text
            logger.info("Chunking text")
            chunker = TextChunker(
                chunk_size=chunk_size, overlap=overlap, strategy="sentences"
            )
            chunks = chunker.chunk_text(text)
            if not chunks:
                raise ValueError("No chunks generated")

            # Step 3: Generate embeddings
            logger.info("Generating embeddings for %d chunks", len(chunks))
            embeddings = self.embedding_model.encode_batch(chunks)

            # Step 4: Prepare data for storage
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [
                {"file_path": file_path, "chunk_index": i, **(metadata or {})}
                for i in range(len(chunks))
            ]

            # Step 5: Store in vector DB
            logger.info("Storing in vector database")
            self.db.add_documents(ids, embeddings, metadatas, chunks)

            logger.info(
                "Successfully indexed %d chunks from %s", len(chunks), file_path
            )

        except Exception as e:
            logger.error("Failed to index %s: %s", file_path, e)
            raise  # Re-raise for error recovery in caller

    def index_documents_batch(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        overlap: int = 128,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index multiple documents with progress tracking.

        Args:
            file_paths: List of document paths.
            chunk_size: Size of text chunks.
            overlap: Overlap between chunks.
            metadata: Additional metadata.
        """
        for file_path in tqdm(file_paths, desc="Indexing documents"):
            try:
                self.index_document(file_path, chunk_size, overlap, metadata)
            except FileNotFoundError as f:
                logger.warning("Skipping %s due to error: %s", file_path, f)
                continue  # Error recovery: skip failed documents


# Convenience function
def get_indexing_pipeline(db_path: Optional[str] = None) -> IndexingPipeline:
    """
    Get an IndexingPipeline instance.

    Args:
        db_path: Path to vector DB.

    Returns:
        IndexingPipeline instance.
    """
    return IndexingPipeline(db_path=db_path)
