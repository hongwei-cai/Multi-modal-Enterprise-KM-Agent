"""
Document indexing pipeline: parse → chunk → embed → store.
"""
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.rag.document_parser import parse_pdf, parse_txt
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

    def _process_file(
        self,
        file_path: str,
        chunk_size: int,
        overlap: int,
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], int]:
        """
        Process a single file: parse and chunk.

        Returns:
            chunks, ids, metadatas, num_chunks
        """
        # Step 1: Parse document
        logger.info("Parsing document: %s", file_path)
        if file_path.lower().endswith(".pdf"):
            text = parse_pdf(file_path)
        elif file_path.lower().endswith(".txt"):
            text = parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        if not text:
            raise ValueError(f"No text extracted from {file_path}")

        # Step 2: Chunk text
        logger.info("Chunking text for %s", file_path)
        chunker = TextChunker(
            chunk_size=chunk_size, overlap=overlap, strategy="sentences"
        )
        chunks = chunker.chunk_text(text)
        if not chunks:
            raise ValueError("No chunks generated")

        # Prepare data
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"file_path": file_path, "chunk_index": i, **(metadata or {})}
            for i in range(len(chunks))
        ]

        return chunks, ids, metadatas, len(chunks)

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
            # Step 1 & 2: Parse and chunk document
            chunks, ids, metadatas, num_chunks = self._process_file(
                file_path, chunk_size, overlap, metadata
            )

            # Step 3 & 4: Generate embeddings and store in vector DB
            logger.info("Generating embeddings and storing in vector database")
            self.db.add_documents_with_embeddings(
                ids, chunks, metadatas, self.embedding_model
            )

            logger.info("Successfully indexed %d chunks from %s", num_chunks, file_path)

        except Exception as e:
            logger.error("Failed to index %s: %s", file_path, e)
            raise  # Re-raise for error recovery in caller

    def index_documents_batch(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        overlap: int = 128,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        """
        Index multiple documents in batch: parse all, chunk all,
        embed in batches, store in batches.

        Args:
            file_paths: List of document paths.
            chunk_size: Size of text chunks.
            overlap: Overlap between chunks.
            metadata: Additional metadata.
            batch_size: Size of batches for embedding and storage.
                If None, process all at once.
            num_workers: Number of worker threads for parallel processing of files.
            If None, process sequentially.
        """
        all_chunks = []
        all_ids = []
        all_metadatas = []
        total_chunks = 0

        if num_workers is None or num_workers == 1:
            # Sequential processing
            for file_path in tqdm(file_paths, desc="Processing documents"):
                try:
                    chunks, ids, metadatas, num_chunks = self._process_file(
                        file_path, chunk_size, overlap, metadata
                    )
                    all_chunks.extend(chunks)
                    all_ids.extend(ids)
                    all_metadatas.extend(metadatas)
                    total_chunks += num_chunks
                except Exception as e:
                    logger.warning("Skipping %s due to error: %s", file_path, e)
                    continue
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_file, file_path, chunk_size, overlap, metadata
                    ): file_path
                    for file_path in file_paths
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing documents",
                ):
                    file_path = futures[future]
                    try:
                        chunks, ids, metadatas, num_chunks = future.result()
                        all_chunks.extend(chunks)
                        all_ids.extend(ids)
                        all_metadatas.extend(metadatas)
                        total_chunks += num_chunks
                    except Exception as e:
                        logger.warning("Skipping %s due to error: %s", file_path, e)
                        continue

        if not all_chunks:
            logger.warning("No chunks to index")
            return

        # Determine batch size
        if batch_size is None:
            batch_size = len(all_chunks)

        # Step 3 & 4: Generate embeddings and store in batches
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]
            batch_metadatas = all_metadatas[i : i + batch_size]

            logger.info(
                "Generating embeddings and storing batch %d-%d in vector database",
                i,
                i + len(batch_chunks),
            )
            self.db.add_documents_with_embeddings(
                batch_ids, batch_chunks, batch_metadatas, self.embedding_model
            )

        logger.info(
            "Successfully indexed %d chunks from %d documents",
            total_chunks,
            len(file_paths),
        )


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
