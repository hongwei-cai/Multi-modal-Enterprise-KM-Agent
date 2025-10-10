#!/usr/bin/env python3
"""
Knowledge Base Indexer

Indexes organized QA pairs into ChromaDB for RAG retrieval.
Supports different indexing strategies for questions, answers, or combined Q&A.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.rag.embedding import get_embedding_model
from src.rag.vector_database import get_vector_db

logger = logging.getLogger(__name__)


class KnowledgeBaseIndexer:
    """
    Indexes QA pairs from organized knowledge base into ChromaDB.

    Supports different indexing strategies:
    - question_only: Index questions for retrieval
    - answer_only: Index answers for retrieval
    - qa_combined: Index question + answer combined
    - qa_separate: Index both Q and A separately
    """

    def __init__(
        self, db_path: Optional[str] = None, collection_name: str = "knowledge_base"
    ):
        """
        Initialize the knowledge base indexer.

        Args:
            db_path: Path to ChromaDB persistence directory
            collection_name: Name of the collection to create/use
        """
        self.db = get_vector_db(db_path=db_path)
        self.embedding_model = get_embedding_model()
        self.collection_name = collection_name

        # Create collection with metadata
        self.db.create_collection(
            name=collection_name,
            metadata={
                "description": "Knowledge base with QA pairs for RAG",
                "type": "qa_pairs",
                "indexed_by": "question",
            },
        )

    def index_qa_pairs(
        self,
        qa_pairs_file: str,
        strategy: str = "question_only",
        batch_size: int = 100,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Index QA pairs from JSONL file into ChromaDB.

        Args:
            qa_pairs_file: Path to QA pairs JSONL file
            strategy: Indexing strategy ('question_only', 'answer_only',
            'qa_combined', 'qa_separate')
            batch_size: Number of pairs to process in each batch
            version: Dataset version for metadata

        Returns:
            Indexing statistics and metadata
        """
        logger.info(f"Starting QA pairs indexing with strategy: {strategy}")

        # Load QA pairs
        qa_pairs = self._load_qa_pairs(qa_pairs_file)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_pairs_file}")

        if not qa_pairs:
            raise ValueError(f"No QA pairs found in {qa_pairs_file}")

        # Prepare data based on strategy
        if strategy == "question_only":
            ids, texts, metadatas = self._prepare_question_only(qa_pairs, version)
        elif strategy == "answer_only":
            ids, texts, metadatas = self._prepare_answer_only(qa_pairs, version)
        elif strategy == "qa_combined":
            ids, texts, metadatas = self._prepare_qa_combined(qa_pairs, version)
        elif strategy == "qa_separate":
            ids, texts, metadatas = self._prepare_qa_separate(qa_pairs, version)
        else:
            raise ValueError(f"Unknown indexing strategy: {strategy}")

        # Index in batches
        total_indexed = 0
        for i in tqdm(range(0, len(texts), batch_size), desc="Indexing batches"):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            # Generate embeddings
            embeddings = self.embedding_model.encode_batch(batch_texts)

            # Add to database
            self.db.add_documents(
                ids=batch_ids,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                documents=batch_texts,
            )

            total_indexed += len(batch_texts)

        logger.info(f"Successfully indexed {total_indexed} QA pair items")

        # Return statistics
        stats = {
            "total_qa_pairs": len(qa_pairs),
            "total_indexed_items": total_indexed,
            "strategy": strategy,
            "collection_name": self.collection_name,
            "version": version or "unknown",
            "avg_question_length": sum(len(p["question"].split()) for p in qa_pairs)
            / len(qa_pairs),
            "avg_answer_length": sum(len(p["answer"].split()) for p in qa_pairs)
            / len(qa_pairs),
        }

        return stats

    def _load_qa_pairs(self, file_path: str) -> List[Dict[str, Any]]:
        """Load QA pairs from JSONL file."""
        qa_pairs = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
        return qa_pairs

    def _prepare_question_only(
        self, qa_pairs: List[Dict[str, Any]], version: Optional[str]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Prepare data for question-only indexing."""
        ids = []
        texts = []
        metadatas = []

        for pair in qa_pairs:
            qa_id = str(uuid.uuid4())
            ids.append(qa_id)
            texts.append(pair["question"])
            metadatas.append(
                {
                    "type": "question",
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "file_path": pair.get("file_path", ""),
                    "chunk_index": pair.get("chunk_index", 0),
                    "version": version,
                    "relevance_score": pair.get("metrics", {}).get(
                        "relevance_score", 0
                    ),
                    "accepted": pair.get("accepted", False),
                }
            )

        return ids, texts, metadatas

    def _prepare_answer_only(
        self, qa_pairs: List[Dict[str, Any]], version: Optional[str]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Prepare data for answer-only indexing."""
        ids = []
        texts = []
        metadatas = []

        for pair in qa_pairs:
            qa_id = str(uuid.uuid4())
            ids.append(qa_id)
            texts.append(pair["answer"])
            metadatas.append(
                {
                    "type": "answer",
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "file_path": pair.get("file_path", ""),
                    "chunk_index": pair.get("chunk_index", 0),
                    "version": version,
                    "relevance_score": pair.get("metrics", {}).get(
                        "relevance_score", 0
                    ),
                    "accepted": pair.get("accepted", False),
                }
            )

        return ids, texts, metadatas

    def _prepare_qa_combined(
        self, qa_pairs: List[Dict[str, Any]], version: Optional[str]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Prepare data for combined Q&A indexing."""
        ids = []
        texts = []
        metadatas = []

        for pair in qa_pairs:
            qa_id = str(uuid.uuid4())
            combined_text = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
            ids.append(qa_id)
            texts.append(combined_text)
            metadatas.append(
                {
                    "type": "qa_combined",
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "file_path": pair.get("file_path", ""),
                    "chunk_index": pair.get("chunk_index", 0),
                    "version": version,
                    "relevance_score": pair.get("metrics", {}).get(
                        "relevance_score", 0
                    ),
                    "accepted": pair.get("accepted", False),
                }
            )

        return ids, texts, metadatas

    def _prepare_qa_separate(
        self, qa_pairs: List[Dict[str, Any]], version: Optional[str]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Prepare data for separate Q&A indexing (creates 2 entries per pair)."""
        ids = []
        texts = []
        metadatas = []

        for pair in qa_pairs:
            base_id = str(uuid.uuid4())

            # Question entry
            ids.append(f"{base_id}_q")
            texts.append(pair["question"])
            metadatas.append(
                {
                    "type": "question",
                    "pair_id": base_id,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "file_path": pair.get("file_path", ""),
                    "chunk_index": pair.get("chunk_index", 0),
                    "version": version,
                    "relevance_score": pair.get("metrics", {}).get(
                        "relevance_score", 0
                    ),
                    "accepted": pair.get("accepted", False),
                }
            )

            # Answer entry
            ids.append(f"{base_id}_a")
            texts.append(pair["answer"])
            metadatas.append(
                {
                    "type": "answer",
                    "pair_id": base_id,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "file_path": pair.get("file_path", ""),
                    "chunk_index": pair.get("chunk_index", 0),
                    "version": version,
                    "relevance_score": pair.get("metrics", {}).get(
                        "relevance_score", 0
                    ),
                    "accepted": pair.get("accepted", False),
                }
            )

        return ids, texts, metadatas

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection."""
        try:
            if self.db.collection is None:
                return {
                    "collection_name": self.collection_name,
                    "total_items": 0,
                    "status": "not_created",
                }

            count = self.db.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_items": count,
                "status": "indexed" if count > 0 else "empty",
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}


def index_knowledge_base(
    qa_pairs_file: str,
    strategy: str = "question_only",
    collection_name: str = "knowledge_base",
    db_path: Optional[str] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to index QA pairs into knowledge base.

    Args:
        qa_pairs_file: Path to QA pairs JSONL file
        strategy: Indexing strategy
        collection_name: ChromaDB collection name
        db_path: ChromaDB path
        version: Dataset version

    Returns:
        Indexing statistics
    """
    indexer = KnowledgeBaseIndexer(db_path=db_path, collection_name=collection_name)
    stats = indexer.index_qa_pairs(qa_pairs_file, strategy=strategy, version=version)

    # Log completion
    logger.info("Knowledge base indexing completed:")
    logger.info(f"  - Strategy: {strategy}")
    logger.info(f"  - Total QA pairs: {stats['total_qa_pairs']}")
    logger.info(f"  - Indexed items: {stats['total_indexed_items']}")
    logger.info(f"  - Collection: {collection_name}")

    return stats


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Index QA pairs into ChromaDB")
    parser.add_argument(
        "--qa_pairs_file", required=True, help="Path to QA pairs JSONL file"
    )
    parser.add_argument(
        "--strategy",
        default="question_only",
        choices=["question_only", "answer_only", "qa_combined", "qa_separate"],
        help="Indexing strategy",
    )
    parser.add_argument(
        "--collection_name", default="knowledge_base", help="ChromaDB collection name"
    )
    parser.add_argument("--version", help="Dataset version")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        stats = index_knowledge_base(
            qa_pairs_file=args.qa_pairs_file,
            strategy=args.strategy,
            collection_name=args.collection_name,
            version=args.version,
        )
        print("Indexing completed successfully!")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        exit(1)
