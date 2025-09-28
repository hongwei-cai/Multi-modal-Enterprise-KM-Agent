"""
Embedding model integration for generating text embeddings using
sentence-transformers. Supports local development with MPS on Apple
Silicon and cloud migration to larger models.
"""
import logging
import os
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
os.environ["HF_HUB_TIMEOUT"] = "60"


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding models.
    Defaults to all-MiniLM-L6-v2 for local dev; switch to BAAI/bge-m3 for cloud.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Hugging Face model name.
            If None, uses EMBEDDING_MODEL env var or default.
        """
        if model_name is None:
            model_name = os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Loaded embedding model: {model_name}")

    def encode(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string.

        Returns:
            Embedding vector as list of floats.
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input text strings.
            batch_size: Batch size for processing.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(
            texts, batch_size=batch_size, convert_to_tensor=False
        )
        return embeddings.tolist()


# Convenience function
def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """
    Get an EmbeddingModel instance.

    Args:
        model_name: Model name. If None, uses env var or default.

    Returns:
        EmbeddingModel instance.
    """
    return EmbeddingModel(model_name=model_name)
