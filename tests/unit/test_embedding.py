from src.rag.embedding import EmbeddingModel, get_embedding_model


def test_embedding_model_init():
    """Test model initialization."""
    model = EmbeddingModel()
    assert model.model is not None


def test_encode_single():
    """Test single text encoding."""
    model = EmbeddingModel()
    embedding = model.encode("Test text")
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension


def test_encode_batch():
    """Test batch encoding."""
    model = EmbeddingModel()
    embeddings = model.encode_batch(["Text 1", "Text 2"])
    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)


def test_get_embedding_model():
    """Test convenience function."""
    model = get_embedding_model()
    assert isinstance(model, EmbeddingModel)
