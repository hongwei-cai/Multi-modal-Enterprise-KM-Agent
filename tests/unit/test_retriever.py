from unittest.mock import MagicMock, patch

import pytest

from src.rag.retriever import Retriever, get_retriever


@pytest.fixture
def mock_db():
    with patch("src.rag.retriever.get_vector_db") as mock_get_db:
        mock_db_instance = MagicMock()
        mock_db_instance.collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"source": "file1"}, {"source": "file2"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_get_db.return_value = mock_db_instance
        yield mock_db_instance


def test_retriever_init(mock_db):
    """Test Retriever initialization."""
    retriever = Retriever()
    assert retriever.top_k == 5


def test_retrieve(mock_db):
    """Test retrieval."""
    with patch("src.rag.retriever.get_embedding_model") as mock_get_emb:
        mock_emb = MagicMock()
        mock_emb.encode.return_value = [0.1, 0.2, 0.3]
        mock_get_emb.return_value = mock_emb

        retriever = Retriever()
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2
        assert results[0]["document"] == "Doc 1"


def test_get_retriever():
    """Test convenience function."""
    retriever = get_retriever(top_k=3)
    assert isinstance(retriever, Retriever)
    assert retriever.top_k == 3
