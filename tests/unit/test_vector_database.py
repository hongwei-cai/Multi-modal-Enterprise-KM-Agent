import tempfile

import pytest

from src.rag.vector_database import VectorDatabase, get_vector_db


@pytest.fixture(name="temp_db")
def temp_db():
    """Fixture for a temporary ChromaDB instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db = VectorDatabase(persist_directory=temp_dir)
        yield db


def test_vector_database_init(temp_db):
    """Test VectorDatabase initialization."""
    assert temp_db.client is not None
    assert temp_db.collection is None


def test_create_collection(temp_db):
    """Test creating a collection."""
    temp_db.create_collection("test_collection")
    assert temp_db.collection is not None
    assert temp_db.collection.name == "test_collection"


def test_add_documents(temp_db):
    """Test adding documents to the collection."""
    temp_db.create_collection("test_collection")

    # Dummy data
    ids = ["id1", "id2"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadatas = [{"source": "file1"}, {"source": "file2"}]
    documents = ["Document 1", "Document 2"]

    temp_db.add_documents(ids, embeddings, metadatas, documents)

    # Verify by querying (ChromaDB stores data)
    results = temp_db.query([0.1, 0.2, 0.3], n_results=2)
    assert len(results["ids"][0]) == 2
    assert results["documents"][0] == documents


def test_query(temp_db):
    """Test querying the collection."""
    temp_db.create_collection("test_collection")

    # Add data
    ids = ["id1"]
    embeddings = [[0.1, 0.2, 0.3]]
    metadatas = [{"source": "file1"}]
    documents = ["Document 1"]
    temp_db.add_documents(ids, embeddings, metadatas, documents)

    # Query with similar embedding
    results = temp_db.query([0.1, 0.2, 0.3], n_results=1)
    assert "id1" in results["ids"][0]
    assert results["documents"][0][0] == "Document 1"


def test_delete_collection(temp_db):
    """Test deleting a collection."""
    temp_db.create_collection("test_collection")
    assert temp_db.collection is not None

    temp_db.delete_collection("test_collection")
    # ChromaDB may not raise error if collection doesn't exist, but check if recreated
    temp_db.create_collection("test_collection")  # Should work


def test_get_vector_db():
    """Test the convenience function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db = get_vector_db(db_path=temp_dir)
        assert isinstance(db, VectorDatabase)
