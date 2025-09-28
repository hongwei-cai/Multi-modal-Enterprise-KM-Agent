""" Integration tests for the indexing pipeline."""

import os
import tempfile

import pytest

from src.rag.indexing_pipeline import get_indexing_pipeline


@pytest.fixture
def temp_db():
    """Create a temporary directory for the vector DB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_index_document_integration(temp_db):
    """Integration test: full pipeline from PDF to vector DB."""
    test_pdf = "tests/data/pdfs/test_document_simple.pdf"
    if not os.path.exists(test_pdf):
        pytest.skip("Test PDF not found")

    pipeline = get_indexing_pipeline(db_path=temp_db)
    # Index the document
    pipeline.index_document(test_pdf)

    # Verify storage: query for a known chunk
    results = pipeline.db.query([0.1] * 384, n_results=1)  # Dummy embedding
    assert len(results["ids"][0]) > 0
    assert "simple pdf" in results["documents"][0][0]  # Based on test PDF content


def test_index_documents_batch_integration(temp_db):
    """Test batch indexing with error recovery."""
    test_pdfs = ["tests/data/pdfs/test_document_simple.pdf", "nonexistent.pdf"]
    pipeline = get_indexing_pipeline(db_path=temp_db)

    # Should index the valid PDF and skip the invalid one
    pipeline.index_documents_batch(test_pdfs)

    results = pipeline.db.query([0.1] * 384, n_results=5)
    assert len(results["ids"][0]) > 0  # At least one chunk indexed
