from unittest.mock import patch

from starlette.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "message": "API is running"}


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


@patch("src.api.main.get_indexing_pipeline")
def test_upload_document(mock_get_indexer):
    """Test document upload."""
    mock_indexer = mock_get_indexer.return_value
    mock_indexer.index_document.return_value = None

    # Mock PDF file
    files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "uploaded" in response.json()["message"]


@patch("src.api.main.get_indexing_pipeline")
def test_upload_invalid_file(mock_get_indexer):
    """Test upload with invalid file type."""
    # No need to set up mock since validation happens before indexing
    files = {"file": ("test.txt", b"content", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@patch("src.api.main.get_rag_pipeline")
def test_ask_question(mock_get_rag):
    """Test QA endpoint."""
    mock_rag = mock_get_rag.return_value
    mock_rag.answer_question.return_value = {
        "answer": "Test answer",
        "context_docs": ["Doc1", "Doc2"],
    }

    payload = {"question": "What is AI?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["context_docs"] == ["Doc1", "Doc2"]


@patch("src.api.main.get_rag_pipeline")
def test_ask_invalid_question(mock_get_rag):
    """Test QA with invalid question."""
    # No need to set up mock since validation happens before RAG
    payload = {"question": ""}
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert "at least 1 characters" in response.json()["detail"][0]["msg"]
