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


@patch("src.api.main.get_rag_pipeline")
def test_ask_question(mock_get_rag):
    """Test QA endpoint."""
    mock_rag = mock_get_rag.return_value
    mock_rag.answer_question.return_value = "Test answer"

    payload = {"question": "What is AI?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"
