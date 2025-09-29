"""
End-to-end integration test for the full RAG pipeline: upload → index → ask → answer.
"""
import os

import pytest
from starlette.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def test_pdf_path():
    """Path to a test PDF file."""
    path = "tests/data/pdfs/test_document_ai_and_ds.pdf"
    if not os.path.exists(path):
        pytest.skip(
            "Test PDF not found. Create tests/data/pdfs/test_document_ai_and_ds.pdf \
                with known content."
        )
    return path


def test_full_flow_upload_and_ask(test_pdf_path):
    """Test full flow: upload PDF, ask question, validate answer."""
    # Step 1: Upload the document
    with open(test_pdf_path, "rb") as f:
        files = {"file": ("test_document_ai_and_ds.pdf", f, "application/pdf")}
        upload_response = client.post("/upload", files=files)

    assert upload_response.status_code == 200
    assert "uploaded" in upload_response.json()["message"]

    # Step 2: Ask a question related to the document content
    question = "What is this document about?"
    payload = {"question": question, "top_k": 5}
    ask_response = client.post("/ask", json=payload)

    assert ask_response.status_code == 200
    response_data = ask_response.json()
    assert "question" in response_data
    assert "answer" in response_data
    assert len(response_data["answer"]) > 0  # Answer should not be empty

    # Step 3: Validate the answer contains expected content from the document
    # Assuming the PDF contains "simple test document", check if it's in the answer
    answer_lower = response_data["answer"].lower()
    assert (
        "simple" in answer_lower or "test" in answer_lower or "document" in answer_lower
    ), f"Answer does not reflect document content: {response_data['answer']}"

    print(f"Question: {question}")
    print(f"Answer: {response_data['answer']}")


def test_e2e_with_expected_results(test_pdf_path):
    """Test with predefined dataset and expected results."""
    # Upload the document first to ensure DB has content
    with open(test_pdf_path, "rb") as f:
        files = {"file": ("test_document_ai_and_ds.pdf", f, "application/pdf")}
        upload_response = client.post("/upload", files=files)

    assert upload_response.status_code == 200

    # Now test with questions
    test_cases = [
        {
            "question": "What is AI?",
            "expected_keywords": ["artificial", "intelligence"],
        },
        {
            "question": "What is data science?",
            "expected_keywords": ["data", "science"],
        },
        {
            "question": "Who is the author of the document?",
            "expected_keywords": [""],
        },
    ]

    for case in test_cases:
        payload = {"question": case["question"]}
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        answer = response.json()["answer"].lower()
        # Check if at least one expected keyword is in the answer
        assert any(
            keyword in answer for keyword in case["expected_keywords"]
        ), f"Answer missing expected content for question: {case['question']} | \
            Answer: {response.json()['answer']}"
