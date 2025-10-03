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


@pytest.fixture(autouse=True)
def set_test_model(monkeypatch):
    """Set a consistent model for integration tests."""
    monkeypatch.setenv("LLM_MODEL_NAME", "google/flan-t5-small")
    monkeypatch.setenv("MODEL_PRIORITY", "balanced")


def test_full_flow_upload_and_ask(test_pdf_path):
    # Upload
    with open(test_pdf_path, "rb") as f:
        files = {"file": ("test_document_ai_and_ds.pdf", f, "application/pdf")}
        upload_response = client.post("/upload", files=files)
    assert upload_response.status_code == 200

    # Ask question
    payload = {"question": "What is this document about?", "top_k": 5}
    ask_response = client.post("/ask", json=payload)
    assert ask_response.status_code == 200
    response_data = ask_response.json()

    # Flexible validation: Check pipeline works (answer generated, context retrieved)
    assert isinstance(response_data["answer"], str) and len(response_data["answer"]) > 0
    assert len(response_data.get("context_docs", [])) > 0
    # Optional: Check for general relevance (e.g., contains relevant terms)
    answer_lower = response_data["answer"].lower()
    assert any(
        term in answer_lower
        for term in ["ai", "artificial", "intelligence", "document", "science", "tech"]
    ), f"Answer seems irrelevant: {response_data['answer']}"


def test_e2e_with_expected_results(test_pdf_path):
    # Upload
    with open(test_pdf_path, "rb") as f:
        files = {"file": ("test_document_ai_and_ds.pdf", f, "application/pdf")}
        upload_response = client.post("/upload", files=files)
    assert upload_response.status_code == 200

    test_cases = [
        {
            "question": "What is this document about?",
            "expected_contains": [
                "ai",
                "intelligence",
                "artificial",
                "science",
                "tech",
            ],
        },
        {
            "question": "What does the document contain?",
            "expected_contains": ["ai", "data", "science", "intelligence", "tech"],
        },
        {
            "question": "Who is the author of the document?",
            "expected_contains": [],
        },
    ]

    for case in test_cases:
        response = client.post("/ask", json={"question": case["question"]})
        assert response.status_code == 200
        answer = response.json()["answer"].lower()

        if case["expected_contains"]:
            assert any(
                keyword in answer for keyword in case["expected_contains"]
            ), f"Answer missing expected content for question: {case['question']}\
                  | Answer: {response.json()['answer']}"
        else:
            # For author, check if answer is short or indicates no author
            assert (
                len(answer.split()) < 50
            ), f"Unexpected long answer for author: {response.json()['answer']}"
