"""
End-to-end integration test for the full RAG pipeline: upload → index → ask → answer.
"""
import os
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def test_pdf_path():
    """Path to a test PDF file."""
    path = "tests/data/pdfs/test_document_simple.pdf"
    if not os.path.exists(path):
        pytest.skip(
            "Test PDF not found. Create tests/data/pdfs/test_document_simple.pdf \
                with known content."
        )
    return path


def test_full_flow_upload_and_ask(test_pdf_path):
    """Test full flow: upload PDF, ask question, validate answer."""
    # Step 1: Upload the document
    with open(test_pdf_path, "rb") as f:
        files = {"file": ("test_document_simple.pdf", f, "application/pdf")}
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


@patch("src.rag.rag_pipeline.get_llm_client")
def test_e2e_with_expected_results(mock_get_llm):
    """Test with predefined dataset and expected results using mocked answers."""
    # Mock the LLM client to return answers containing expected keywords
    mock_llm_instance = mock_get_llm.return_value
    mock_llm_instance.generate.side_effect = [
        "AI stands for artificial intelligence, \
            which is a field of computer science.",
        "Python is a programming language commonly \
            used for data science and machine learning.",
    ]

    # Define test cases with expected results
    test_cases = [
        {
            "question": "What is AI?",
            "expected_keywords": ["artificial", "intelligence"],
        },
        {
            "question": "What is Python used for?",
            "expected_keywords": ["programming", "data", "science"],
        },
    ]

    for i, case in enumerate(test_cases):
        payload = {"question": case["question"]}
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        answer = response.json()["answer"].lower()
        # Check if at least one expected keyword is in the mocked answer
        assert any(
            keyword in answer for keyword in case["expected_keywords"]
        ), f"Answer missing expected content for question: \
            {case['question']} | Answer: {answer}"

        print(
            f"Test Case {i+1}: Question: {case['question']} | Answer: \
                {response.json()['answer']}"
        )
