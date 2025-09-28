import pytest

from src.rag.llm_client import get_llm_client


@pytest.mark.skipif(
    not __import__("os").path.exists(
        __import__("os").path.expanduser(
            "~/.cache/huggingface/hub/models--microsoft--DialoGPT-medium"
        )
    ),
    reason="DialoGPT-medium model not cached; run client once to download",
)
def test_llm_client_integration_generate():
    """Light integration test: Generate response with real model (if cached)."""
    client = get_llm_client()
    response = client.generate("Hello", max_length=20)  # Short prompt/length for speed
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Hello" in response or len(response) > 5  # Basic check for generation


@pytest.mark.skipif(
    not __import__("os").path.exists(
        __import__("os").path.expanduser(
            "~/.cache/huggingface/hub/models--microsoft--DialoGPT-medium"
        )
    ),
    reason="DialoGPT-medium model not cached",
)
def test_qa_response_quality():
    """Integration test for QA response quality."""
    client = get_llm_client()

    # Test conversational QA
    questions = ["What is AI?", "Tell me a joke.", "How does machine learning work?"]

    for question in questions:
        response = client.generate(question, max_length=50)
        assert isinstance(response, str)
        assert len(response) > 10  # Reasonable length
        assert any(
            word in response.lower() for word in ["ai", "machine", "learning", "joke"]
        )  # Basic relevance check
        print(f"Q: {question} | A: {response}")  # For manual inspection
