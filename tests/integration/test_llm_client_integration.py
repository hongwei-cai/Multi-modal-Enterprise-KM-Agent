from unittest.mock import MagicMock, patch

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
    response = client.generate("Hello", max_length=20)
    assert isinstance(response, str)
    assert len(response) > 0
    assert "Hello" in response or len(response) > 5


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
    questions = ["What is AI?"]

    for question in questions:
        response = client.generate(question, max_length=50)
        assert isinstance(response, str)
        assert len(response) > 10  # Reasonable length
        assert any(
            word in response.lower() for word in ["ai", "machine", "learning"]
        )  # Basic relevance check
        print(f"Q: {question} | A: {response}")  # For manual inspection


def test_llm_client_cloud_integration():
    """Integration test for cloud mode (mocked API)."""
    with patch("src.rag.llm_client.requests.post") as mock_post, patch.dict(
        "os.environ", {"CLOUD_ENV": "1", "VLLM_API_URL": "http://mock-api.com"}
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Mocked cloud response"}}]
        }
        mock_post.return_value = mock_response

        client = get_llm_client()
        response = client.generate("Test prompt", max_length=30)
        assert response == "Mocked cloud response"
        mock_post.assert_called_once()


@pytest.mark.skipif(
    not __import__("os").path.exists(
        __import__("os").path.expanduser(
            "~/.cache/huggingface/hub/models--microsoft--DialoGPT-medium"
        )
    ),
    reason="DialoGPT-medium model not cached",
)
def test_parameter_handling():
    """Test parameter handling in local generation."""
    client = get_llm_client()
    response = client.generate("Test", temperature=0.5, top_p=0.8, max_length=25)
    assert isinstance(response, str)
    assert len(response) > 0
