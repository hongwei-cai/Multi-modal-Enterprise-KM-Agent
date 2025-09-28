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
