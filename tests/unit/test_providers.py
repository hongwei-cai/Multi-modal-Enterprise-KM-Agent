from unittest.mock import Mock, patch

from src.rag.providers.dashscope_client import DashscopeClient
from src.rag.providers.ollama_client import OllamaClient


def make_response(status_code=200, json_data=None, text_data="ok"):
    mock = Mock()
    mock.status_code = status_code
    mock.text = text_data
    if json_data is not None:
        mock.json.return_value = json_data
    else:
        mock.json.side_effect = ValueError("No JSON")

    def raise_for_status():
        if not (200 <= status_code < 300):
            raise Exception(f"HTTP {status_code}")

    mock.raise_for_status = raise_for_status
    return mock


def test_ollama_generate_success():
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3-8b")
    sample_json = {"choices": [{"text": "Hello from Ollama"}]}

    with patch("requests.post") as mock_post:
        mock_post.return_value = make_response(status_code=200, json_data=sample_json)
        out = client.generate("Say hi", max_tokens=10)
        assert "Hello from Ollama" in out


def test_ollama_generate_fallback_text():
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3-8b")
    # Return raw text when JSON parsing isn't available
    with patch("requests.post") as mock_post:
        mock_post.return_value = make_response(
            status_code=200, json_data=None, text_data="raw response body"
        )
        out = client.generate("Say hi", max_tokens=10)
        assert "raw response body" in out


def test_dashscope_generate_success():
    client = DashscopeClient(
        api_key="fake", base_url="https://dashscope.test/v1/generate", model="qwen3-max"
    )
    sample_json = {"result": "Hello from Dashscope"}

    with patch("requests.post") as mock_post:
        mock_post.return_value = make_response(status_code=200, json_data=sample_json)
        out = client.generate("Hello", max_tokens=20)
        assert "Hello from Dashscope" in out


def test_dashscope_generate_choices_shape():
    client = DashscopeClient(
        api_key="fake", base_url="https://dashscope.test/v1/generate", model="qwen3-max"
    )
    sample_json = {"choices": [{"text": "Choice text"}]}

    with patch("requests.post") as mock_post:
        mock_post.return_value = make_response(status_code=200, json_data=sample_json)
        out = client.generate("Test", max_tokens=5)
        assert "Choice text" in out
