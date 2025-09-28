from unittest.mock import MagicMock, patch

import pytest

from src.rag.llm_client import LLMClient, get_llm_client


def test_llm_client_init_local():
    """Test local initialization with Transformers."""
    with patch("src.rag.llm_client.AutoTokenizer"), patch(
        "src.rag.llm_client.AutoModelForCausalLM"
    ):
        client = LLMClient()
        assert client.is_cloud is False
        assert client.device in ["mps", "cpu"]


def test_llm_client_init_cloud():
    """Test cloud initialization."""
    with patch.dict("os.environ", {"CLOUD_ENV": "1"}):
        client = LLMClient()
        assert client.is_cloud is True


def test_generate_local():
    """Test local generation."""
    with patch("src.rag.llm_client.AutoTokenizer") as mock_tokenizer_class, patch(
        "src.rag.llm_client.AutoModelForCausalLM"
    ) as mock_model_class:
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.decode = MagicMock(return_value="Generated response")
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = [1, 2, 3]
        mock_model_class.from_pretrained.return_value = mock_model

        client = LLMClient()
        response = client.generate("Test prompt")
        assert response == "Generated response"


def test_generate_cloud():
    """Test cloud generation via API."""
    with patch("src.rag.llm_client.requests.post") as mock_post, patch.dict(
        "os.environ", {"CLOUD_ENV": "1"}
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "API response"}}]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        response = client.generate("Test prompt")
        assert response == "API response"


def test_get_llm_client():
    """Test convenience function."""
    client = get_llm_client()
    assert isinstance(client, LLMClient)


def test_generate_with_parameters():
    """Test generation with custom parameters."""
    with patch("src.rag.llm_client.AutoTokenizer") as mock_tokenizer_class, patch(
        "src.rag.llm_client.AutoModelForCausalLM"
    ) as mock_model_class:
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.decode = MagicMock(return_value="Generated response")
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = [1, 2, 3]
        mock_model_class.from_pretrained.return_value = mock_model

        client = LLMClient()
        response = client.generate("Test", temperature=0.5, top_p=0.8)
        assert response == "Generated response"


def test_parameter_validation():
    """Test parameter validation."""
    client = LLMClient()
    with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
        client.generate("Test", temperature=3.0)
    with pytest.raises(ValueError, match="Top-p must be between 0 and 1"):
        client.generate("Test", top_p=1.5)
