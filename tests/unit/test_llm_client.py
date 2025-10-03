from unittest.mock import MagicMock, patch

import pytest

from src.rag.llm_client import LLMClient, get_llm_client


def test_llm_client_init_local():
    """Test local initialization with model_manager."""
    with patch("src.rag.llm_client.get_model_manager") as mock_get_manager:
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
        mock_get_manager.return_value = mock_manager

        client = LLMClient(model_name="google/flan-t5-small")
        assert client.is_cloud is False
        assert client.model == mock_model
        assert client.tokenizer == mock_tokenizer
        mock_manager.load_model.assert_called_once_with(
            "google/flan-t5-small", use_quantization=False, quant_type="dynamic"
        )


def test_llm_client_init_cloud():
    """Test cloud initialization."""
    with patch.dict("os.environ", {"CLOUD_ENV": "1"}):
        client = LLMClient(model_name="microsoft/DialoGPT-small")
        assert client.is_cloud is True
        assert hasattr(client, "api_url")


@patch("src.rag.llm_client.get_model_manager")
def test_generate_local(mock_get_manager):
    """Test local generation."""
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
    mock_get_manager.return_value = mock_manager

    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = MagicMock(return_value="Generated response")
    mock_model.generate.return_value = [1, 2, 3]
    mock_model.device = "cpu"

    client = LLMClient()
    response = client.generate("Test prompt")
    assert response == "Generated response"
    mock_tokenizer.decode.assert_called_once()


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
        mock_post.assert_called_once()


@patch("src.rag.llm_client.get_model_manager")
def test_get_llm_client(mock_get_manager):
    """Test convenience function."""
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
    mock_get_manager.return_value = mock_manager

    client = get_llm_client()
    assert isinstance(client, LLMClient)


@patch("transformers.AutoConfig")
@patch("src.rag.llm_client.get_model_manager")
def test_generate_with_parameters(mock_get_manager, mock_config):
    """Test generation with custom parameters."""
    mock_config.from_pretrained.return_value.model_type = "t5"
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
    mock_get_manager.return_value = mock_manager

    # Mock tokenizer to return inputs dict
    mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    for key in mock_inputs:
        mock_inputs[key].to.return_value = mock_inputs[key]
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.decode = MagicMock(return_value="Generated response")
    mock_model.generate.return_value = [1, 2, 3]
    mock_model.device = "cpu"

    client = LLMClient()
    response = client.generate("Test", temperature=0.5, top_p=0.8)
    assert response == "Generated response"
    mock_model.generate.assert_called_with(
        **mock_inputs,
        max_new_tokens=50,
        temperature=0.5,
        top_p=0.8,
        do_sample=True,
        pad_token_id=0
    )


@patch("src.rag.llm_client.get_model_manager")
def test_parameter_validation(mock_get_manager):
    """Test parameter validation."""
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
    mock_get_manager.return_value = mock_manager

    client = LLMClient()
    with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
        client.generate("Test", temperature=3.0)
    with pytest.raises(ValueError, match="Top-p must be between 0 and 1"):
        client.generate("Test", top_p=1.5)


@patch("src.rag.llm_client.get_model_manager")
def test_response_format(mock_get_manager):
    """Test that responses are valid strings."""
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
    mock_get_manager.return_value = mock_manager

    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode = MagicMock(return_value="This is a test response.")
    mock_model.generate.return_value = [1, 2, 3]
    mock_model.device = "cpu"

    client = LLMClient()
    response = client.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "test" in response.lower()  # Basic content check


@patch("src.rag.llm_client.get_model_manager")
def test_response_quality_basic(mock_get_manager):
    """Test basic response quality (length, no errors)."""
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model.return_value = (mock_model, mock_tokenizer)
    mock_get_manager.return_value = mock_manager

    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode = MagicMock(return_value="Hello! How can I help you today?")
    mock_model.generate.return_value = [1, 2, 3]
    mock_model.device = "cpu"

    client = LLMClient()
    response = client.generate("Hi", max_length=50)
    assert len(response.split()) > 3  # At least a few words
    assert not response.startswith("Error")  # No error messages
