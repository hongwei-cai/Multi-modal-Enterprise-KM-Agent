from unittest.mock import MagicMock, patch

import pytest

from src.rag.llm_client import LLMClient, get_llm_client


def test_llm_client_init_local():
    """Test local initialization with model_manager."""
    with patch("src.rag.llm_client.get_model_manager") as mock_get_manager:
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_manager.load_model_with_fallback.return_value = (
            mock_model,
            mock_tokenizer,
        )
        mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
        mock_get_manager.return_value = mock_manager

        client = LLMClient(model_name="google/flan-t5-small")
        assert client.is_cloud is False
        assert client.model == mock_model
        assert client.tokenizer == mock_tokenizer
        mock_manager.load_model_with_fallback.assert_called_once_with(
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
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
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
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
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
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
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
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
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
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
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
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
    mock_get_manager.return_value = mock_manager

    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode = MagicMock(return_value="Hello! How can I help you today?")
    mock_model.generate.return_value = [1, 2, 3]
    mock_model.device = "cpu"

    client = LLMClient()
    response = client.generate("Hi", max_length=50)
    assert len(response.split()) > 3  # At least a few words
    assert not response.startswith("Error")  # No error messages


def test_llm_client_dynamic_model_selection():
    """Test dynamic model selection based on priority."""
    with patch("src.rag.llm_client.get_model_manager") as mock_get_manager, patch.dict(
        "os.environ", {"LLM_MODEL_NAME": "gpt2"}
    ):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock the model recommendation
        mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
        mock_manager.load_model_with_fallback.return_value = (
            mock_model,
            mock_tokenizer,
        )

        mock_get_manager.return_value = mock_manager

        # Test with speed priority
        LLMClient(priority="speed")
        mock_manager.get_model_recommendation.assert_called_with("speed")

        # Reset mock for next test
        mock_manager.reset_mock()

        # Test with quality priority
        LLMClient(priority="quality")
        mock_manager.get_model_recommendation.assert_called_with("quality")


def test_llm_client_model_switching():
    """Test dynamic model switching."""
    with patch("src.rag.llm_client.get_model_manager") as mock_get_manager:
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.load_model_with_fallback.return_value = (
            mock_model,
            mock_tokenizer,
        )
        mock_get_manager.return_value = mock_manager

        client = LLMClient(model_name="microsoft/DialoGPT-medium")

        # Test model switching
        client.switch_model("microsoft/DialoGPT-large")
        assert client.model_name == "microsoft/DialoGPT-large"
        mock_manager.load_model_with_fallback.assert_called()


def test_llm_client_benchmarking():
    """Test model benchmarking functionality."""
    with patch("src.rag.llm_client.get_model_manager") as mock_get_manager:
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_benchmark_result = MagicMock()

        mock_manager.benchmark_model.return_value = mock_benchmark_result
        mock_manager.load_model_with_fallback.return_value = (
            mock_model,
            mock_tokenizer,
        )
        mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
        mock_get_manager.return_value = mock_manager

        client = LLMClient()

        # Test benchmarking
        result = client.benchmark_current_model()
        assert result == mock_benchmark_result
        mock_manager.benchmark_model.assert_called_once()


def test_llm_client_optimal_model_selection():
    """Test optimal model selection for constraints."""
    with patch("src.rag.llm_client.get_model_manager") as mock_get_manager:
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.get_best_model_for_constraints.return_value = (
            "microsoft/DialoGPT-small"
        )
        mock_manager.load_model_with_fallback.return_value = (
            mock_model,
            mock_tokenizer,
        )
        mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
        mock_get_manager.return_value = mock_manager

        client = LLMClient()

        # Test optimal model selection
        optimal = client.get_optimal_model_for_constraints(max_latency_ms=100)
        assert optimal == "microsoft/DialoGPT-small"
        mock_manager.get_best_model_for_constraints.assert_called_with(100, None)
