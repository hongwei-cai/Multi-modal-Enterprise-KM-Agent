from unittest.mock import MagicMock, patch

import pytest

from src.rag.components import RAGPipeline, get_rag_pipeline


@pytest.fixture
def mock_components():
    with patch("src.rag.components.rag_pipeline.get_retriever") as mock_ret, patch(
        "src.rag.components.rag_pipeline.get_prompt_template"
    ) as mock_pt, patch("src.rag.components.rag_pipeline.get_llm_client") as mock_llm:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [{"document": "Context text"}]
        mock_ret.return_value = mock_retriever

        mock_template = MagicMock()
        mock_template.format_prompt.return_value = "Formatted prompt"
        mock_pt.return_value = mock_template

        mock_client = MagicMock()
        mock_client.generate.return_value = "Generated answer"
        mock_llm.return_value = mock_client

        yield


def test_rag_pipeline_init(mock_components):
    """Test pipeline initialization."""
    pipeline = RAGPipeline()
    assert pipeline.max_context_length == 2000


def test_answer_question(mock_components):
    """Test end-to-end question answering."""
    pipeline = RAGPipeline()
    response = pipeline.answer_question("What is AI?")
    assert response["answer"] == "Generated answer"
    assert response["context_docs"] == ["Context text"]


def test_answer_question_no_context(mock_components):
    """Test with no retrieved context."""
    with patch("src.rag.components.rag_pipeline.get_retriever") as mock_ret:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_ret.return_value = mock_retriever

        pipeline = RAGPipeline()
        response = pipeline.answer_question("Test?")
        assert response["answer"] == "No relevant context found."
        assert response["context_docs"] == []


@patch("src.rag.llm_client.get_model_manager")
@patch("importlib.metadata.version")  # Patch to avoid bitsandbytes check
def test_get_rag_pipeline(mock_version, mock_get_manager):
    """Test convenience function."""
    mock_version.return_value = "0.41.0"  # Mock bitsandbytes version
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_manager.load_model_with_fallback.return_value = (mock_model, mock_tokenizer)
    mock_manager.get_model_recommendation.return_value = "microsoft/DialoGPT-medium"
    mock_get_manager.return_value = mock_manager

    pipeline = get_rag_pipeline(top_k=3)
    assert isinstance(pipeline, RAGPipeline)
