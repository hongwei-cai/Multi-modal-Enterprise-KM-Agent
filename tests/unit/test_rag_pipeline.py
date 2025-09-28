from unittest.mock import MagicMock, patch

import pytest

from src.rag.rag_pipeline import RAGPipeline, get_rag_pipeline


@pytest.fixture
def mock_components():
    with patch("src.rag.rag_pipeline.get_retriever") as mock_ret, patch(
        "src.rag.rag_pipeline.get_prompt_template"
    ) as mock_pt, patch("src.rag.rag_pipeline.get_llm_client") as mock_llm:
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
    answer = pipeline.answer_question("What is AI?")
    assert answer == "Generated answer"


def test_answer_question_no_context(mock_components):
    """Test with no retrieved context."""
    with patch("src.rag.rag_pipeline.get_retriever") as mock_ret:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_ret.return_value = mock_retriever

        pipeline = RAGPipeline()
        answer = pipeline.answer_question("Test?")
        assert answer == "No relevant context found."


def test_get_rag_pipeline():
    """Test convenience function."""
    pipeline = get_rag_pipeline(top_k=3)
    assert isinstance(pipeline, RAGPipeline)
