""" Unit tests for text chunking functions in src.rag.text_chunker."""

import pytest

from src.rag.text_chunker import (
    chunk_chinese_text,
    chunk_text_by_sentences,
    chunk_text_by_tokens,
    chunk_text_by_words,
    normalize_text,
)


def test_chunk_text_by_tokens_basic():
    """Test basic token-based chunking with overlap."""
    text = "This is a test document for chunking."
    chunk_size = 10
    overlap = 2
    chunks = chunk_text_by_tokens(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        len(chunk.split()) <= chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"
    # Note: Overlap check may vary due to tokenization;
    # adjust based on tokenizer behavior


def test_chunk_text_by_tokens_no_overlap():
    """Test token-based chunking with zero overlap."""
    text = "Short text."
    chunk_size = 5
    overlap = 0
    chunks = chunk_text_by_tokens(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "Chunks should be generated"
    assert all(
        len(chunk.split()) <= chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"


def test_chunk_text_by_tokens_empty():
    """Test token-based chunking with empty input."""
    chunks = chunk_text_by_tokens("", chunk_size=10, overlap=2)
    assert chunks == [], "Empty input should return empty chunks"


def test_chunk_text_by_tokens_invalid_params():
    """Test invalid parameters for token-based chunking."""
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_text_by_tokens("test", chunk_size=0)
    with pytest.raises(
        ValueError, match="overlap must be non-negative and less than chunk_size"
    ):
        chunk_text_by_tokens("test", chunk_size=10, overlap=10)


def test_chunk_text_by_words_basic():
    """Test word-based chunking."""
    text = "This is a test document for chunking."
    chunk_size = 5
    overlap = 2
    chunks = chunk_text_by_words(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        len(chunk.split()) <= chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"


def test_chunk_text_by_sentences_basic():
    """Test sentence-based chunking."""
    text = "This is the first sentence. This is the second sentence."
    chunk_size = 10
    overlap = 2
    chunks = chunk_text_by_sentences(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "Chunks should not be empty"
    # Additional assertions based on sentence boundaries


def test_chunk_chinese_text_basic():
    """Test Chinese text chunking."""
    text = "这是中文文本。用于测试分词。"
    chunk_size = 5
    overlap = 2
    chunks = chunk_chinese_text(text, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        len(chunk.split()) <= chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"


def test_normalize_text():
    """Test text normalization."""
    text = "Hello　world"  # Full-width space
    normalized = normalize_text(text)
    assert normalized == "Hello world", "Full-width characters should be normalized"
