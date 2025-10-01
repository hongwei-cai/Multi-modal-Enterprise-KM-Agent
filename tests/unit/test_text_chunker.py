""" Unit tests for TextChunker class in src.rag.text_chunker."""

import pytest

from src.rag.text_chunker import TextChunker  # Import the class instead of functions


def test_chunk_text_by_tokens_basic():
    """Test basic token-based chunking with overlap."""
    chunker = TextChunker(strategy="tokens", chunk_size=10, overlap=2)
    text = "This is a test document for chunking."
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        len(chunker.tokenizer.tokenize(chunk)) <= chunker.chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"
    # Note: Overlap check may vary due to tokenization;
    # adjust based on tokenizer behavior


def test_chunk_text_by_tokens_no_overlap():
    """Test token-based chunking with zero overlap."""
    chunker = TextChunker(strategy="tokens", chunk_size=5, overlap=0)
    text = "Short text."
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 0, "Chunks should be generated"
    assert all(
        len(chunker.tokenizer.tokenize(chunk)) <= chunker.chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"


def test_chunk_text_by_tokens_empty():
    """Test token-based chunking with empty input."""
    chunker = TextChunker(strategy="tokens")
    chunks = chunker.chunk_text("")
    assert chunks == [], "Empty input should return empty chunks"


def test_chunk_text_by_tokens_invalid_params():
    """Test invalid parameters for token-based chunking."""
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        TextChunker(strategy="tokens", chunk_size=0)
    with pytest.raises(
        ValueError, match="overlap must be non-negative and less than chunk_size"
    ):
        TextChunker(strategy="tokens", chunk_size=10, overlap=10)


def test_chunk_text_by_words_basic():
    """Test word-based chunking."""
    chunker = TextChunker(strategy="words", chunk_size=5, overlap=2)
    text = "This is a test document for chunking."
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        len(chunk.split()) <= chunker.chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"


def test_chunk_text_by_sentences_basic():
    """Test sentence-based chunking."""
    chunker = TextChunker(strategy="sentences", chunk_size=10, overlap=2)
    text = "This is the first sentence. This is the second sentence."
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 0, "Chunks should not be empty"
    # Additional assertions based on sentence boundaries


def test_chunk_chinese_text_basic():
    """Test Chinese text chunking."""
    chunker = TextChunker(strategy="chinese", chunk_size=5, overlap=2)
    text = "这是中文文本。用于测试分词。"
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        len(chunk.split()) <= chunker.chunk_size for chunk in chunks
    ), "All chunks should respect chunk_size"


def test_normalize_text():
    """Test text normalization."""
    chunker = TextChunker()  # Default strategy
    text = "Hello　world"  # Full-width space
    normalized = chunker._normalize_text(
        text
    )  # Make _normalize_text public as normalize_text in the class
    assert normalized == "Hello world", "Full-width characters should be normalized"
