import logging
import re
import unicodedata
from typing import List

import jieba
import spacy
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__)


def chunk_text_by_tokens(
    text: str, chunk_size: int = 512, overlap: int = 128
) -> List[str]:
    """
    Splits text into fixed-size chunks using subword tokenization
    (for transformer models). Ensures chunks are valid (non-empty) and handles
    overlapping gracefully.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum number of subwords per chunk (default: 512).
        overlap: Number of overlapping subwords between chunks (default: 128).

    Returns:
        List of chunked text strings.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")
    if not text:
        logger.debug("Empty input text: returning empty chunks")
        return []

    text = normalize_text(text)
    try:
        # Tokenize text into subwords (handles special tokens
        # like [CLS]/[SEP] internally)
        tokens = tokenizer.tokenize(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            # Convert tokens back to text (without special tokens for consistency)
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            if chunk_text:  # Skip empty chunks (unlikely but possible with edge cases)
                chunks.append(chunk_text)
            start += chunk_size - overlap  # Apply overlap

        logger.info(
            f"Token-based chunking: {len(chunks)} chunks from {len(tokens)} tokens "
            f"(size={chunk_size}, overlap={overlap})"
        )
        return chunks

    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise


def chunk_text_by_words(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> List[str]:
    """
    Splits English text into word-based chunks.
    Preserves word boundaries for readability.

    Args:
        text: Input English text.
        chunk_size: Maximum number of words per chunk (default: 200).
        overlap: Number of overlapping words between chunks (default: 50).

    Returns:
        List of chunked text strings.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    # Split text into words, preserving spaces and punctuation
    text = normalize_text(text)
    words = re.findall(r"\S+", text)
    if not words:
        logger.warning("Input text is empty or contains no words")
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        logger.debug(f"Created chunk {len(chunks)}: {len(chunk_words)} words")

        # Move start position with overlap
        start += chunk_size - overlap
        if start >= len(words):
            break

    logger.info(
        f"Text chunking complete: {len(chunks)} chunks created from {len(words)} words"
    )
    return chunks


def chunk_text_by_sentences(
    text: str, chunk_size: int = 512, overlap: int = 128
) -> List[str]:
    """
    Splits text into sentence-based chunks with overlap. Uses spaCy for sentence
    boundary detection.

    Args:
        text: Input text (any language supported by spaCy).
        chunk_size: Maximum number of subwords per chunk (default: 512).
        overlap: Number of overlapping subwords between chunks (default: 128).

    Returns:
        List of chunked text strings.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")
    if not text:
        logger.debug("Empty input text: returning empty chunks")
        return []

    text = normalize_text(text)
    try:
        doc = nlp(text)
        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for sent in doc.sents:
            sent_tokens = tokenizer.tokenize(sent.text)
            sent_length = len(sent_tokens)

            # If adding the whole sentence exceeds chunk_size, split it
            if current_length + sent_length > chunk_size and current_chunk:
                # Add the remaining part of the current chunk (with overlap if needed)
                remaining_space = chunk_size - current_length
                if remaining_space > 0:
                    current_chunk = current_chunk[
                        -remaining_space:
                    ]  # Keep the last 'remaining_space' tokens
                    chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
                    current_length = 0
                    current_chunk = []

            # Add the current sentence (or part of it) to the chunk
            if current_length + sent_length <= chunk_size:
                current_chunk.extend(sent_tokens)
                current_length += sent_length
            else:
                # Split the sentence into smaller parts to fit the chunk
                remaining_tokens = sent_tokens
                while remaining_tokens:
                    if current_length >= chunk_size:
                        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
                        current_chunk = []
                        current_length = 0
                    # Take a slice of tokens that fits in the remaining space
                    take = min(chunk_size - current_length, len(remaining_tokens))
                    current_chunk.extend(remaining_tokens[:take])
                    current_length += take
                    remaining_tokens = remaining_tokens[take:]
                    if remaining_tokens:  # Apply overlap for the next chunk
                        overlap_tokens = remaining_tokens[:overlap]
                        current_chunk.extend(overlap_tokens)
                        current_length += len(overlap_tokens)
                        remaining_tokens = remaining_tokens[overlap:]

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

        logger.info(
            f"Sentence-based chunking: {len(chunks)} chunks from "
            f"{len(list(doc.sents))} sentences (size={chunk_size}, overlap={overlap})"
        )
        return chunks

    except Exception as e:
        logger.error(f"Sentence processing failed: {str(e)}")
        raise


def chunk_chinese_text(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> List[str]:
    """
    Splits Chinese text into word-based chunks using jieba. Joins words with
    spaces for readability.

    Args:
        text: Input Chinese text.
        chunk_size: Maximum number of words per chunk (default: 1000).
        overlap: Number of overlapping words between chunks (default: 200).

    Returns:
        List of chunked text strings.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")
    if not text:
        logger.debug("Empty input text: returning empty chunks")
        return []

    text = normalize_text(text)
    try:
        # Use jieba for Chinese word segmentation
        words = list(jieba.cut(text))
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(
                words[start:end]
            )  # Join words with spaces for better readability
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
            start += chunk_size - overlap  # Apply overlap

        logger.info(
            f"Chinese word-based chunking: {len(chunks)} chunks from {len(words)} \
                words (size={chunk_size}, overlap={overlap})"
        )
        return chunks

    except Exception as e:
        logger.error(f"Chinese text processing failed: {str(e)}")
        raise


def normalize_text(text: str) -> str:
    """
    Normalizes Unicode text (e.g., converts full-width to half-width characters).
    Useful for preprocessing text before chunking.

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.
    """
    return unicodedata.normalize("NFKC", text)
