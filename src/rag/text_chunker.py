""" Text chunking utilities for various languages and strategies. """

import logging
import re
import unicodedata
from typing import List

import jieba
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(
        self, chunk_size: int = 512, overlap: int = 128, strategy: str = "sentences"
    ):
        """
        Args:
            chunk_size: Max size per chunk (tokens, words, or characters \
                depending on strategy).
            overlap: Overlap between chunks.
            strategy: Chunking method - "tokens", "words", "sentences", "chinese".
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")
        if strategy not in ["tokens", "words", "sentences", "chinese"]:
            raise ValueError(
                "Invalid strategy. Choose from: tokens, words, sentences, chinese"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy

        # Lazy-load dependencies to avoid import-time failures
        self._tokenizer = None
        self._nlp = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                raise
        return self._tokenizer

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(
                    f"Failed to load spaCy model: {e}. Install with \
                        'python -m spacy download en_core_web_sm'"
                )
                raise
        return self._nlp

    def chunk_text(self, text: str) -> List[str]:
        """Unified chunking method based on selected strategy."""
        text = self._normalize_text(text)
        if self.strategy == "tokens":
            return self._chunk_by_tokens(text)
        elif self.strategy == "words":
            return self._chunk_by_words(text)
        elif self.strategy == "sentences":
            return self._chunk_by_sentences(text)
        elif self.strategy == "chinese":
            return self._chunk_chinese_text(text)
        else:
            raise ValueError("Unsupported strategy")

    def _chunk_by_tokens(self, text: str) -> List[str]:
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
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")
        if not text:
            logger.debug("Empty input text: returning empty chunks")
            return []

        try:
            # Tokenize text into subwords (handles special tokens
            # like [CLS]/[SEP] internally)
            tokens = self.tokenizer.tokenize(text)
            chunks = []
            start = 0

            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                # Convert tokens back to text (without special tokens for consistency)
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                if (
                    chunk_text
                ):  # Skip empty chunks (unlikely but possible with edge cases)
                    chunks.append(chunk_text)
                start += self.chunk_size - self.overlap  # Apply overlap

            logger.info(
                f"Token-based chunking: {len(chunks)} chunks from {len(tokens)} tokens "
                f"(size={self.chunk_size}, overlap={self.overlap})"
            )
            return chunks

        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise

    def _chunk_by_words(self, text: str) -> List[str]:
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
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")

        # Split text into words, preserving spaces and punctuation
        words = re.findall(r"\S+", text)
        if not words:
            logger.warning("Input text is empty or contains no words")
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            logger.debug(f"Created chunk {len(chunks)}: {len(chunk_words)} words")

            # Move start position with overlap
            start += self.chunk_size - self.overlap
            if start >= len(words):
                break

        logger.info(
            f"Text chunking complete: {len(chunks)} chunks created \
                from {len(words)} words"
        )
        return chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
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
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")
        if not text:
            logger.debug("Empty input text: returning empty chunks")
            return []

        try:
            doc = self.nlp(text)
            chunks = []
            current_chunk: List[str] = []
            current_length = 0

            for sent in doc.sents:
                sent_tokens = self.tokenizer.tokenize(sent.text)
                sent_length = len(sent_tokens)

                # If adding the whole sentence exceeds chunk_size, split it
                if current_length + sent_length > self.chunk_size and current_chunk:
                    # Add the remaining part of the current chunk
                    remaining_space = self.chunk_size - current_length
                    if remaining_space > 0:
                        current_chunk = current_chunk[
                            -remaining_space:
                        ]  # Keep the last 'remaining_space' tokens
                        chunks.append(
                            self.tokenizer.convert_tokens_to_string(current_chunk)
                        )
                        current_length = 0
                        current_chunk = []

                # Add the current sentence (or part of it) to the chunk
                if current_length + sent_length <= self.chunk_size:
                    current_chunk.extend(sent_tokens)
                    current_length += sent_length
                else:
                    # Split the sentence into smaller parts to fit the chunk
                    remaining_tokens = sent_tokens
                    while remaining_tokens:
                        if current_length >= self.chunk_size:
                            chunks.append(
                                self.tokenizer.convert_tokens_to_string(current_chunk)
                            )
                            current_chunk = []
                            current_length = 0
                        # Take a slice of tokens that fits in the remaining space
                        take = min(
                            self.chunk_size - current_length, len(remaining_tokens)
                        )
                        current_chunk.extend(remaining_tokens[:take])
                        current_length += take
                        remaining_tokens = remaining_tokens[take:]
                        if remaining_tokens:  # Apply overlap for the next chunk
                            overlap_tokens = remaining_tokens[: self.overlap]
                            current_chunk.extend(overlap_tokens)
                            current_length += len(overlap_tokens)
                            remaining_tokens = remaining_tokens[self.overlap :]

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(self.tokenizer.convert_tokens_to_string(current_chunk))

            logger.info(
                f"Sentence-based chunking: {len(chunks)} chunks from "
                f"{len(list(doc.sents))} sentences (size={self.chunk_size}, \
                    overlap={self.overlap})"
            )
            return chunks

        except Exception as e:
            logger.error(f"Sentence processing failed: {str(e)}")
            raise

    def _chunk_chinese_text(self, text: str) -> List[str]:
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
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk_size")
        if not text:
            logger.debug("Empty input text: returning empty chunks")
            return []

        try:
            # Use jieba for Chinese word segmentation
            words = list(jieba.cut(text))
            chunks = []
            start = 0

            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk = " ".join(
                    words[start:end]
                )  # Join words with spaces for better readability
                if chunk:  # Skip empty chunks
                    chunks.append(chunk)
                start += self.chunk_size - self.overlap  # Apply overlap

            logger.info(
                f"Chinese word-based chunking: {len(chunks)} chunks from {len(words)} \
                    words (size={self.chunk_size}, overlap={self.overlap})"
            )
            return chunks

        except Exception as e:
            logger.error(f"Chinese text processing failed: {str(e)}")
            raise

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes Unicode text (e.g., converts full-width to half-width characters).
        Useful for preprocessing text before chunking.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text string.
        """
        return unicodedata.normalize("NFKC", text)
