"""
Automated QA Pair Generation Pipeline

Uses existing RAG building blocks to: parse documents, chunk text,
generate question-answer pairs with the LLM, validate answers using
embedding-based relevance, and export high-quality QA pairs.

Designed to integrate with the project's `LLMClient`, `TextChunker`, and
embedding utilities.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..document_parser import parse_pdf, parse_txt
from ..embedding import get_embedding_model
from ..llm_client import get_llm_client
from ..text_chunker import TextChunker

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    # lightweight cosine similarity
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class QAPairGenerator:
    """Generate, validate, filter, and save QA pairs from documents."""

    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # LLM client (defaults to project default, usually DialoGPT-medium)
        self.llm = get_llm_client(model_name=llm_model)

        # Embedding model (SentenceTransformer)
        self.embedding_model = get_embedding_model()

    def _parse_file(self, file_path: str) -> str:
        if file_path.lower().endswith(".pdf"):
            return parse_pdf(file_path)
        elif file_path.lower().endswith(".txt"):
            return parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _prompt_for_qa(self, passage: str) -> str:
        prompt = (
            "You are a helpful assistant. Given the passage below, "
            "generate ONE clear, answerable question and a concise answer "
            "(1-2 sentences). "
            "Format the output exactly as:\n"
            "Question: <question>\n"
            "Answer: <answer>\n"
            "Only output the question and answer. Passage:\n\n" + passage
        )
        return prompt

    def _extract_qa_from_generation(
        self, gen_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        q = None
        a = None
        try:
            # Simple parsing: look for lines starting with Question: and Answer:
            for line in gen_text.splitlines():
                if line.strip().lower().startswith("question:"):
                    q = line.split(":", 1)[1].strip()
                elif line.strip().lower().startswith("answer:"):
                    a = line.split(":", 1)[1].strip()

            # Fallback: if the model returned two lines without labels
            if q is None or a is None:
                parts = [part.strip() for part in gen_text.splitlines() if part.strip()]
                if len(parts) >= 2:
                    # Assume first is question, second is answer
                    if q is None:
                        q = parts[0]
                    if a is None:
                        a = parts[1]

        except Exception:
            return None, None

        return q, a

    def _validate_pair(
        self, question: str, answer: str, passage: str, relevance_threshold: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate QA pair using simple heuristics and embedding relevance."""
        metadata: Dict[str, Any] = {}

        # Clarity: question should contain a question mark and be reasonably short
        clarity_ok = bool(
            question and "?" in question and 5 <= len(question.split()) <= 40
        )
        metadata["clarity_ok"] = clarity_ok
        metadata["question_len_words"] = len(question.split()) if question else 0

        # Answer presence: check if answer substring appears in passage
        answer_in_passage = False
        if answer and passage:
            answer_in_passage = answer.lower() in passage.lower()
        metadata["answer_in_passage"] = answer_in_passage

        # Embedding-based relevance: compute similarity between answer and passage
        try:
            ans_emb = self.embedding_model.encode(answer)
            pass_emb = self.embedding_model.encode(passage)
            relevance = _cosine_similarity(
                ans_emb.tolist() if hasattr(ans_emb, "tolist") else list(ans_emb),
                pass_emb.tolist() if hasattr(pass_emb, "tolist") else list(pass_emb),
            )
        except Exception:
            relevance = 0.0

        metadata["relevance_score"] = float(relevance)

        ok = clarity_ok and (relevance >= relevance_threshold)
        # If answer is explicitly in passage, be more permissive
        if answer_in_passage and relevance >= (relevance_threshold - 0.05):
            ok = ok or True

        return ok, metadata

    def generate_from_files(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        overlap: int = 128,
        max_pairs: int = 1000,
        max_q_per_chunk: int = 1,
        relevance_threshold: float = 0.6,
        output_path: str = "data/processed/qa_pairs.jsonl",
    ) -> Path:
        """Main entrypoint: generate QA pairs from a list of document paths.

        Returns path to the saved JSONL file containing generated QA pairs.
        """
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with outp.open("w", encoding="utf-8") as fo:
            for file_path in file_paths:
                try:
                    text = self._parse_file(file_path)
                except Exception as e:
                    logger.warning("Skipping %s: %s", file_path, e)
                    continue

                chunker = TextChunker(
                    chunk_size=chunk_size, overlap=overlap, strategy="sentences"
                )
                chunks = chunker.chunk_text(text)
                if not chunks:
                    continue

                for idx, chunk in enumerate(
                    tqdm(chunks, desc=f"Chunks from {Path(file_path).name}")
                ):
                    # Stop if reached the target
                    if written >= max_pairs:
                        logger.info("Reached target of %d pairs", max_pairs)
                        return outp

                    # Generate up to max_q_per_chunk questions per chunk
                    for _ in range(max_q_per_chunk):
                        prompt = self._prompt_for_qa(chunk)
                        gen = self.llm.generate(
                            prompt=prompt, max_length=128, temperature=0.7, top_p=0.9
                        )
                        question, answer = self._extract_qa_from_generation(gen)
                        if not question or not answer:
                            continue

                        ok, meta = self._validate_pair(
                            question, answer, chunk, relevance_threshold
                        )
                        record = {
                            "file_path": file_path,
                            "chunk_index": idx,
                            "question": question,
                            "answer": answer,
                            "accepted": bool(ok),
                            "metrics": meta,
                        }

                        fo.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1

                        if written >= max_pairs:
                            logger.info("Reached target of %d pairs", max_pairs)
                            return outp

        logger.info("Generation complete. Wrote %d QA pairs to %s", written, outp)
        return outp


def get_qa_generator() -> QAPairGenerator:
    return QAPairGenerator()
