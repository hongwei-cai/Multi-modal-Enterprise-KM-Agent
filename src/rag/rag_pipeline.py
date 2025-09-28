"""
End-to-end RAG pipeline: retrieve → prompt → generate.
"""
import logging
from typing import Optional

from .llm_client import get_llm_client
from .prompt_template import get_prompt_template
from .retriever import get_retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for question answering.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        top_k: int = 5,
        max_context_length: int = 2000,
        system_role: Optional[str] = None,
    ):
        self.retriever = get_retriever(db_path=db_path, top_k=top_k)
        self.prompt_template = get_prompt_template(system_role=system_role)
        self.llm_client = get_llm_client()
        self.max_context_length = max_context_length

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Answer a question using RAG.

        Args:
            question: User question.
            top_k: Number of context docs to retrieve.
            temperature: Generation temperature.
            top_p: Generation top-p.

        Returns:
            Generated answer.
        """
        # Step 1: Retrieve context
        context_docs = self.retriever.retrieve(question, top_k=top_k)
        if not context_docs:
            return "No relevant context found."

        # Step 2: Format prompt with context truncation
        prompt = self.prompt_template.format_prompt(
            question=question,
            context_docs=context_docs,
            max_context_length=self.max_context_length,
        )

        # Step 3: Generate answer
        answer = self.llm_client.generate(
            prompt=prompt, temperature=temperature, top_p=top_p
        )

        logger.info(f"Answered question: {question[:50]}...")
        return answer


# Convenience function
def get_rag_pipeline(
    db_path: Optional[str] = None,
    top_k: int = 5,
    max_context_length: int = 2000,
    system_role: Optional[str] = None,
) -> RAGPipeline:
    """
    Get a RAGPipeline instance.

    Args:
        db_path: Vector DB path.
        top_k: Retrieval top-k.
        max_context_length: Max context length.
        system_role: System role for prompts.

    Returns:
        RAGPipeline instance.
    """
    return RAGPipeline(
        db_path=db_path,
        top_k=top_k,
        max_context_length=max_context_length,
        system_role=system_role,
    )
