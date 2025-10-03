"""
RAG prompt template for context-aware question answering.
"""
from typing import Any, Dict, List, Optional


class PromptTemplate:
    """
    Template for RAG prompts with system role, context, and question.
    """

    def __init__(
        self,
        system_role: str = "You are a helpful assistant. Answer the question \
            based on the provided context. If you don't know the answer, say \
                \"I don't know\".",
        template: Optional[str] = None,
    ):
        self.system_role = system_role
        if template is None:
            template = "Answer the question based on the context.\
                \n\nQuestion: {question}\n\nContext: {context}"
        self.template = template

    def format_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        max_context_length: int = 500,
    ) -> str:
        """
        Format the prompt with context and question.

        Args:
            question: User question.
            context_docs: List of retrieved documents (dicts with 'document' key).
            max_context_length: Max length of context to include.

        Returns:
            Formatted prompt string.
        """
        # Extract and truncate context
        context_texts = [doc["document"] for doc in context_docs]
        context = "\n".join(context_texts)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # Format the prompt
        prompt = self.template.format(context=context, question=question)
        return prompt


# Convenience function
def get_prompt_template(system_role: Optional[str] = None) -> PromptTemplate:
    """
    Get a PromptTemplate instance.

    Args:
        system_role: Custom system role (uses default if None).

    Returns:
        PromptTemplate instance.
    """
    if system_role is None:
        return PromptTemplate()
    return PromptTemplate(system_role=system_role)
