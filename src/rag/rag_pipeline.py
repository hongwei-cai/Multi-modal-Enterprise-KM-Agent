"""
End-to-end RAG pipeline: retrieve → prompt → generate.
"""
import logging
import time
from typing import Optional

from .experiment_tracker import (
    ExperimentConfig,
    MLflowExperimentTracker,
    PromptResponseLog,
    RetrievalMetrics,
)
from .llm_client import get_llm_client
from .prompt_template import get_prompt_template
from .retriever import get_retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for question answering with\
        comprehensive experiment tracking.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        top_k: int = 5,
        max_context_length: int = 2000,
        system_role: Optional[str] = None,
        experiment_tracker: Optional[MLflowExperimentTracker] = None,
        experiment_name: str = "rag_pipeline",
    ):
        self.retriever = get_retriever(
            db_path=db_path, top_k=top_k, experiment_tracker=experiment_tracker
        )
        self.prompt_template = get_prompt_template(system_role=system_role)
        self.max_context_length = max_context_length
        self.experiment_tracker = experiment_tracker
        self.experiment_name = experiment_name
        self.current_run_id: Optional[str] = None
        self.llm_client = get_llm_client(
            experiment_tracker=experiment_tracker, run_id=self.current_run_id
        )

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:  # Change return type to dict
        """
        Answer a question using RAG with comprehensive experiment tracking.

        Args:
            question: User question.
            top_k: Number of context docs to retrieve.
            temperature: Generation temperature.
            top_p: Generation top-p.

        Returns:
            Generated answer with metadata.
        """
        start_time = time.time()
        step = 0

        # Initialize experiment run if tracker is available
        if self.experiment_tracker and not self.current_run_id:
            config = ExperimentConfig(
                experiment_name=self.experiment_name,
                run_name=f"rag_query_{int(time.time())}",
                model_name="rag_pipeline",
                model_version="v1.0",
                parameters={
                    "top_k": top_k or self.retriever.top_k,
                    "max_context_length": self.max_context_length,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            self.current_run_id = self.experiment_tracker.start_experiment(config)

        # Log initial system resources
        if self.experiment_tracker and self.current_run_id:
            system_metrics = self.experiment_tracker.get_current_system_resources()
            self.experiment_tracker.log_system_resources(
                self.current_run_id, system_metrics, step
            )

        # Step 1: Retrieve context
        retrieval_start = time.time()
        context_docs = self.retriever.retrieve(question, top_k=top_k)
        retrieval_latency = (time.time() - retrieval_start) * 1000

        if not context_docs:
            # Log failed retrieval
            if self.experiment_tracker and self.current_run_id:
                retrieval_metrics = RetrievalMetrics(
                    query=question,
                    retrieved_docs_count=0,
                    retrieval_latency_ms=retrieval_latency,
                )
                self.experiment_tracker.log_retrieval_metrics(
                    self.current_run_id, retrieval_metrics, step
                )
                self.experiment_tracker.end_experiment(self.current_run_id)
                self.current_run_id = None

            return {"answer": "No relevant context found.", "context_docs": []}

        # Log retrieval metrics
        if self.experiment_tracker and self.current_run_id:
            retrieval_metrics = RetrievalMetrics(
                query=question,
                retrieved_docs_count=len(context_docs),
                retrieval_latency_ms=retrieval_latency,
            )
            self.experiment_tracker.log_retrieval_metrics(
                self.current_run_id, retrieval_metrics, step
            )
            step += 1

        # Step 2: Format prompt with context truncation
        prompt = self.prompt_template.format_prompt(
            question=question,
            context_docs=context_docs,
            max_context_length=self.max_context_length,
        )

        # Step 3: Generate answer
        generation_start = time.time()
        answer = self.llm_client.generate(
            prompt=prompt, temperature=temperature, top_p=top_p
        )
        generation_latency = (time.time() - generation_start) * 1000

        # Log prompt and response
        if self.experiment_tracker and self.current_run_id:
            prompt_log = PromptResponseLog(
                prompt=prompt,
                response=answer,
                prompt_template=self.prompt_template.template,
                prompt_parameters={
                    "question": question,
                    "context_docs_count": len(context_docs),
                    "max_context_length": self.max_context_length,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                response_metadata={
                    "generation_latency_ms": generation_latency,
                    "response_length": len(answer),
                },
            )
            self.experiment_tracker.log_prompt_response(
                self.current_run_id, prompt_log, step
            )
            step += 1

        # Log final system resources
        if self.experiment_tracker and self.current_run_id:
            system_metrics = self.experiment_tracker.get_current_system_resources()
            self.experiment_tracker.log_system_resources(
                self.current_run_id, system_metrics, step
            )

            # End experiment
            self.experiment_tracker.end_experiment(self.current_run_id)
            self.current_run_id = None

        total_latency = (time.time() - start_time) * 1000
        logger.info(
            "Answered question: %s... (total latency: %.2fms)",
            question[:50],
            total_latency,
        )

        return {
            "answer": answer,
            "context_docs": [doc["document"] for doc in context_docs],
            "metadata": {
                "retrieval_latency_ms": retrieval_latency,
                "generation_latency_ms": generation_latency,
                "total_latency_ms": total_latency,
                "retrieved_docs_count": len(context_docs),
            },
        }


# Convenience function
def get_rag_pipeline(
    db_path: Optional[str] = None,
    top_k: int = 5,
    max_context_length: int = 2000,
    system_role: Optional[str] = None,
    experiment_tracker: Optional[MLflowExperimentTracker] = None,
    experiment_name: str = "rag_pipeline",
) -> RAGPipeline:
    """
    Get a RAGPipeline instance with optional experiment tracking.

    Args:
        db_path: Vector DB path.
        top_k: Retrieval top-k.
        max_context_length: Max context length.
        system_role: System role for prompts.
        experiment_tracker: Optional MLflow experiment tracker.
        experiment_name: Name for experiments.

    Returns:
        RAGPipeline instance.
    """
    return RAGPipeline(
        db_path=db_path,
        top_k=top_k,
        max_context_length=max_context_length,
        system_role=system_role,
        experiment_tracker=experiment_tracker,
        experiment_name=experiment_name,
    )
