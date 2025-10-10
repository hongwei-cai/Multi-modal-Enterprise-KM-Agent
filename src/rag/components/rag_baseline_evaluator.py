#!/usr/bin/env python3
"""
RAG Baseline Evaluator

Evaluates RAG system performance using the knowledge base before fine-tuning.
Uses existing benchmarking components to establish baseline metrics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.benchmarker import Benchmarker
from src.rag.llm_client import LLMClient
from src.rag.retriever import Retriever

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGBaselineEvaluator:
    """
    Evaluates RAG system baseline performance using knowledge base.

    This evaluator measures retrieval and generation performance before
    any fine-tuning, establishing benchmarks for comparison.
    """

    def __init__(
        self,
        knowledge_base_collection: str = "knowledge_base",
        model_name: str = "microsoft/DialoGPT-medium",
        db_path: Optional[str] = None,
    ):
        """
        Initialize RAG baseline evaluator.

        Args:
            knowledge_base_collection: ChromaDB collection name
            model_name: LLM model for generation
            db_path: ChromaDB path
        """
        self.knowledge_base_collection = knowledge_base_collection
        self.model_name = model_name
        self.db_path = db_path

        # Initialize components
        db_path = db_path or str(
            Path(__file__).parent.parent.parent.parent / "chroma_db"
        )
        self.retriever = Retriever(
            db_path=db_path, collection_name=knowledge_base_collection
        )
        self.llm_client = LLMClient(model_name=model_name)

        # Initialize benchmarking
        self.benchmarker = Benchmarker()

        logger.info("Initialized RAG baseline evaluator with model: %s", model_name)

    def load_evaluation_dataset(self, eval_file: str) -> List[Dict[str, Any]]:
        """
        Load evaluation dataset from JSONL file.

        Args:
            eval_file: Path to evaluation dataset

        Returns:
            List of evaluation samples
        """
        eval_samples = []
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_samples.append(json.loads(line))

        logger.info(
            "Loaded %d evaluation samples from %s", len(eval_samples), eval_file
        )
        return eval_samples

    def evaluate_retrieval(self, queries: List[str], k: int = 5) -> Dict[str, Any]:
        """
        Evaluate retrieval performance.

        Args:
            queries: List of query strings
            k: Number of documents to retrieve

        Returns:
            Retrieval evaluation metrics
        """
        logger.info(f"Evaluating retrieval performance on {len(queries)} queries")

        retrieval_results = []
        total_retrieval_time = 0

        for query in queries:
            start_time = __import__("time").time()

            # Perform retrieval
            results = self.retriever.retrieve(query, top_k=k)

            retrieval_time = __import__("time").time() - start_time
            total_retrieval_time += retrieval_time

            retrieval_results.append(
                {
                    "query": query,
                    "retrieval_time": retrieval_time,
                    "num_results": len(results),
                    "results": results,
                }
            )

        # Calculate metrics
        avg_retrieval_time = total_retrieval_time / len(queries)
        success_rate = sum(1 for r in retrieval_results if r["num_results"] > 0) / len(
            queries
        )

        metrics = {
            "total_queries": len(queries),
            "avg_retrieval_time": avg_retrieval_time,
            "retrieval_success_rate": success_rate,
            "k": k,
            "results": retrieval_results,
        }

        logger.info(
            "Retrieval evaluation completed. Avg time: %.3f, Success rate: %.2f",
            avg_retrieval_time,
            success_rate,
        )
        return metrics

    def evaluate_generation(
        self, qa_pairs: List[Dict[str, Any]], max_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate generation performance on QA pairs.

        Args:
            qa_pairs: List of QA pairs for evaluation
            max_samples: Maximum number of samples to evaluate

        Returns:
            Generation evaluation metrics
        """
        logger.info(
            "Evaluating generation performance on %d samples",
            min(len(qa_pairs), max_samples),
        )

        # Sample QA pairs
        eval_pairs = qa_pairs[:max_samples]

        generation_results = []
        total_generation_time = 0

        for pair in eval_pairs:
            question = pair["question"]
            expected_answer = pair["answer"]

            # Generate answer
            start_time = __import__("time").time()

            try:
                generated_answer = self.llm_client.generate(
                    prompt=question, max_length=100, temperature=0.7
                )
                generation_time = __import__("time").time() - start_time
                total_generation_time += generation_time

                # Simple quality metrics
                answer_length = len(generated_answer.split())
                expected_length = len(expected_answer.split())

                generation_results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "generation_time": generation_time,
                        "answer_length": answer_length,
                        "expected_length": expected_length,
                        "success": True,
                    }
                )

            except Exception as e:
                generation_results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": "",
                        "generation_time": __import__("time").time() - start_time,
                        "error": str(e),
                        "success": False,
                    }
                )

        # Calculate metrics
        successful_generations = [r for r in generation_results if r["success"]]
        avg_generation_time = (
            total_generation_time / len(eval_pairs) if eval_pairs else 0
        )
        success_rate = (
            len(successful_generations) / len(eval_pairs) if eval_pairs else 0
        )

        if successful_generations:
            avg_answer_length = sum(
                r["answer_length"] for r in successful_generations
            ) / len(successful_generations)
            avg_expected_length = sum(
                r["expected_length"] for r in successful_generations
            ) / len(successful_generations)
        else:
            avg_answer_length = 0
            avg_expected_length = 0

        metrics = {
            "total_samples": len(eval_pairs),
            "successful_generations": len(successful_generations),
            "success_rate": success_rate,
            "avg_generation_time": avg_generation_time,
            "avg_answer_length": avg_answer_length,
            "avg_expected_length": avg_expected_length,
            "results": generation_results,
        }

        logger.info(".3f")
        return metrics

    def evaluate_rag_pipeline(
        self, qa_pairs: List[Dict[str, Any]], k: int = 3, max_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Evaluate complete RAG pipeline (retrieval + generation).

        Args:
            qa_pairs: QA pairs for evaluation
            k: Number of documents to retrieve
            max_samples: Maximum samples to evaluate

        Returns:
            Complete RAG evaluation metrics
        """
        logger.info(
            f"Evaluating complete RAG pipeline on {min(len(qa_pairs), max_samples)} samples"
        )

        eval_pairs = qa_pairs[:max_samples]
        rag_results = []

        for pair in eval_pairs:
            question = pair["question"]
            expected_answer = pair["answer"]

            # Step 1: Retrieval
            retrieval_start = __import__("time").time()
            retrieved_docs = self.retriever.retrieve(question, top_k=k)
            retrieval_time = __import__("time").time() - retrieval_start

            # Step 2: Generation with context
            context = "\n".join([doc["document"] for doc in retrieved_docs])
            prompt = "Context: %s\n\nQuestion: %s\n\nAnswer:" % (context, question)

            generation_start = __import__("time").time()
            try:
                generated_answer = self.llm_client.generate(
                    prompt=prompt, max_length=150, temperature=0.7
                )
                generation_time = __import__("time").time() - generation_start

                rag_results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "retrieved_context": context,
                        "num_retrieved_docs": len(retrieved_docs),
                        "retrieval_time": retrieval_time,
                        "generation_time": generation_time,
                        "total_time": retrieval_time + generation_time,
                        "success": True,
                    }
                )

            except Exception as e:
                rag_results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": "",
                        "retrieved_context": context,
                        "num_retrieved_docs": len(retrieved_docs),
                        "retrieval_time": retrieval_time,
                        "generation_time": __import__("time").time() - generation_start,
                        "total_time": retrieval_time
                        + (__import__("time").time() - generation_start),
                        "error": str(e),
                        "success": False,
                    }
                )

        # Calculate RAG metrics
        successful_rag = [r for r in rag_results if r["success"]]

        if successful_rag:
            avg_retrieval_time = sum(r["retrieval_time"] for r in successful_rag) / len(
                successful_rag
            )
            avg_generation_time = sum(
                r["generation_time"] for r in successful_rag
            ) / len(successful_rag)
            avg_total_time = sum(r["total_time"] for r in successful_rag) / len(
                successful_rag
            )
            avg_retrieved_docs = sum(
                r["num_retrieved_docs"] for r in successful_rag
            ) / len(successful_rag)
        else:
            avg_retrieval_time = 0
            avg_generation_time = 0
            avg_total_time = 0
            avg_retrieved_docs = 0

        metrics = {
            "total_samples": len(eval_pairs),
            "successful_rag": len(successful_rag),
            "success_rate": len(successful_rag) / len(eval_pairs) if eval_pairs else 0,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_generation_time": avg_generation_time,
            "avg_total_time": avg_total_time,
            "avg_retrieved_docs": avg_retrieved_docs,
            "k": k,
            "results": rag_results,
        }

        logger.info(".3f")
        return metrics

    def run_baseline_evaluation(
        self,
        eval_dataset_file: str,
        output_dir: str = "data/processed/baseline_evaluation",
    ) -> Dict[str, Any]:
        """
        Run complete baseline evaluation.

        Args:
            eval_dataset_file: Path to evaluation dataset
            output_dir: Output directory for results

        Returns:
            Complete evaluation results
        """
        logger.info("Starting RAG baseline evaluation...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load evaluation dataset
        eval_samples = self.load_evaluation_dataset(eval_dataset_file)

        # Run evaluations
        retrieval_metrics = self.evaluate_retrieval(
            [s["question"] for s in eval_samples]
        )
        generation_metrics = self.evaluate_generation(eval_samples)
        rag_metrics = self.evaluate_rag_pipeline(eval_samples)

        # Compile results
        evaluation_results = {
            "evaluation_config": {
                "model": self.model_name,
                "knowledge_base_collection": self.knowledge_base_collection,
                "eval_dataset": eval_dataset_file,
                "eval_samples": len(eval_samples),
            },
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "rag_metrics": rag_metrics,
            "summary": {
                "retrieval_success_rate": retrieval_metrics["retrieval_success_rate"],
                "generation_success_rate": generation_metrics["success_rate"],
                "rag_success_rate": rag_metrics["success_rate"],
                "avg_rag_total_time": rag_metrics["avg_total_time"],
            },
        }

        # Save results
        results_file = output_path / "baseline_evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        logger.info("Baseline evaluation completed. Results saved to: %s", results_file)

        # Print summary
        self._print_evaluation_summary(evaluation_results)

        return evaluation_results

    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("RAG BASELINE EVALUATION RESULTS")
        print("=" * 60)

        config = results["evaluation_config"]
        summary = results["summary"]

        print(f"Model: {config['model']}")
        print(f"Knowledge Base: {config['knowledge_base_collection']}")
        print(f"Evaluation Samples: {config['eval_samples']}")

        print("\n📊 PERFORMANCE METRICS:")
        print(
            "Retrieval Success Rate: %.1f%%" % (summary["retrieval_success_rate"] * 100)
        )
        print(
            "Generation Success Rate: %.1f%%"
            % (summary["generation_success_rate"] * 100)
        )
        print("RAG Success Rate: %.1f%%" % (summary["rag_success_rate"] * 100))
        print("Average RAG Total Time: %.3fs" % summary["avg_rag_total_time"])

        print("\n🔍 DETAILED METRICS:")

        ret_metrics = results["retrieval_metrics"]
        print("Retrieval - Avg Time: %.3fs" % ret_metrics["avg_retrieval_time"])
        print(
            "Retrieval - Success Rate: %.1f%%"
            % (ret_metrics["retrieval_success_rate"] * 100)
        )

        gen_metrics = results["generation_metrics"]
        print("Generation - Avg Time: %.3fs" % gen_metrics["avg_generation_time"])
        print("Generation - Avg BLEU: %.1f" % gen_metrics["avg_bleu_score"])
        print("Generation - Avg ROUGE-L: %.1f" % gen_metrics["avg_rouge_l_score"])

        rag_metrics = results["rag_metrics"]
        print("RAG - Avg Retrieval Time: %.3fs" % rag_metrics["avg_retrieval_time"])
        print("RAG - Avg Generation Time: %.3fs" % rag_metrics["avg_generation_time"])
        print("RAG - Avg Total Time: %.3fs" % rag_metrics["avg_total_time"])
        print("RAG - Avg Retrieved Docs: %.1f" % rag_metrics["avg_retrieved_docs"])

        print("\n" + "=" * 60)


def run_baseline_evaluation(
    eval_dataset: str = "data/processed/test_auto_topics/test.jsonl",
    model_name: str = "microsoft/DialoGPT-medium",
    collection_name: str = "knowledge_base",
    output_dir: str = "data/processed/baseline_evaluation",
) -> Dict[str, Any]:
    """
    Convenience function to run baseline RAG evaluation.

    Args:
        eval_dataset: Path to evaluation dataset
        model_name: LLM model name
        collection_name: Knowledge base collection name
        output_dir: Output directory

    Returns:
        Evaluation results
    """
    evaluator = RAGBaselineEvaluator(
        knowledge_base_collection=collection_name, model_name=model_name
    )

    return evaluator.run_baseline_evaluation(eval_dataset, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG baseline evaluation")
    parser.add_argument(
        "--eval_dataset",
        default="data/processed/test_auto_topics/test.jsonl",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--model_name", default="microsoft/DialoGPT-medium", help="LLM model name"
    )
    parser.add_argument(
        "--collection_name",
        default="knowledge_base",
        help="Knowledge base collection name",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/baseline_evaluation",
        help="Output directory",
    )

    args = parser.parse_args()

    try:
        results = run_baseline_evaluation(
            eval_dataset=args.eval_dataset,
            model_name=args.model_name,
            collection_name=args.collection_name,
            output_dir=args.output_dir,
        )
        print("Baseline evaluation completed successfully!")
    except Exception as e:
        logger.error("Baseline evaluation failed: %s", e)
        import traceback

        traceback.print_exc()
        exit(1)
