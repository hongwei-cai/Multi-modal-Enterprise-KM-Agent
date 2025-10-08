"""
Benchmarking Workflow Component

This streamlined component orchestrates the complete benchmarking workflow,
combining the Benchmarker building block with experiment tracking and result
presentation. Can be used in notebooks, scripts, or APIs.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import psutil

from configs.model_config import BenchmarkSummary
from src.rag.benchmarker import get_benchmarker
from src.rag.experiment_tracker import (
    ExperimentConfig,
    MLflowExperimentTracker,
    PerformanceMetrics,
)


class BenchmarkingWorkflow:
    """
    Streamlined component for complete model benchmarking workflow.

    This orchestrates the benchmarking process from model comparison through
    result logging and presentation, using the Benchmarker building block.
    """

    def __init__(
        self, mlruns_dir: Optional[str] = None, experiment_name: str = "model_benchmark"
    ):
        """
        Initialize the benchmarking workflow.

        Args:
            mlruns_dir: Directory for MLflow runs (defaults to env var)
            experiment_name: Name for the MLflow experiment
        """
        # Set up MLflow
        if mlruns_dir is None:
            project_root = Path(os.getenv("PROJECT_ROOT", "..")).resolve()
            mlruns_dir = str(project_root / "mlruns")

        Path(mlruns_dir).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("file:" + mlruns_dir)

        self.experiment_name = experiment_name
        self.experiment_tracker = MLflowExperimentTracker()
        self.benchmarker = get_benchmarker()

    def run_comparison_benchmark(
        self,
        baseline_model: str,
        fine_tuned_model: str,
        adapter_path: str,
        test_queries: List[str],
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete comparison benchmark between baseline and fine-tuned models.

        Args:
            baseline_model: Name/path of baseline model
            fine_tuned_model: Name/path of base model for fine-tuned version
            adapter_path: Path to LoRA adapter
            test_queries: List of queries to test
            run_name: Optional name for this benchmark run

        Returns:
            Dictionary with results, metrics, and sample responses
        """
        if run_name is None:
            run_name = f"benchmark_{int(time.time())}"

        print("🚀 Starting A/B Benchmarking...")
        print("=" * 60)

        # Run benchmarks using Benchmarker building block
        comparison_results = self.benchmarker.compare_models(
            baseline_model=baseline_model,
            fine_tuned_model=fine_tuned_model,
            adapter_path=adapter_path,
            queries=test_queries,
        )

        baseline_results = comparison_results["baseline"]
        finetuned_results = comparison_results["finetuned"]

        if not baseline_results or not finetuned_results:
            return {
                "success": False,
                "error": "Benchmarking failed. Check model loading and adapter paths.",
            }

        # Calculate metrics
        metrics = self._calculate_comparison_metrics(
            baseline_results, finetuned_results
        )

        # Log to MLflow
        run_id = self._log_to_mlflow(
            baseline_model, adapter_path, test_queries, metrics, run_name
        )

        # Prepare sample responses
        sample_responses = self._get_sample_responses(
            test_queries, baseline_results, finetuned_results
        )

        return {
            "success": True,
            "baseline_results": baseline_results,
            "finetuned_results": finetuned_results,
            "metrics": metrics,
            "sample_responses": sample_responses,
            "mlflow_run_id": run_id,
        }

    def _calculate_comparison_metrics(
        self, baseline_results: BenchmarkSummary, finetuned_results: BenchmarkSummary
    ) -> Dict[str, Any]:
        """Calculate comparison metrics between baseline and fine-tuned results."""
        # Calculate improvements
        latency_change = (
            (baseline_results.avg_latency - finetuned_results.avg_latency)
            / baseline_results.avg_latency
        ) * 100

        length_change = (
            (
                finetuned_results.avg_response_length
                - baseline_results.avg_response_length
            )
            / baseline_results.avg_response_length
        ) * 100

        return {
            "baseline": {
                "latency": baseline_results.avg_latency,
                "response_length": baseline_results.avg_response_length,
                "success_rate": baseline_results.success_rate,
            },
            "finetuned": {
                "latency": finetuned_results.avg_latency,
                "response_length": finetuned_results.avg_response_length,
                "success_rate": finetuned_results.success_rate,
            },
            "improvements": {
                "latency_percent": latency_change,
                "response_length_percent": length_change,
            },
        }

    def _log_to_mlflow(
        self,
        baseline_model: str,
        adapter_path: str,
        test_queries: List[str],
        metrics: Dict[str, Any],
        run_name: str,
    ) -> str:
        """Log benchmarking results to MLflow."""
        print("\n📝 Logging results to MLflow...")

        # Start experiment
        experiment_config = ExperimentConfig(
            experiment_name=self.experiment_name,
            run_name=run_name,
            model_name=f"{baseline_model}_vs_lora",
            model_version="comparison",
            parameters={
                "baseline_model": baseline_model,
                "fine_tuned_adapter": os.path.basename(adapter_path),
                "num_queries": len(test_queries),
            },
        )

        run_id = self.experiment_tracker.start_experiment(experiment_config)

        # Log baseline metrics
        baseline_metrics = PerformanceMetrics(
            latency_ms=metrics["baseline"]["latency"] * 1000,
            memory_usage_mb=self._calculate_avg_memory(metrics["baseline"]),
            cpu_usage_percent=psutil.cpu_percent(),
            response_quality_score=metrics["baseline"]["success_rate"],
        )
        self.experiment_tracker.log_metrics(
            run_id, baseline_metrics, prefix="baseline_"
        )

        # Log fine-tuned metrics
        finetuned_metrics = PerformanceMetrics(
            latency_ms=metrics["finetuned"]["latency"] * 1000,
            memory_usage_mb=self._calculate_avg_memory(metrics["finetuned"]),
            cpu_usage_percent=psutil.cpu_percent(),
            response_quality_score=metrics["finetuned"]["success_rate"],
        )
        self.experiment_tracker.log_metrics(
            run_id, finetuned_metrics, prefix="finetuned_"
        )

        self.experiment_tracker.end_experiment(run_id)
        print("✓ Benchmarking complete! Results logged to MLflow.")

        return run_id

    def _calculate_avg_memory(self, model_metrics: Dict[str, Any]) -> float:
        """Calculate average memory usage (placeholder - would need
        actual memory tracking)."""
        # This would need to be implemented based on how memory is tracked
        return 0.0  # Placeholder

    def _get_sample_responses(
        self,
        test_queries: List[str],
        baseline_results: BenchmarkSummary,
        finetuned_results: BenchmarkSummary,
    ) -> List[Dict[str, str]]:
        """Get sample responses for the first few queries."""
        samples = []
        num_samples = min(2, len(test_queries))

        for i in range(num_samples):
            samples.append(
                {
                    "query": test_queries[i],
                    "baseline_response": baseline_results.results[i]["response"][:100]
                    + "...",
                    "finetuned_response": finetuned_results.results[i]["response"][:100]
                    + "...",
                }
            )

        return samples

    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted summary of benchmarking results.

        Args:
            results: Results dictionary from run_comparison_benchmark
        """
        if not results["success"]:
            print(f"❌ {results['error']}")
            return

        metrics = results["metrics"]

        print("\n🎯 BENCHMARK RESULTS COMPARISON")
        print("=" * 60)

        baseline_model = "microsoft/DialoGPT-medium"  # Would be parameterized

        print(f"Baseline Model ({baseline_model}):")
        print(f"  Average Latency: {metrics['baseline']['latency']:.2f}s")
        print(
            f"  Average Response Length: \
              {metrics['baseline']['response_length']:.1f} words"
        )
        print(f"  Success Rate: {metrics['baseline']['success_rate']:.1%}")

        print(f"\nFine-tuned Model ({baseline_model} + LoRA):")
        print(f"  Average Latency: {metrics['finetuned']['latency']:.2f}s")
        print(
            f"  Average Response Length: "
            f"{metrics['finetuned']['response_length']:.1f} words"
        )
        print(f"  Success Rate: {metrics['finetuned']['success_rate']:.1%}")

        latency_improvement = metrics["improvements"]["latency_percent"]
        response_improvement = metrics["improvements"]["response_length_percent"]

        print("\n📈 IMPROVEMENTS:")
        print(
            f"  Latency: {latency_improvement:+.1f}% "
            f"({'faster' if latency_improvement > 0 else 'slower'})"
        )

        print(
            f"  Response Length: {response_improvement:+.1f}% "
            f"({'longer' if response_improvement > 0 else 'shorter'})"
        )

        # Print sample responses
        print("\n💬 SAMPLE RESPONSES:")
        print("-" * 40)
        for sample in results["sample_responses"]:
            print(f"\nQuery: {sample['query']}")
            print(f"Baseline: {sample['baseline_response']}")
            print(f"Fine-tuned: {sample['finetuned_response']}")


def get_benchmarking_workflow(
    mlruns_dir: Optional[str] = None, experiment_name: str = "model_benchmark"
) -> BenchmarkingWorkflow:
    """
    Factory function to get a BenchmarkingWorkflow instance.

    Args:
        mlruns_dir: Directory for MLflow runs
        experiment_name: Name for the MLflow experiment

    Returns:
        Configured BenchmarkingWorkflow instance
    """
    return BenchmarkingWorkflow(mlruns_dir=mlruns_dir, experiment_name=experiment_name)
