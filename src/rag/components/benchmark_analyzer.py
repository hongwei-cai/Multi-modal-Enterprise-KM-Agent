"""
Benchmark Analyzer Component

This streamlined component analyzes and visualizes benchmark results
from MLflow experiments.
It provides comprehensive analysis of model comparison results
with plots and statistics.
Can be used in notebooks, scripts, or APIs for any model comparison scenario.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.rag.experiment_tracker import MLflowExperimentTracker


class BenchmarkAnalyzer:
    """
    Streamlined component for analyzing and visualizing benchmark results.

    This component extracts metrics from MLflow experiments, performs statistical
    analysis, creates visualizations, and provides comprehensive comparison reports.
    """

    def __init__(self, mlruns_dir: Optional[str] = None):
        """
        Initialize the benchmark analyzer.

        Args:
            mlruns_dir: Directory containing MLflow runs (defaults to env var)
        """
        if mlruns_dir is None:
            project_root = Path(os.getenv("PROJECT_ROOT", "..")).resolve()
            mlruns_dir = str(project_root / "mlruns")

        Path(mlruns_dir).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("file:" + mlruns_dir)

        self.mlruns_dir = mlruns_dir
        self.experiment_tracker = MLflowExperimentTracker()

    def analyze_experiment(
        self,
        experiment_name: str,
        baseline_model_name: str = "Baseline",
        fine_tuned_model_name: str = "Fine-tuned",
        show_plots: bool = True,
        save_plots: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze benchmark results from an MLflow experiment.

        Args:
            experiment_name: Name of the MLflow experiment to analyze
            baseline_model_name: Display name for baseline model
            fine_tuned_model_name: Display name for fine-tuned model
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots (defaults to mlruns_dir)

        Returns:
            Dictionary containing analysis results and statistics
        """
        # Get the experiment
        experiment = self._get_experiment(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

        # Extract metrics
        baseline_metrics, finetuned_metrics = self._extract_metrics(runs)

        if not baseline_metrics["latencies"] or not finetuned_metrics["latencies"]:
            raise ValueError("Could not find both baseline and fine-tuned metrics")

        # Calculate statistics
        results = self._calculate_statistics(baseline_metrics, finetuned_metrics)

        # Create visualizations
        if show_plots or save_plots:
            plots = self._create_plots(
                baseline_metrics,
                finetuned_metrics,
                baseline_model_name,
                fine_tuned_model_name,
                show_plots,
                save_plots,
                output_dir,
            )
            results["plots"] = plots

        # Print summary
        self._print_summary(results, baseline_model_name, fine_tuned_model_name)

        return results

    def _get_experiment(self, experiment_name: str):
        """Get MLflow experiment by name with error handling."""
        try:
            experiments = mlflow.search_experiments()
            for exp in experiments:
                if exp.name == experiment_name:
                    return exp
        except Exception as e:
            print(f"Warning: Error searching experiments: {e}")

        # Fallback: try direct access
        try:
            return mlflow.get_experiment_by_name(experiment_name)
        except Exception:
            return None

    def _extract_metrics(
        self, runs: pd.DataFrame
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Extract baseline and fine-tuned metrics from runs."""
        baseline_latencies = []
        baseline_response_qualities = []
        finetuned_latencies = []
        finetuned_response_qualities = []

        for idx, run in runs.iterrows():
            # Extract baseline metrics
            if pd.notna(run.get("metrics.baseline_latency_ms")):
                baseline_latencies.append(
                    run["metrics.baseline_latency_ms"] / 1000
                )  # Convert to seconds
            if pd.notna(run.get("metrics.baseline_response_quality_score")):
                baseline_response_qualities.append(
                    run["metrics.baseline_response_quality_score"] * 100
                )  # Convert to percentage

            # Extract fine-tuned metrics
            if pd.notna(run.get("metrics.finetuned_latency_ms")):
                finetuned_latencies.append(
                    run["metrics.finetuned_latency_ms"] / 1000
                )  # Convert to seconds
            if pd.notna(run.get("metrics.finetuned_response_quality_score")):
                finetuned_response_qualities.append(
                    run["metrics.finetuned_response_quality_score"] * 100
                )  # Convert to percentage

        baseline_metrics = {
            "latencies": baseline_latencies,
            "response_qualities": baseline_response_qualities,
        }

        finetuned_metrics = {
            "latencies": finetuned_latencies,
            "response_qualities": finetuned_response_qualities,
        }

        return baseline_metrics, finetuned_metrics

    def _calculate_statistics(
        self, baseline_metrics: Dict, finetuned_metrics: Dict
    ) -> Dict[str, Any]:
        """Calculate statistical summaries and improvements."""
        # Convert to numpy arrays
        baseline_latency = np.array(baseline_metrics["latencies"])
        baseline_response_quality = np.array(baseline_metrics["response_qualities"])
        finetuned_latency = np.array(finetuned_metrics["latencies"])
        finetuned_response_quality = np.array(finetuned_metrics["response_qualities"])

        # Calculate improvements
        latency_improvement = (
            (np.mean(baseline_latency) - np.mean(finetuned_latency))
            / np.mean(baseline_latency)
        ) * 100
        quality_improvement = (
            (np.mean(finetuned_response_quality) - np.mean(baseline_response_quality))
            / np.mean(baseline_response_quality)
        ) * 100

        return {
            "baseline": {
                "latency_mean": np.mean(baseline_latency),
                "latency_std": np.std(baseline_latency),
                "response_quality_mean": np.mean(baseline_response_quality),
                "response_quality_std": np.std(baseline_response_quality),
                "sample_count": len(baseline_latency),
            },
            "finetuned": {
                "latency_mean": np.mean(finetuned_latency),
                "latency_std": np.std(finetuned_latency),
                "response_quality_mean": np.mean(finetuned_response_quality),
                "response_quality_std": np.std(finetuned_response_quality),
                "sample_count": len(finetuned_latency),
            },
            "improvements": {
                "latency_percent": latency_improvement,
                "quality_percent": quality_improvement,
                "latency_direction": "faster" if latency_improvement > 0 else "slower",
                "quality_direction": "better" if quality_improvement > 0 else "worse",
            },
        }

    def _create_plots(
        self,
        baseline_metrics: Dict,
        finetuned_metrics: Dict,
        baseline_name: str,
        finetuned_name: str,
        show_plots: bool,
        save_plots: bool,
        output_dir: Optional[str],
    ) -> Dict[str, plt.Figure]:
        """Create comparison plots."""
        # Suppress matplotlib warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Latency plot
            baseline_latency = np.array(baseline_metrics["latencies"])
            finetuned_latency = np.array(finetuned_metrics["latencies"])

            ax1.bar(
                [baseline_name, finetuned_name],
                [np.mean(baseline_latency), np.mean(finetuned_latency)],
                yerr=[np.std(baseline_latency), np.std(finetuned_latency)],
                capsize=5,
                color=["skyblue", "lightcoral"],
            )
            ax1.set_ylabel("Latency (seconds)")
            ax1.set_title("Response Latency Comparison")
            ax1.grid(True, alpha=0.3)

            # Response quality plot
            baseline_quality = np.array(baseline_metrics["response_qualities"])
            finetuned_quality = np.array(finetuned_metrics["response_qualities"])

            ax2.bar(
                [baseline_name, finetuned_name],
                [np.mean(baseline_quality), np.mean(finetuned_quality)],
                yerr=[np.std(baseline_quality), np.std(finetuned_quality)],
                capsize=5,
                color=["skyblue", "lightcoral"],
            )
            ax2.set_ylabel("Success Rate (%)")
            ax2.set_title("Response Quality Comparison")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            plots = {"comparison": fig}

            if save_plots:
                if output_dir is None:
                    output_dir = self.mlruns_dir

                output_path = Path(output_dir) / "benchmark_analysis.png"
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"Plot saved to: {output_path}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig)

            return plots

    def _print_summary(self, results: Dict, baseline_name: str, finetuned_name: str):
        """Print analysis summary."""
        baseline = results["baseline"]
        finetuned = results["finetuned"]
        improvements = results["improvements"]

        print("Benchmark Results Summary:")
        print(f"{baseline_name}:")
        print(
            f"  Average Latency: {baseline['latency_mean']:.2f}s ± "
            f"{baseline['latency_std']:.2f}s"
        )
        print(
            f"  Success Rate: {baseline['response_quality_mean']:.1f}% ± "
            f"{baseline['response_quality_std']:.1f}%"
        )
        print(f"  Sample Count: {baseline['sample_count']}")
        print(f"{finetuned_name}:")
        print(
            f"  Average Latency: {finetuned['latency_mean']:.2f}s ± "
            f"{finetuned['latency_std']:.2f}s"
        )
        print(
            f"  Success Rate: {finetuned['response_quality_mean']:.1f}% ± "
            f"{finetuned['response_quality_std']:.1f}%"
        )
        print(f"  Sample Count: {finetuned['sample_count']}")

        print("\nImprovements:")
        print(
            f"  Latency: {improvements['latency_percent']:+.1f}% "
            f"({improvements['latency_direction']})"
        )
        print(
            f"  Success Rate: {improvements['quality_percent']:+.1f}% "
            f"({improvements['quality_direction']})"
        )


def get_benchmark_analyzer(mlruns_dir: Optional[str] = None) -> BenchmarkAnalyzer:
    """
    Factory function to get a BenchmarkAnalyzer instance.

    Args:
        mlruns_dir: Directory containing MLflow runs

    Returns:
        Configured BenchmarkAnalyzer instance
    """
    return BenchmarkAnalyzer(mlruns_dir=mlruns_dir)
