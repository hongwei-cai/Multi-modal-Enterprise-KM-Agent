"""
Benchmark Analyzer Component

This streamlined component analyzes and visualizes benchmark results
from MLflow experiments.
It provides comprehensive analysis of model comparison results
with plots and statistics.
Can be used in notebooks, scripts, or APIs for any model comparison scenario.
"""

import gc
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        baseline_model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
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
            baseline_model_path: Path/name of baseline model for parameter analysis
            adapter_path: Path to LoRA adapter for parameter analysis
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

        # Add memory and parameter analysis if paths provided
        if baseline_model_path:
            memory_param_analysis = self._analyze_memory_and_parameters(
                baseline_model_path, adapter_path
            )
            results["memory_parameters"] = memory_param_analysis

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

    def _analyze_memory_and_parameters(
        self, baseline_model_path: str, adapter_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze memory consumption and parameter counts."""
        print("🔍 Analyzing memory consumption and parameters...")

        # Measure baseline model
        baseline_stats = self._measure_model_stats(baseline_model_path)

        # Measure LoRA model if adapter provided
        lora_stats = None
        if adapter_path and Path(adapter_path).exists():
            try:
                lora_stats = self._measure_model_stats(
                    baseline_model_path, adapter_path
                )
            except Exception as e:
                print(f"Warning: Could not load LoRA adapter: {e}")

        # Calculate comparisons
        analysis = {"baseline": baseline_stats, "lora": lora_stats, "comparisons": {}}

        if lora_stats:
            # Parameter comparison (based on trainable parameters for efficiency)
            baseline_trainable = baseline_stats["trainable_parameters"]
            lora_trainable = lora_stats["trainable_parameters"]
            param_reduction = (
                (baseline_trainable - lora_trainable) / baseline_trainable * 100
            )

            # Memory comparison
            memory_reduction = (
                (baseline_stats["peak_memory_mb"] - lora_stats["peak_memory_mb"])
                / baseline_stats["peak_memory_mb"]
                * 100
            )

            # Latency comparison
            latency_improvement = (
                (
                    baseline_stats["avg_inference_time"]
                    - lora_stats["avg_inference_time"]
                )
                / baseline_stats["avg_inference_time"]
                * 100
            )

            analysis["comparisons"] = {
                "parameter_reduction_percent": param_reduction,
                "memory_reduction_percent": memory_reduction,
                "latency_improvement_percent": latency_improvement,
                "trainable_params_baseline": baseline_trainable,
                "trainable_params_lora": lora_trainable,
                "lora_adapter_params": lora_stats.get("lora_adapter_parameters", 0),
            }

        return analysis

    def _measure_model_stats(
        self, model_path: str, adapter_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Measure model statistics including parameters and memory usage."""
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if adapter_path:
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, adapter_path)
            except Exception:
                pass  # Continue without adapter

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count LoRA adapter parameters separately (they may not be marked as trainable)
        lora_adapter_params = 0
        if adapter_path:
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    lora_adapter_params += param.numel()

        # If we have LoRA adapter params but trainable_params is 0,
        # use adapter params as trainable
        if adapter_path and trainable_params == 0 and lora_adapter_params > 0:
            trainable_params = lora_adapter_params

        # Measure memory usage during inference
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning briefly.",
        ]

        inference_times = []
        peak_memory = 0

        for query in test_queries:
            # Reset memory tracking
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            inputs = tokenizer(
                query, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Track peak memory
            if torch.cuda.is_available():
                peak_memory = max(
                    peak_memory, torch.cuda.max_memory_allocated() / (1024**2)
                )
            else:
                process = psutil.Process()
                peak_memory = max(peak_memory, process.memory_info().rss / (1024**2))

        # Clean up
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_adapter_parameters": lora_adapter_params,
            "peak_memory_mb": peak_memory,
            "avg_inference_time": np.mean(inference_times),
            "model_size_mb": total_params * 4 / (1024**2),  # Approximate size in MB
        }

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

        # Print memory and parameter analysis if available
        if "memory_parameters" in results:
            mem_param = results["memory_parameters"]
            print("\n💾 Memory & Parameters Analysis:")

            if mem_param["baseline"]:
                baseline_stats = mem_param["baseline"]
                print(f"{baseline_name} Model:")
                print(
                    (
                        f"  Total Parameters: {baseline_stats['total_parameters']:,} "
                        f"({baseline_stats['total_parameters']/1e6:.1f}M)"
                    )
                )
                print(
                    f"  Trainable Parameters: "
                    f"{baseline_stats['trainable_parameters']:,}"
                )
                print(f"  Model Size: {baseline_stats['model_size_mb']:.1f} MB")
                print(f"  Peak Memory: {baseline_stats['peak_memory_mb']:.1f} MB")
                print(
                    f"  Avg Inference Time: {baseline_stats['avg_inference_time']:.2f}s"
                )

            if mem_param["lora"]:
                lora_stats = mem_param["lora"]
                print(f"\n{finetuned_name} Model:")
                print(
                    (
                        f"  Total Parameters: {lora_stats['total_parameters']:,} "
                        f"({lora_stats['total_parameters']/1e6:.1f}M)"
                    )
                )
                print(f"  Trainable Parameters: {lora_stats['trainable_parameters']:,}")
                if lora_stats.get("lora_adapter_parameters", 0) > 0:
                    print(
                        (
                            f"  LoRA Adapter Parameters: "
                            f"{lora_stats['lora_adapter_parameters']:,} "
                            f"({lora_stats['lora_adapter_parameters']/1e6:.3f}M)"
                        )
                    )
                print(f"  Model Size: {lora_stats['model_size_mb']:.1f} MB")
                print(f"  Peak Memory: {lora_stats['peak_memory_mb']:.1f} MB")
                print(f"  Avg Inference Time: {lora_stats['avg_inference_time']:.2f}s")

            if mem_param["comparisons"]:
                comp = mem_param["comparisons"]
                print("\nEfficiency Improvements:")
                print(
                    f"  Trainable Parameters Reduction: "
                    f"{comp['parameter_reduction_percent']:.1f}%"
                )
                print(f"  Memory Reduction: {comp['memory_reduction_percent']:+.1f}%")
                print(
                    f"  Latency Improvement: "
                    f"{comp['latency_improvement_percent']:+.1f}%"
                )
                adapter_params = comp.get(
                    "lora_adapter_params", comp["trainable_params_lora"]
                )
                print(f"  LoRA Adapter Size: {adapter_params/1e6:.3f}M parameters")


def get_benchmark_analyzer(mlruns_dir: Optional[str] = None) -> BenchmarkAnalyzer:
    """
    Factory function to get a BenchmarkAnalyzer instance.

    Args:
        mlruns_dir: Directory containing MLflow runs

    Returns:
        Configured BenchmarkAnalyzer instance
    """
    return BenchmarkAnalyzer(mlruns_dir=mlruns_dir)
