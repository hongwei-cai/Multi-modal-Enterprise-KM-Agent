"""
MLflow Experiment Tracking for Multi-modal Enterprise KM Agent

This module provides comprehensive experiment tracking capabilities for model
performance monitoring, A/B testing results, and system optimization experiments.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import psutil
from mlflow.tracking import MlflowClient


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    experiment_name: str
    run_name: str
    model_name: str
    model_version: str
    parameters: Dict[str, Any]
    tags: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class PerformanceMetrics:
    """Performance metrics for model operations."""

    latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_tokens_per_sec: Optional[float] = None
    response_quality_score: Optional[float] = None
    error_rate: float = 0.0


class MLflowExperimentTracker:
    """
    MLflow-based experiment tracking system with SQLite backend.

    Provides comprehensive tracking for:
    - Model performance metrics
    - A/B testing results
    - System resource usage
    - Experiment versioning and organization
    """

    active_runs: Dict[str, ExperimentConfig]

    def __init__(
        self, tracking_uri: Optional[str] = None, database_path: Optional[str] = None
    ):
        """
        Initialize MLflow experiment tracker.

        Args:
            tracking_uri: MLflow tracking URI (defaults to local SQLite)
            database_path: Path to SQLite database for experiments
        """
        if tracking_uri is None:
            # Use local SQLite backend
            if database_path is None:
                database_path = os.path.join(os.getcwd(), "mlruns", "experiments.db")
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
            tracking_uri = f"sqlite:///{database_path}"

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        self.active_runs = {}

        # Ensure default experiment exists
        self._ensure_default_experiment()

    def _ensure_default_experiment(self):
        """Ensure the default experiment exists."""
        try:
            self.client.get_experiment_by_name("Default")
        except Exception:
            self.client.create_experiment("Default")

    def start_experiment(self, config: ExperimentConfig) -> str:
        """
        Start a new experiment run.

        Args:
            config: Experiment configuration

        Returns:
            Run ID for the started experiment
        """
        # Set or create experiment
        try:
            self.client.get_experiment_by_name(config.experiment_name)
        except Exception:
            self.client.create_experiment(config.experiment_name)

        # Start run
        mlflow.set_experiment(config.experiment_name)
        run = mlflow.start_run(run_name=config.run_name)

        # Log parameters
        mlflow.log_params(config.parameters)

        # Log model info
        mlflow.log_param("model_name", config.model_name)
        mlflow.log_param("model_version", config.model_version)

        # Log tags
        if config.tags:
            mlflow.set_tags(config.tags)

        # Log system info
        self._log_system_info()

        run_id = run.info.run_id
        self.active_runs[run_id] = config

        return run_id

    def log_metrics(self, run_id: str, metrics: PerformanceMetrics):
        """
        Log performance metrics for an experiment run.

        Args:
            run_id: Experiment run ID
            metrics: Performance metrics to log
        """
        if run_id not in self.active_runs:
            raise ValueError(f"Run ID {run_id} not found in active runs")

        # Convert metrics to dict and filter out None values
        metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
        mlflow.log_metrics(metrics_dict, step=0)

        # Also log as JSON artifact for structured access
        self._log_metrics_artifact(run_id, metrics)

    def log_ab_test_result(
        self,
        run_id: str,
        test_name: str,
        variant_a: str,
        variant_b: str,
        winner: str,
        confidence: float,
        metrics_a: PerformanceMetrics,
        metrics_b: PerformanceMetrics,
    ):
        """
        Log A/B test results.

        Args:
            run_id: Experiment run ID
            test_name: Name of the A/B test
            variant_a: Name/version of variant A
            variant_b: Name/version of variant B
            winner: Winning variant
            confidence: Confidence level of the result
            metrics_a: Performance metrics for variant A
            metrics_b: Performance metrics for variant B
        """
        ab_test_data = {
            "test_name": test_name,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "winner": winner,
            "confidence": confidence,
            "metrics_a": asdict(metrics_a),
            "metrics_b": asdict(metrics_b),
            "timestamp": datetime.now().isoformat(),
        }

        # Log as parameters
        mlflow.log_param(f"ab_test_{test_name}_winner", winner)
        mlflow.log_param(f"ab_test_{test_name}_confidence", confidence)

        # Log as artifact
        artifact_path = f"ab_test_results/{test_name}.json"
        self._log_json_artifact(run_id, ab_test_data, artifact_path)

    def log_model_artifact(
        self, run_id: str, model_path: str, artifact_name: str = "model"
    ):
        """
        Log a model as an MLflow artifact.

        Args:
            run_id: Experiment run ID
            model_path: Path to the model file/directory
            artifact_name: Name for the artifact
        """
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_name)

    def end_experiment(self, run_id: str, status: str = "FINISHED"):
        """
        End an experiment run.

        Args:
            run_id: Experiment run ID to end
            status: Final status ("FINISHED", "FAILED", etc.)
        """
        if run_id in self.active_runs:
            mlflow.end_run(status)
            del self.active_runs[run_id]
        else:
            # Try to end current run if no specific run_id
            try:
                mlflow.end_run(status)
            except Exception:
                pass

    def get_experiment_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """
        Get all runs for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            List of run information dictionaries
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            runs = self.client.search_runs([experiment.experiment_id])
            return [run.to_dictionary() for run in runs]
        except Exception:
            return []

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.

        Args:
            experiment_names: List of experiment names to compare

        Returns:
            Comparison data including metrics and parameters
        """
        comparison_data = {}

        for exp_name in experiment_names:
            runs = self.get_experiment_runs(exp_name)
            if runs:
                # Get the best run (highest response quality or lowest latency)
                best_run = max(
                    runs,
                    key=lambda x: x.get("metrics", {}).get("response_quality_score", 0),
                )
                comparison_data[exp_name] = best_run

        return comparison_data

    def _log_system_info(self):
        """Log system information as run parameters."""
        mlflow.log_param(
            "python_version", f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        )
        mlflow.log_param("platform", os.sys.platform)

        # Log CPU and memory info
        cpu_count = psutil.cpu_count()
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB

        mlflow.log_param("cpu_count", cpu_count)
        mlflow.log_param("total_memory_gb", round(total_memory, 2))

    def _log_metrics_artifact(self, run_id: str, metrics: PerformanceMetrics):
        """Log metrics as a JSON artifact."""
        metrics_data = {
            "metrics": asdict(metrics),
            "timestamp": datetime.now().isoformat(),
        }
        self._log_json_artifact(run_id, metrics_data, "performance_metrics.json")

    def _log_json_artifact(self, run_id: str, data: Dict[str, Any], filename: str):
        """Log data as a JSON artifact."""
        # Create temporary file
        temp_dir = f"/tmp/mlflow_artifacts_{run_id}"
        os.makedirs(temp_dir, exist_ok=True)

        filepath = os.path.join(temp_dir, filename)
        os.makedirs(
            os.path.dirname(filepath), exist_ok=True
        )  # Ensure parent directories exist

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        mlflow.log_artifact(filepath, "")

        # Cleanup
        try:
            os.remove(filepath)
            os.rmdir(temp_dir)
        except Exception:
            pass

    def create_experiment_version(self, base_experiment: str, version_name: str) -> str:
        """
        Create a new version of an experiment.

        Args:
            base_experiment: Name of the base experiment
            version_name: Name for the new version

        Returns:
            New experiment name
        """
        new_experiment_name = f"{base_experiment}_v_{version_name}"

        # Copy tags and description from base experiment
        try:
            base_exp = self.client.get_experiment_by_name(base_experiment)
            self.client.create_experiment(new_experiment_name, base_exp.tags)
        except Exception:
            self.client.create_experiment(new_experiment_name)

        return new_experiment_name


# Convenience functions for common use cases
def track_model_performance(
    model_name: str,
    model_version: str,
    operation: str,
    latency_ms: float,
    memory_mb: float,
) -> str:
    """
    Convenience function to track model performance.

    Args:
        model_name: Name of the model
        model_version: Version of the model
        operation: Operation being performed (e.g., "generate", "embed")
        latency_ms: Response latency in milliseconds
        memory_mb: Memory usage in MB

    Returns:
        Run ID of the tracking run
    """
    tracker = MLflowExperimentTracker()

    config = ExperimentConfig(
        experiment_name=f"model_performance_{model_name}",
        run_name=f"{operation}_{int(time.time())}",
        model_name=model_name,
        model_version=model_version,
        parameters={"operation": operation},
    )

    run_id = tracker.start_experiment(config)

    metrics = PerformanceMetrics(
        latency_ms=latency_ms,
        memory_usage_mb=memory_mb,
        cpu_usage_percent=psutil.cpu_percent(),
    )

    tracker.log_metrics(run_id, metrics)
    tracker.end_experiment(run_id)

    return run_id


def track_ab_test(
    test_name: str,
    variant_a: str,
    variant_b: str,
    metrics_a: PerformanceMetrics,
    metrics_b: PerformanceMetrics,
) -> str:
    """
    Convenience function to track A/B test results.

    Args:
        test_name: Name of the A/B test
        variant_a: Name/version of variant A
        variant_b: Name/version of variant B
        metrics_a: Performance metrics for variant A
        metrics_b: Performance metrics for variant B

    Returns:
        Run ID of the tracking run
    """
    tracker = MLflowExperimentTracker()

    # Determine winner based on response quality (if available) or latency
    if (
        metrics_a.response_quality_score is not None
        and metrics_b.response_quality_score is not None
    ):
        winner = (
            variant_a
            if metrics_a.response_quality_score > metrics_b.response_quality_score
            else variant_b
        )
        confidence = abs(
            metrics_a.response_quality_score - metrics_b.response_quality_score
        )
    else:
        winner = variant_a if metrics_a.latency_ms < metrics_b.latency_ms else variant_b
        confidence = abs(metrics_b.latency_ms - metrics_a.latency_ms) / max(
            metrics_a.latency_ms, metrics_b.latency_ms
        )

    config = ExperimentConfig(
        experiment_name=f"ab_test_{test_name}",
        run_name=f"test_{int(time.time())}",
        model_name=f"{variant_a}_vs_{variant_b}",
        model_version="ab_test",
        parameters={"test_name": test_name},
    )

    run_id = tracker.start_experiment(config)
    tracker.log_ab_test_result(
        run_id,
        test_name,
        variant_a,
        variant_b,
        winner,
        confidence,
        metrics_a,
        metrics_b,
    )
    tracker.end_experiment(run_id)

    return run_id
