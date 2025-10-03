"""
Test script for MLflow experiment tracking functionality.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from rag.experiment_tracker import (  # noqa: E402
    ExperimentConfig,
    MLflowExperimentTracker,
    PerformanceMetrics,
    track_ab_test,
    track_model_performance,
)


@pytest.fixture
def temp_mlflow_dir():
    """Create a temporary directory for MLflow experiments."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def tracker(temp_mlflow_dir):
    """Create an MLflowExperimentTracker with temporary database."""
    db_path = os.path.join(temp_mlflow_dir, "experiments.db")
    tracker = MLflowExperimentTracker(database_path=db_path)
    yield tracker
    # Cleanup: end any active runs
    import mlflow

    try:
        mlflow.end_run()
    except Exception:
        pass


def test_basic_experiment_tracking(tracker):
    """Test basic experiment tracking functionality."""
    print("Testing basic experiment tracking...")

    # Create experiment config
    config = ExperimentConfig(
        experiment_name="test_experiment",
        run_name="test_run_1",
        model_name="test_model",
        model_version="v1.0",
        parameters={"learning_rate": 0.01, "batch_size": 32},
    )

    # Start experiment
    run_id = tracker.start_experiment(config)
    print(f"Started experiment with run ID: {run_id}")

    # Log metrics
    metrics = PerformanceMetrics(
        latency_ms=150.5,
        memory_usage_mb=512.0,
        cpu_usage_percent=45.2,
        throughput_tokens_per_sec=25.3,
        response_quality_score=0.85,
    )

    tracker.log_metrics(run_id, metrics)
    print("Logged performance metrics")

    # End experiment
    tracker.end_experiment(run_id)
    print("Ended experiment")

    # Get experiment runs
    runs = tracker.get_experiment_runs("test_experiment")
    print(f"Found {len(runs)} runs in test_experiment")

    assert len(runs) >= 1


def test_convenience_functions(temp_mlflow_dir):
    """Test convenience functions for tracking."""
    print("\nTesting convenience functions...")

    # Set tracking URI to temp directory
    db_path = os.path.join(temp_mlflow_dir, "experiments.db")
    tracking_uri = f"sqlite:///{db_path}"

    # Track model performance
    run_id = track_model_performance(
        model_name="gpt2",
        model_version="v1.0",
        operation="generate",
        latency_ms=200.0,
        memory_mb=300.0,
        tracking_uri=tracking_uri,
    )
    print(f"Tracked model performance with run ID: {run_id}")
    assert run_id is not None

    # Track A/B test
    metrics_a = PerformanceMetrics(
        latency_ms=180.0,
        memory_usage_mb=250.0,
        cpu_usage_percent=40.0,
        response_quality_score=0.8,
    )

    metrics_b = PerformanceMetrics(
        latency_ms=220.0,
        memory_usage_mb=280.0,
        cpu_usage_percent=42.0,
        response_quality_score=0.75,
    )

    ab_run_id = track_ab_test(
        test_name="model_comparison_test",
        variant_a="gpt2_small",
        variant_b="gpt2_medium",
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        tracking_uri=tracking_uri,
    )
    print(f"Tracked A/B test with run ID: {ab_run_id}")
    assert ab_run_id is not None


def test_experiment_comparison(tracker):
    """Test experiment comparison functionality."""
    print("\nTesting experiment comparison...")

    # Compare experiments
    comparison = tracker.compare_experiments(
        ["test_experiment", "model_performance_gpt2"]
    )
    print(f"Comparison results: {len(comparison)} experiments compared")

    assert isinstance(comparison, dict)
