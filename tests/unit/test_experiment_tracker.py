"""
Test script for MLflow experiment tracking functionality.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from rag.experiment_tracker import (  # noqa: E402
    ExperimentConfig,
    MLflowExperimentTracker,
    PerformanceMetrics,
    track_ab_test,
    track_model_performance,
)


def test_basic_experiment_tracking():
    """Test basic experiment tracking functionality."""
    print("Testing basic experiment tracking...")

    tracker = MLflowExperimentTracker()

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

    return True


def test_convenience_functions():
    """Test convenience functions for tracking."""
    print("\nTesting convenience functions...")

    # Track model performance
    run_id = track_model_performance(
        model_name="gpt2",
        model_version="v1.0",
        operation="generate",
        latency_ms=200.0,
        memory_mb=300.0,
    )
    print(f"Tracked model performance with run ID: {run_id}")

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
    )
    print(f"Tracked A/B test with run ID: {ab_run_id}")

    return True


def test_experiment_comparison():
    """Test experiment comparison functionality."""
    print("\nTesting experiment comparison...")

    tracker = MLflowExperimentTracker()

    # Compare experiments
    comparison = tracker.compare_experiments(
        ["test_experiment", "model_performance_gpt2"]
    )
    print(f"Comparison results: {len(comparison)} experiments compared")

    return True


if __name__ == "__main__":
    print("Starting MLflow experiment tracking tests...")

    try:
        test_basic_experiment_tracking()
        test_convenience_functions()
        test_experiment_comparison()

        print("\n✅ All tests passed! MLflow experiment tracking is working correctly.")

    except (ImportError, RuntimeError) as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
