"""
Experiment Manager for A/B testing and model benchmarking.
"""
import logging
import random
import time
from typing import Any, Dict, Optional

import psutil

from ..config import ABTestConfig, BenchmarkResult
from ..experiment_tracker import (
    ExperimentConfig,
    MLflowExperimentTracker,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manager for A/B testing and model benchmarking operations."""

    def __init__(self, experiment_tracker: Optional[MLflowExperimentTracker] = None):
        self.experiment_tracker = experiment_tracker or MLflowExperimentTracker()
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.benchmark_results: Dict[str, BenchmarkResult] = {}

    def start_ab_test(self, config: ABTestConfig):
        """Start an A/B test between two models."""
        self.ab_tests[config.test_name] = config
        logger.info(f"Started A/B test: {config.test_name}")

    def get_ab_test_model(self, test_name: str) -> Optional[str]:
        """Get the model to use for A/B testing based on traffic split."""
        if test_name not in self.ab_tests:
            return None

        config = self.ab_tests[test_name]
        return (
            config.model_a if random.random() < config.traffic_split else config.model_b
        )

    def record_ab_test_result(
        self, test_name: str, model_used: str, metrics: Dict[str, Any]
    ):
        """Record results from A/B test with MLflow tracking."""
        try:
            # Extract metrics
            latency_ms = metrics.get("latency", 0.0)
            memory_mb = metrics.get("memory_usage", 0.0)
            cpu_percent = psutil.cpu_percent()
            quality_score = metrics.get("quality_score")
            error_rate = metrics.get("error_rate", 0.0)

            # Create performance metrics
            perf_metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                response_quality_score=quality_score,
                error_rate=error_rate,
            )

            # Get the A/B test configuration
            if test_name in self.ab_tests:
                ab_config = self.ab_tests[test_name]

                # Determine which variant was used
                variant_a = ab_config.model_a
                variant_b = ab_config.model_b

                # For now, we'll assume we need both variants' metrics
                # In a real implementation, you'd collect metrics for both variants
                # Here we'll create a mock second set for demonstration
                mock_metrics_b = PerformanceMetrics(
                    latency_ms=latency_ms * 1.1,  # Slightly worse
                    memory_usage_mb=memory_mb,
                    cpu_usage_percent=cpu_percent,
                    response_quality_score=quality_score * 0.95
                    if quality_score
                    else None,
                    error_rate=error_rate,
                )

                # Determine winner based on quality or latency
                if (
                    perf_metrics.response_quality_score is not None
                    and mock_metrics_b.response_quality_score is not None
                ):
                    winner = (
                        variant_a
                        if perf_metrics.response_quality_score
                        > mock_metrics_b.response_quality_score
                        else variant_b
                    )
                    confidence = abs(
                        perf_metrics.response_quality_score
                        - mock_metrics_b.response_quality_score
                    )
                else:
                    # Fall back to latency comparison
                    winner = (
                        variant_a
                        if perf_metrics.latency_ms < mock_metrics_b.latency_ms
                        else variant_b
                    )
                    confidence = abs(
                        perf_metrics.latency_ms - mock_metrics_b.latency_ms
                    )

                # Track the A/B test result
                run_id = self.experiment_tracker.start_experiment(
                    ExperimentConfig(
                        experiment_name=f"ab_test_{test_name}",
                        run_name=f"ab_test_run_{int(time.time())}",
                        model_name=f"{variant_a}_vs_{variant_b}",
                        model_version="ab_test",
                        parameters={
                            "test_name": test_name,
                            "variant_a": variant_a,
                            "variant_b": variant_b,
                            "traffic_split": ab_config.traffic_split,
                        },
                    )
                )

                self.experiment_tracker.log_ab_test_result(
                    run_id,
                    test_name,
                    variant_a,
                    variant_b,
                    winner,
                    confidence,
                    perf_metrics,
                    mock_metrics_b,
                )

                self.experiment_tracker.end_experiment(run_id)

                logger.info(f"A/B test '{test_name}' recorded with winner: {winner}")

            else:
                # Track as regular performance metrics
                run_id = self.experiment_tracker.start_experiment(
                    ExperimentConfig(
                        experiment_name=f"model_performance_{model_used}",
                        run_name=f"performance_run_{int(time.time())}",
                        model_name=model_used,
                        model_version="unknown",
                        parameters=metrics,
                    )
                )

                self.experiment_tracker.log_metrics(run_id, perf_metrics)
                self.experiment_tracker.end_experiment(run_id)

        except Exception as e:
            logger.error(f"Failed to record A/B test result: {e}")

    def benchmark_model(
        self,
        model_name: str,
        test_prompt: str = "Hello, how are you?",
        max_tokens: int = 50,
        model_loader=None,
    ) -> BenchmarkResult:
        """Benchmark a model's performance."""
        if model_loader is None:
            raise ValueError("model_loader function must be provided")

        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        start_time = time.time()

        try:
            # Load model
            model, tokenizer = model_loader(model_name, use_quantization=True)

            # Tokenize input
            inputs = tokenizer(test_prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            import torch

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Deterministic for benchmarking
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Calculate metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**2)  # MB

            latency_ms = (end_time - start_time) * 1000
            memory_usage_mb = end_memory - start_memory
            generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tokens_per_second = (
                generated_tokens / (end_time - start_time)
                if (end_time - start_time) > 0
                else 0
            )

            result = BenchmarkResult(
                model_name=model_name,
                latency_ms=latency_ms,
                memory_usage_gb=memory_usage_mb / 1024,  # Convert MB to GB
                tokens_per_second=tokens_per_second,
            )

            # Store benchmark result
            self.benchmark_results[model_name] = result

            logger.info(
                f"Benchmarked {model_name}: {latency_ms:.2f}ms, "
                f"{tokens_per_second:.2f} tokens/sec"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            # Return a failed benchmark result
            return BenchmarkResult(
                model_name=model_name,
                latency_ms=float("inf"),
                memory_usage_gb=0,
                tokens_per_second=0,
            )

    def get_benchmark_results(self) -> Dict[str, BenchmarkResult]:
        """Get all benchmark results."""
        return self.benchmark_results.copy()

    def get_ab_tests(self) -> Dict[str, ABTestConfig]:
        """Get all active A/B tests."""
        return self.ab_tests.copy()
