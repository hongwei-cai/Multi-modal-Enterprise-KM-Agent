from configs.model_config import BenchmarkSummary
from src.rag.components.benchmarking_workflow import BenchmarkingWorkflow


def test_calculate_comparison_metrics(monkeypatch):
    # Create fake BenchmarkSummary objects
    baseline = BenchmarkSummary(
        model_type="baseline",
        load_time=1.0,
        avg_latency=1.0,
        avg_response_length=10.0,
        success_rate=1.0,
        total_queries=2,
        results=[
            {"tokens_per_second": 10.0, "memory_usage_mb": 100.0, "response": "OK"},
            {"tokens_per_second": 12.0, "memory_usage_mb": 110.0, "response": "Fine"},
        ],
    )

    finetuned = BenchmarkSummary(
        model_type="finetuned",
        load_time=1.2,
        avg_latency=0.8,
        avg_response_length=12.0,
        success_rate=0.9,
        total_queries=2,
        results=[
            {"tokens_per_second": 15.0, "memory_usage_mb": 90.0, "response": "Good"},
            {"tokens_per_second": 16.0, "memory_usage_mb": 95.0, "response": "Better"},
        ],
    )

    wf = BenchmarkingWorkflow()

    # Monkeypatch quality comparator to avoid heavy embedding work
    monkeypatch.setattr(
        "src.rag.components.benchmarking_workflow.compare_responses",
        lambda a, b: {"avg_similarity": 0.85, "per_query": [0.8, 0.9]},
    )

    metrics = wf._calculate_comparison_metrics(baseline, finetuned)

    assert "baseline" in metrics and "finetuned" in metrics
    assert metrics["baseline"]["avg_tokens_per_second"] == 11.0
    assert metrics["finetuned"]["avg_tokens_per_second"] == 15.5
    assert "quality" in metrics
    assert metrics["quality"]["avg_similarity"] == 0.85
