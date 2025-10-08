#!/usr/bin/env python3
"""
Example script showing how to use the BenchmarkingWorkflow component
outside of notebooks for automated benchmarking.
"""

from pathlib import Path

from src.rag.components import get_benchmarking_workflow


def main():
    """Run automated model benchmarking."""

    # Configuration
    baseline_model = "microsoft/DialoGPT-medium"
    project_root = Path(__file__).parent.parent
    model_configs_path = project_root / "model_configs"
    adapter_path = (
        model_configs_path
        / "lora_adapters"
        / "microsoft"
        / "DialoGPT-medium"
        / "lora_adapter"
    )

    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "What are the benefits of fine-tuning?",
    ]

    # Initialize workflow
    workflow = get_benchmarking_workflow(experiment_name="automated_benchmark")

    # Run benchmarking
    print("Starting automated benchmarking...")
    results = workflow.run_comparison_benchmark(
        baseline_model=baseline_model,
        fine_tuned_model=baseline_model,
        adapter_path=str(adapter_path),
        test_queries=test_queries,
        run_name="script_run",
    )

    # Print results
    workflow.print_results_summary(results)

    print("\n✅ Automated benchmarking complete!")


if __name__ == "__main__":
    main()
