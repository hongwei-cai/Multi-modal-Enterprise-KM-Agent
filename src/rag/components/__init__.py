"""
Streamlined Components Package

This package contains high-level workflow components that orchestrate
building blocks into complete workflows. These components can be used
in notebooks, scripts, APIs, or other applications.
"""

from pathlib import Path
from typing import Optional, Union

from .benchmark_analyzer import BenchmarkAnalyzer
from .benchmarking_workflow import BenchmarkingWorkflow, get_benchmarking_workflow
from .indexing_pipeline import IndexingPipeline
from .lora_finetuner import LoRAFinetuner
from .rag_pipeline import RAGPipeline

__all__ = [
    "BenchmarkingWorkflow",
    "get_benchmarking_workflow",
    "BenchmarkAnalyzer",
    "get_benchmark_analyzer",
    "IndexingPipeline",
    "RAGPipeline",
    "LoRAFinetuner",
    "get_indexing_pipeline",
    "get_rag_pipeline",
]


def get_indexing_pipeline(db_path: Optional[str] = None) -> IndexingPipeline:
    """
    Factory function to get an IndexingPipeline instance.

    Args:
        db_path: Path to the vector database

    Returns:
        Configured IndexingPipeline instance
    """
    return IndexingPipeline(db_path=db_path)


def get_rag_pipeline(db_path: Optional[str] = None, top_k: int = 5) -> RAGPipeline:
    """
    Factory function to get a RAGPipeline instance.

    Args:
        db_path: Path to the vector database
        top_k: Number of documents to retrieve

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline(db_path=db_path, top_k=top_k)


def get_lora_finetuner(
    adapter_save_path: Optional[Union[Path, str]] = None, device: str = "cpu"
) -> LoRAFinetuner:
    """
    Factory function to get a LoRAFinetuner instance.

    Args:
        adapter_save_path: Path to save the adapter
        device: Device to run the fine-tuning on

    Returns:
        Configured LoRAFinetuner instance
    """
    return LoRAFinetuner(adapter_save_path=adapter_save_path, device=device)


def get_benchmark_analyzer(mlruns_dir: Optional[str] = None) -> BenchmarkAnalyzer:
    """
    Factory function to get a BenchmarkAnalyzer instance.

    Args:
        mlruns_dir: Directory containing MLflow runs

    Returns:
        Configured BenchmarkAnalyzer instance
    """
    return BenchmarkAnalyzer(mlruns_dir=mlruns_dir)
