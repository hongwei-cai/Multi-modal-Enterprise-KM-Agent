"""
Benchmarker Building Block

This module provides comprehensive model benchmarking capabilities for comparing
baseline models against fine-tuned variants (LoRA adapters). It handles model
loading, query execution, performance measurement, and result aggregation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import BenchmarkSummary

logger = logging.getLogger(__name__)


class Benchmarker:
    """
    Comprehensive model benchmarking utility.

    This building block handles the low-level details of model benchmarking,
    including loading models (with or without LoRA adapters), running test
    queries, measuring performance metrics, and aggregating results.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the benchmarker.

        Args:
            device: Target device for model inference
            ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Benchmarker initialized for device: {self.device}")

    def benchmark_model(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        queries: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **generation_kwargs,
    ) -> Optional[BenchmarkSummary]:
        """
        Benchmark a model by running multiple queries and measuring performance.

        Args:
            model_name_or_path: HuggingFace model name or local path
            adapter_path: Optional path to LoRA adapter for fine-tuned model
            queries: List of test queries to run
            max_new_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature for generation
            **generation_kwargs: Additional generation parameters

        Returns:
            BenchmarkSummary with comprehensive performance metrics
        """
        if queries is None:
            queries = ["What is artificial intelligence?"]

        results = []
        start_load_time = time.time()

        try:
            # Load model and tokenizer
            model, tokenizer = self._load_model(model_name_or_path, adapter_path)
            model_type = (
                f"{model_name_or_path} + LoRA" if adapter_path else model_name_or_path
            )

            load_time = time.time() - start_load_time
            logger.info(f"✓ Loaded {model_type} in {load_time:.2f}s")

            # Move model to device
            model.to(self.device)

            # Benchmark each query
            for query in queries:
                query_result = self._benchmark_single_query(
                    model,
                    tokenizer,
                    query,
                    max_new_tokens,
                    temperature,
                    **generation_kwargs,
                )
                results.append(query_result)
                logger.info(
                    f"  ✓ Query: {query[:50]}... -> "
                    f"{query_result['response_length']} \
                        words in {query_result['latency']:.2f}s"
                )

        except Exception as e:
            logger.error(f"✗ Failed to load or benchmark {model_name_or_path}: {e}")
            return None

        # Calculate summary statistics
        successful_results = [r for r in results if r["success"]]
        avg_latency = (
            sum(r["latency"] for r in results) / len(results) if results else 0
        )
        avg_response_length = (
            sum(r["response_length"] for r in successful_results)
            / len(successful_results)
            if successful_results
            else 0
        )
        success_rate = len(successful_results) / len(results) if results else 0

        return BenchmarkSummary(
            model_type=model_type,
            load_time=load_time,
            avg_latency=avg_latency,
            avg_response_length=avg_response_length,
            success_rate=success_rate,
            total_queries=len(results),
            results=results,
        )

    def _load_model(
        self, model_name_or_path: str, adapter_path: Optional[str] = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer, optionally with LoRA adapter.

        Args:
            model_name_or_path: Base model name/path
            adapter_path: Optional LoRA adapter path

        Returns:
            Tuple of (model, tokenizer)
        """
        # Determine dtype based on device
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA adapter if provided
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)

        return model, tokenizer

    def _benchmark_single_query(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        query: str,
        max_new_tokens: int,
        temperature: float,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        Benchmark a single query and return detailed metrics.

        Args:
            model: The model to benchmark
            tokenizer: The tokenizer to use
            query: The query string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with query results and metrics
        """
        # Measure memory before
        memory_before = psutil.virtual_memory().used / (1024**2)  # MB
        start_time = time.time()

        try:
            # Tokenize input
            inputs = tokenizer(
                query, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_kwargs,
                )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_length = len(response.split())

            # Measure performance
            latency = time.time() - start_time
            memory_after = psutil.virtual_memory().used / (1024**2)  # MB
            memory_usage = memory_after - memory_before

            return {
                "query": query,
                "response": response,
                "response_length": response_length,
                "latency": latency,
                "memory_usage_mb": memory_usage,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Failed to benchmark query '{query[:50]}...': {e}")
            return {
                "query": query,
                "response": "",
                "response_length": 0,
                "latency": time.time() - start_time,
                "memory_usage_mb": 0,
                "success": False,
                "error": str(e),
            }

    def compare_models(
        self,
        baseline_model: str,
        fine_tuned_model: str,
        adapter_path: str,
        queries: Optional[List[str]] = None,
    ) -> Dict[str, Optional[BenchmarkSummary]]:
        """
        Compare baseline and fine-tuned models side by side.

        Args:
            baseline_model: Name/path of baseline model
            fine_tuned_model: Name/path of base model for fine-tuned version
            adapter_path: Path to LoRA adapter
            queries: Test queries to run

        Returns:
            Dictionary with 'baseline' and 'finetuned' BenchmarkSummary objects
        """
        logger.info("🚀 Starting model comparison benchmarking...")

        # Benchmark baseline
        logger.info("📊 Benchmarking baseline model...")
        baseline_results = self.benchmark_model(baseline_model, queries=queries)

        # Benchmark fine-tuned
        logger.info("📊 Benchmarking fine-tuned model...")
        finetuned_results = self.benchmark_model(
            fine_tuned_model, adapter_path=adapter_path, queries=queries
        )

        return {"baseline": baseline_results, "finetuned": finetuned_results}


def get_benchmarker(device: Optional[str] = None) -> Benchmarker:
    """
    Factory function to get a Benchmarker instance.

    Args:
        device: Target device ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Configured Benchmarker instance
    """
    return Benchmarker(device=device)
