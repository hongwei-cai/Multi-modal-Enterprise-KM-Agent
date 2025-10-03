"""
LLM inference client: Local (Transformers) or Cloud (vLLM).
"""
import logging
import os
from typing import List, Optional

import requests  # type: ignore

from .model_manager import get_model_manager

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self, model_name: Optional[str] = None, priority: Optional[str] = None
    ):
        if model_name is None:
            model_name = os.getenv("LLM_MODEL_NAME", "gpt2")
        self.model_name = model_name
        self.is_cloud = bool(os.getenv("CLOUD_ENV"))
        self.model_manager = get_model_manager()

        # Dynamic model selection
        env_priority = os.getenv("MODEL_PRIORITY", "balanced")
        self.priority: str = priority or (env_priority if env_priority else "balanced")

        if not self.is_cloud:
            self.use_quantization = (
                os.getenv("USE_QUANTIZATION", "false").lower() == "true"
            )  # Default to False for better generation on M1
            self.quant_type = os.getenv("QUANT_TYPE", "dynamic")

            # Use dynamic model selection if no specific model requested
            if model_name == os.getenv("LLM_MODEL_NAME", "gpt2"):  # Default model
                selected_model = self.model_manager.get_model_recommendation(
                    self.priority
                )
                logger.info(
                    f"Selected model '{selected_model}' for priority '{self.priority}'"
                )
                self.model_name = selected_model

            self.model, self.tokenizer = self.model_manager.load_model_with_fallback(
                self.model_name,
                use_quantization=self.use_quantization,
                quant_type=self.quant_type,
            )
            # Check if seq2seq model
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            self.is_seq2seq = config.model_type in ["t5", "mt5", "bart", "pegasus"]
        else:
            self.api_url = os.getenv(
                "VLLM_API_URL", "http://localhost:8000/v1/chat/completions"
            )

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        # Validate parameters
        if max_length is None:
            max_length = 50 if not self.is_cloud else 50
        if not (0 < temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        if not (0 < top_p <= 1):
            raise ValueError("Top-p must be between 0 and 1")

        if self.is_cloud:
            # Use vLLM API with parameters
            response = requests.post(
                self.api_url,
                json={  # Changed to json for proper payload
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                timeout=30,
            )
            response.raise_for_status()
            response = response.json()["choices"][0]["message"]["content"]
        else:
            # Local generation with parameters
            logger.debug("Prompt length: %s", len(prompt))
            logger.debug("Prompt start: %s...", prompt[:200])
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            logger.debug("Tokenized inputs: %s", inputs)

            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id
                    if self.is_seq2seq
                    else self.tokenizer.eos_token_id,
                )
            except Exception as e:
                logger.error("Generation error: %s", e)
                return "Error: Generation failed"
            logger.debug("Generated outputs: %s", outputs)

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response if it's included
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()
            logger.debug("Decoded response: '%s'", response)

        # Post-process response
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response

    def benchmark_current_model(
        self, test_prompt: str = "Hello, how are you?", max_tokens: int = 50
    ):
        """Benchmark the current model's performance."""
        if self.is_cloud:
            logger.warning("Benchmarking not supported for cloud models")
            return None
        return self.model_manager.benchmark_model(
            self.model_name, test_prompt, max_tokens
        )

    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model version."""
        manager = get_model_manager()
        return manager.switch_model(model_name)

    def get_available_models(self) -> List[str]:
        """Get list of available model versions."""
        manager = get_model_manager()
        return manager.list_model_versions()

    def start_ab_test(
        self,
        test_name: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
    ):
        """Start A/B testing between two models."""
        from src.rag.model_manager import ABTestConfig

        manager = get_model_manager()
        config = ABTestConfig(
            test_name=test_name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
        )
        manager.start_ab_test(config)

    def generate_with_ab_test(self, prompt: str, test_name: str, **kwargs) -> str:
        """Generate response using A/B tested model."""
        manager = get_model_manager()
        model_name = manager.get_ab_test_model(test_name)
        if not model_name:
            raise ValueError(f"A/B test '{test_name}' not found")

        # Temporarily switch to test model
        original_model = manager.current_model
        if manager.switch_model(model_name):
            try:
                response = self.generate(prompt, **kwargs)
                # Record metrics (simplified)
                manager.record_ab_test_result(
                    test_name, model_name, {"response_length": len(response)}
                )
                return response
            finally:
                if original_model:
                    manager.switch_model(original_model)
        else:
            raise RuntimeError(f"Failed to switch to model {model_name}")

    def get_optimal_model_for_constraints(
        self,
        max_latency_ms: Optional[float] = None,
        max_memory_gb: Optional[float] = None,
    ):
        """Get the optimal model for given constraints and switch to it."""
        optimal_model = self.model_manager.get_best_model_for_constraints(
            max_latency_ms, max_memory_gb
        )

        if optimal_model != self.model_name:
            logger.info(f"Switching to optimal model {optimal_model} for constraints")
            self.switch_model(optimal_model)

        return optimal_model


# Convenience function
def get_llm_client(
    model_name: Optional[str] = None, priority: Optional[str] = None
) -> LLMClient:
    return LLMClient(model_name, priority)
