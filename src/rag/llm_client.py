"""
LLM inference client: Local (Transformers) or Cloud (vLLM).
"""
import logging
import os
from typing import Optional

import requests  # type: ignore

from .model_manager import get_model_manager

logger = logging.getLogger(__name__)
# os.environ["HF_HUB_TIMEOUT"] = "60"


class LLMClient:
    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = os.getenv("LLM_MODEL_NAME", "gpt2")
        self.model_name = model_name
        self.is_cloud = bool(os.getenv("CLOUD_ENV"))
        self.model_manager = get_model_manager()

        if not self.is_cloud:
            self.use_quantization = (
                os.getenv("USE_QUANTIZATION", "false").lower() == "true"
            )  # Default to False for better generation on M1
            self.quant_type = os.getenv("QUANT_TYPE", "dynamic")
            self.model, self.tokenizer = self.model_manager.load_model(
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


# Convenience function
def get_llm_client(model_name: Optional[str] = None) -> LLMClient:
    return LLMClient(model_name)
