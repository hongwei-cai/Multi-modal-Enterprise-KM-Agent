"""
LLM inference client: Local (Transformers) or Cloud (vLLM).
"""
import os
from typing import Optional

import requests  # type: ignore
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


class LLMClient:
    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = os.getenv("LLM_MODEL_NAME", "gpt2")
        self.model_name = model_name
        self.is_cloud = bool(os.getenv("CLOUD_ENV"))

        if self.is_cloud:
            self.api_url = os.getenv(
                "VLLM_API_URL", "http://localhost:8000/v1/chat/completions"
            )
        else:
            # Local: Load model with Transformers
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(
                self.device
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
            max_length = 100 if not self.is_cloud else 50  # Shorter for API
        if not (0 < temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        if not (0 < top_p <= 1):
            raise ValueError("Top-p must be between 0 and 1")

        if self.is_cloud:
            # Use vLLM API with parameters
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
            response.raise_for_status()
            response = response.json()["choices"][0]["message"]["content"]
        else:
            # Local generation with parameters
            print(f"DEBUG: Prompt length: {len(prompt)}")
            print(f"DEBUG: Prompt start: {prompt[:200]}...")  # First 200 chars
            inputs = self.tokenizer(prompt, return_tensors="pt")
            print(f"DEBUG: Tokenized inputs: {inputs}")

            try:
                outputs = self.model.generate(
                    **inputs,
                    max_length=1024,
                    max_new_tokens=100,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            except Exception as e:
                print(f"DEBUG: Generation error: {e}")
                return "Error: Generation failed"
            print(f"DEBUG: Generated outputs: {outputs}")

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"DEBUG: Decoded response: '{response}'")

            # Parse answer after "Answer:"
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            print(f"DEBUG: Final answer: '{response}'")

        # Post-process response
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response


# Convenience function
def get_llm_client(model_name: Optional[str] = None) -> LLMClient:
    return LLMClient(model_name)
