"""Simple Ollama HTTP client adapter for qwen3-8b.

This adapter expects an environment variable `OLLAMA_URL` pointing to the
Ollama server (for example: http://127.0.0.1:11434). It exposes a small
`generate(prompt, **kwargs)` method returning text output.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests  # type: ignore

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, model: str = "qwen3-8b") -> None:
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.model = model

    def _endpoint(self) -> str:
        # Ollama API path for completions (varies by deployment); default to /api/completions
        return f"{self.base_url}/api/completions"

    def health_check(self, timeout: float = 2.0) -> bool:
        try:
            resp = requests.get(self.base_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Merge any provider-specific overrides
        payload.update(kwargs or {})

        try:
            resp = requests.post(self._endpoint(), json=payload, timeout=30)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                data = None

            # Ollama responses differ; try common keys
            if isinstance(data, dict):
                # Try standard structure
                if "choices" in data and data["choices"]:
                    return data["choices"][0].get("text", "").strip()
                if "output" in data:
                    # Some Ollama deployments return `output` as text
                    return str(data["output"]).strip()

            # Fallback to raw text when JSON is not present or doesn't have expected keys
            return resp.text.strip()
        except requests.RequestException as e:
            logger.error("Ollama generate failed: %s", e)
            raise
