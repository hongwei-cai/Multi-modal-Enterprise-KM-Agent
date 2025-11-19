"""Dashscope client adapter for qwen3-max.

Renamed from `aliyun_bailian_client` in earlier versions. Uses
`DASHSCOPE_API_KEY` environment variable for authentication and
`DASHSCOPE_URL` as an optional override for the inference endpoint.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests  # type: ignore

logger = logging.getLogger(__name__)


class DashscopeClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen3-max",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_URL", "https://dashscope.aliyun.example/v1/generate"
        )
        self.model = model

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health_check(self, timeout: float = 2.0) -> bool:
        try:
            resp = requests.get(self.base_url, timeout=timeout, headers=self._headers())
            return resp.status_code in (200, 204)
        except requests.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(kwargs or {})

        try:
            resp = requests.post(
                self.base_url, json=payload, headers=self._headers(), timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            # Extract predicted text; API shapes may vary
            if isinstance(data, dict):
                if "result" in data:
                    return str(data["result"]).strip()
                if "choices" in data and data["choices"]:
                    return str(data["choices"][0].get("text", "")).strip()
            return resp.text.strip()
        except requests.RequestException as e:
            logger.error("Dashscope generate failed: %s", e)
            raise
