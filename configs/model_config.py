"""
Model Configuration Classes

This module contains dataclasses used for model configuration,
benchmarking, versioning, A/B testing, and LoRA fine-tuning.

This simplified variant removes the previous `ModelTier` enum and
provides a small helper to select between the local Ollama model and
Dashscope cloud model. It also exposes the HF base model name for
LoRA/PEFT workflows.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a model with performance characteristics.

    Note: This dataclass deliberately does not include a tier enum.
    It stores concrete resource estimates and a short description.
    """

    name: str
    memory_gb: float
    latency_ms: float  # Estimated latency per token
    quality_score: float  # Relative quality score (0-1)
    description: str


@dataclass
class BenchmarkSummary:
    """Summary of benchmarking results for a model."""

    model_type: str
    load_time: float
    avg_latency: float
    avg_response_length: float
    success_rate: float
    total_queries: int
    results: List[Dict[str, Any]]


@dataclass
class BenchmarkResult:
    """Individual benchmark result for a model."""

    model_name: str
    latency_ms: float
    memory_usage_gb: float
    tokens_per_second: float


@dataclass
class ModelVersion:
    """Represents a model version with configuration."""

    name: str
    config: ModelConfig
    created_at: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def version(self) -> str:
        """Get the version string."""
        return self.name


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    test_name: str
    model_a: str
    model_b: str
    traffic_split: float = 0.5  # 50/50 split
    duration_hours: int = 24
    metrics: List[str] = field(default_factory=lambda: ["latency", "quality"])
    test_queries: List[str] = field(default_factory=list)
    vector_db_path: str = ""


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 16  # LoRA rank
    lora_alpha: int = 8  # LoRA scaling parameter
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"  # Bias handling
    task_type: str = "CAUSAL_LM"  # Task type for PEFT
    inference_mode: bool = True  # Use inference mode for memory efficiency


# Supported provider-backed models
SUPPORTED_MODELS = {
    "ollama:qwen3-8b": ModelConfig(
        name="ollama:qwen3-8b",
        memory_gb=8.0,
        latency_ms=150.0,
        quality_score=0.9,
        description="Qwen3 8B served via Ollama (local or remote)",
    ),
    "dashscope:qwen3-max": ModelConfig(
        name="dashscope:qwen3-max",
        memory_gb=16.0,
        latency_ms=200.0,
        quality_score=0.98,
        description="Qwen3-max served via Dashscope (production hosted provider)",
    ),
}

# Default model names (can be overridden via environment variables)
DEFAULT_LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "ollama:qwen3-8b")
DEFAULT_CLOUD_MODEL = os.getenv("DASHSCOPE_MODEL", "dashscope:qwen3-max")
# Hugging Face base model used for LoRA/PEFT training
HF_BASE_MODEL = os.getenv("HF_MODEL_FOR_FINETUNE", "qwen/qwen-3-8b")


def _detect_deployment_mode() -> str:
    """Detect whether we should select ``local`` or ``cloud`` models.

    Priority (highest -> lowest):
    - If ``LLM_PROVIDER`` env var is set to either "ollama" or "dashscope", use it.
    - If ``CLOUD_ENV`` is set (truthy), treat as cloud.
    - Otherwise default to local.
    """
    prov = os.getenv("LLM_PROVIDER")
    if prov:
        prov = prov.lower()
        if prov.startswith("dashscope"):
            return "cloud"
        if prov.startswith("ollama"):
            return "local"

    if os.getenv("CLOUD_ENV"):
        return "cloud"

    return "local"


def get_model_name(
    provider_override: Optional[str] = None, prefer_hf: bool = False
) -> str:
    """Return the effective model name.

    - If ``provider_override`` is a provider-prefixed full model string
    (contains ':'), it is returned.
    - If ``provider_override`` is "local" or "cloud" it selects the corresponding default.
    - If ``prefer_hf`` is True, returns the HuggingFace base model name (for LoRA training).
    - Otherwise selects based on environment via ``_detect_deployment_mode()``.
    """
    if provider_override:
        if ":" in provider_override:
            return provider_override
        if provider_override in ("local", "cloud"):
            mode = provider_override
        else:
            mode = _detect_deployment_mode()
    else:
        mode = _detect_deployment_mode()

    if prefer_hf:
        return HF_BASE_MODEL

    return DEFAULT_CLOUD_MODEL if mode == "cloud" else DEFAULT_LOCAL_MODEL
