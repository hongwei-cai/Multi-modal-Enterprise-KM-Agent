"""
Model Configuration Classes

This module contains all the data classes and enums used for model configuration,
benchmarking, versioning, A/B testing, and LoRA fine-tuning.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ModelTier(Enum):
    """Model performance tiers for quality-speed tradeoffs."""

    SPEED = "speed"  # Small, fast models for quick responses
    BALANCED = "balanced"  # Medium models balancing speed and quality
    QUALITY = "quality"  # Large models for high-quality responses


@dataclass
class ModelConfig:
    """Configuration for a model with performance characteristics."""

    name: str
    tier: ModelTier  # SPEED/BALANCED/QUALITY
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
    lora_alpha: int = 16  # LoRA scaling parameter
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )  # Target attention modules
    lora_dropout: float = 0.05
    bias: str = "none"  # Bias handling
    task_type: str = "CAUSAL_LM"  # Task type for PEFT
    inference_mode: bool = True  # Use inference mode for memory efficiency
