"""
Model Configuration Classes

This module contains all the data classes and enums used for model configuration,
benchmarking, versioning, and A/B testing.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelTier(Enum):
    """Model performance tiers for quality-speed tradeoffs."""

    FAST = "fast"  # Small, fast models for quick responses
    BALANCED = "balanced"  # Medium models balancing speed and quality
    QUALITY = "quality"  # Large models for high-quality responses


@dataclass
class ModelConfig:
    """Configuration for a model with performance characteristics."""

    name: str
    tier: ModelTier
    memory_gb: float
    latency_ms: float  # Estimated latency per token
    quality_score: float  # Relative quality score (0-1)
    description: str


@dataclass
class BenchmarkResult:
    """Result of model performance benchmarking."""

    model_name: str
    latency_ms: float
    memory_usage_gb: float
    tokens_per_second: float
    quality_score: Optional[float] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


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
