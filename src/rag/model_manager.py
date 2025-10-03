"""
Advanced Model Manager for Transformers with caching, quantization, and M1 optimization.
"""
import gc
import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


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


class ModelManager:
    """Advanced model manager with caching, memory management,\
        and device optimization."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "km_agent_models"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_threshold_gb = 12.0  # Leave 4GB for system on 16GB M1 Pro
        self.device = self._detect_optimal_device()

        # Dynamic model selection
        self.model_configs = self._initialize_model_configs()
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        self.current_tier = ModelTier.BALANCED

    def _detect_optimal_device(self) -> str:
        """Detect the optimal device for M1 Pro."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU acceleration")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) for M1 Pro acceleration")
            return "mps"
        else:
            logger.warning(
                "Using CPU - consider upgrading to M1/M2 for better performance"
            )
            return "cpu"

    def _get_model_cache_key(self, model_name: str, quantization: bool = False) -> str:
        """Generate a unique cache key for the model configuration."""
        config_str = f"{model_name}_quantized_{quantization}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _check_memory_usage(self) -> float:
        """Check current memory usage in GB."""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        return memory_gb

    def _cleanup_memory(self):
        """Aggressive memory cleanup for M1 Pro constraints."""
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()
        logger.debug(
            f"Memory cleanup completed. Current usage:\
                {self._check_memory_usage():.2f}GB"
        )

    def apply_pytorch_quantization(
        self, model: AutoModelForCausalLM, quant_type: str = "dynamic"
    ) -> AutoModelForCausalLM:
        """Apply PyTorch quantization to the model using torch.ao.quantization."""
        # Set quantization engine for ARM (M1)
        torch.backends.quantized.engine = "qnnpack"

        if quant_type == "dynamic":
            # Dynamic quantization: quantize weights on-the-fly
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization (8-bit)")
        elif quant_type == "static":
            # Static quantization: requires calibration, more complex
            model.eval()
            model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
            torch.ao.quantization.prepare(model, inplace=True)
            # Note: In practice, you'd run calibration data here
            torch.ao.quantization.convert(model, inplace=True)
            logger.info("Applied static quantization")
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
        return model

    def load_model(
        self,
        model_name: str,
        use_quantization: bool = True,
        quant_type: str = "dynamic",
        use_cache: bool = True,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model with advanced optimization and caching."""

        cache_key = self._get_model_cache_key(model_name, use_quantization)

        # Check cache first
        if use_cache and cache_key in self.model_cache:
            logger.info(f"Loading model {model_name} from cache")
            cached = self.model_cache[cache_key]
            return cached["model"], cached["tokenizer"]

        # Memory check before loading
        current_memory = self._check_memory_usage()
        if current_memory > self.memory_threshold_gb:
            logger.warning(
                f"High memory usage ({current_memory:.2f}GB). Cleaning up..."
            )
            self._cleanup_memory()
            self.unload_unused_models()

        logger.info(f"Loading model {model_name} with quantization={use_quantization}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=str(self.cache_dir / "tokenizers")
            )

            # Set pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Determine device: quantization requires CPU
            model_device = "cpu" if use_quantization else self.device

            # Load model with optimizations
            model_kwargs = {
                "cache_dir": str(self.cache_dir / "models"),
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.bfloat16
                if not use_quantization
                else torch.float32,  # Quantization works with float32
                "device_map": {"": model_device},  # Use CPU for quantized models
            }

            # Determine model type
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name)
            if config.model_type in ["t5", "mt5", "bart", "pegasus"]:
                from transformers import AutoModelForSeq2SeqLM

                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, **model_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

            # Apply PyTorch quantization if requested
            if use_quantization:
                model = self.apply_pytorch_quantization(model, quant_type)

            # Cache the loaded model
            if use_cache:
                self.model_cache[cache_key] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "last_used": torch.cuda.Event() if self.device == "cuda" else None,
                }

            logger.info(f"Successfully loaded model {model_name} on {model_device}")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def unload_model(self, model_name: str, quantization: bool = False):
        """Unload a specific model from memory."""
        cache_key = self._get_model_cache_key(model_name, quantization)
        if cache_key in self.model_cache:
            del self.model_cache[cache_key]
            self._cleanup_memory()
            logger.info(f"Unloaded model {model_name}")

    def unload_unused_models(self, max_models: int = 2):
        """Unload least recently used models to free memory."""
        if len(self.model_cache) <= max_models:
            return

        # Sort by last used (simplified - in real implementation, track timestamps)
        items_to_remove = list(self.model_cache.keys())[max_models:]
        for key in items_to_remove:
            # model_info = self.model_cache[key]
            del self.model_cache[key]
            logger.info("Unloaded unused model from cache")

        self._cleanup_memory()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name in SUITABLE_MODELS:
            return SUITABLE_MODELS[model_name].copy()
        else:
            return {
                "model": model_name,
                "memory": "Unknown",
                "finetuning": "Unknown - verify compatibility",
            }

    def list_cached_models(self) -> list:
        """List currently cached models."""
        return list(self.model_cache.keys())

    def clear_cache(self):
        """Clear all cached models."""
        self.model_cache.clear()
        self._cleanup_memory()
        logger.info("Cleared model cache")

    def load_tokenizer(self, model_name: str, use_cache: bool = True) -> AutoTokenizer:
        """Load tokenizer with caching."""
        cache_key = f"tokenizer_{model_name}"
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]["tokenizer"]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(self.cache_dir / "tokenizers")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_cache:
            self.model_cache[cache_key] = {"tokenizer": tokenizer}
        return tokenizer

    def load_embedding_model(
        self, model_name: str, use_cache: bool = True
    ) -> SentenceTransformer:
        """Load embedding model with caching and device optimization."""
        cache_key = f"embedding_{model_name}"
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]["model"]

        model = SentenceTransformer(model_name, device=self.device)
        if use_cache:
            self.model_cache[cache_key] = {"model": model}
        return model

    def load_spacy_model(
        self, model_name: str = "en_core_web_sm", use_cache: bool = True
    ) -> Any:
        """Load spaCy model with caching."""
        cache_key = f"spacy_{model_name}"
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]["model"]

        try:
            import spacy

            model = spacy.load(model_name)
            if use_cache:
                self.model_cache[cache_key] = {"model": model}
            return model
        except Exception as e:
            logger.error(f"Failed to load spaCy model {model_name}: {e}")
            raise

    def load_jieba(self, use_cache: bool = True) -> Any:
        """Load jieba library with caching."""
        cache_key = "jieba"
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]["library"]

        try:
            import jieba

            if use_cache:
                self.model_cache[cache_key] = {"library": jieba}
            return jieba
        except ImportError as e:
            logger.error(
                f"Failed to import jieba: {e}. Install with 'pip install jieba'"
            )
            raise

    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations with quality-speed tradeoffs."""
        return {
            "google/flan-t5-small": ModelConfig(
                name="google/flan-t5-small",
                tier=ModelTier.FAST,
                memory_gb=1.2,
                latency_ms=60,
                quality_score=0.7,
                description="Fast T5 model optimized for QA tasks",
            ),
            "google/flan-t5-base": ModelConfig(
                name="google/flan-t5-base",
                tier=ModelTier.BALANCED,
                memory_gb=2.5,
                latency_ms=100,
                quality_score=0.85,
                description="Balanced T5 model with good QA performance",
            ),
            "microsoft/DialoGPT-small": ModelConfig(
                name="microsoft/DialoGPT-small",
                tier=ModelTier.FAST,
                memory_gb=1.0,
                latency_ms=50,
                quality_score=0.6,
                description="Small, fast conversational model",
            ),
            "microsoft/DialoGPT-medium": ModelConfig(
                name="microsoft/DialoGPT-medium",
                tier=ModelTier.BALANCED,
                memory_gb=2.0,
                latency_ms=80,
                quality_score=0.8,
                description="Balanced conversational model",
            ),
            "microsoft/DialoGPT-large": ModelConfig(
                name="microsoft/DialoGPT-large",
                tier=ModelTier.QUALITY,
                memory_gb=4.0,
                latency_ms=120,
                quality_score=0.9,
                description="Large conversational model",
            ),
            "microsoft/phi-2": ModelConfig(
                name="microsoft/phi-2",
                tier=ModelTier.QUALITY,
                memory_gb=5.0,
                latency_ms=150,
                quality_score=0.95,
                description="High-performance general-purpose model",
            ),
        }

    def select_model_for_tier(self, tier: ModelTier) -> str:
        """Select the best model for a given performance tier."""
        tier_models = [
            config for config in self.model_configs.values() if config.tier == tier
        ]
        if not tier_models:
            # Fallback to any available model
            return list(self.model_configs.keys())[0]

        # Select model with best quality for the tier
        return max(tier_models, key=lambda x: x.quality_score).name

    def get_optimal_model(
        self, max_memory_gb: Optional[float] = None, min_quality: Optional[float] = None
    ) -> str:
        """Get the optimal model based on constraints."""
        if max_memory_gb is None:
            max_memory_gb = self.memory_threshold_gb

        available_models = [
            config
            for config in self.model_configs.values()
            if config.memory_gb <= max_memory_gb
        ]

        if not available_models:
            # Return smallest model if none fit
            return min(self.model_configs.values(), key=lambda x: x.memory_gb).name

        if min_quality is not None:
            quality_models = [
                m for m in available_models if m.quality_score >= min_quality
            ]
            if quality_models:
                available_models = quality_models

        # Return model with best quality that fits constraints
        return max(available_models, key=lambda x: x.quality_score).name

    def should_downgrade_model(self, current_model: str) -> bool:
        """Check if we should downgrade to a smaller model due to memory pressure."""
        current_memory = self._check_memory_usage()
        memory_pressure = (
            current_memory > self.memory_threshold_gb * 0.8
        )  # 80% threshold

        if not memory_pressure:
            return False

        if current_model not in self.model_configs:
            return False

        current_config = self.model_configs[current_model]

        # Check if there's a smaller model available
        smaller_models = [
            config
            for config in self.model_configs.values()
            if config.memory_gb < current_config.memory_gb
        ]

        return len(smaller_models) > 0

    def get_downgrade_model(self, current_model: str) -> Optional[str]:
        """Get the best downgrade model for memory pressure."""
        if current_model not in self.model_configs:
            return None

        current_config = self.model_configs[current_model]

        # Find smaller models
        smaller_models = [
            config
            for config in self.model_configs.values()
            if config.memory_gb < current_config.memory_gb
        ]

        if not smaller_models:
            return None

        # Return the largest of the smaller models (best quality downgrade)
        return max(smaller_models, key=lambda x: x.quality_score).name

    def benchmark_model(
        self,
        model_name: str,
        test_prompt: str = "Hello, how are you?",
        max_tokens: int = 50,
    ) -> BenchmarkResult:
        """Benchmark a model's performance."""
        import time

        start_memory = self._check_memory_usage()
        start_time = time.time()

        try:
            # Load model if not cached
            model, tokenizer = self.load_model(model_name, use_quantization=True)

            # Tokenize input
            inputs = tokenizer(test_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Deterministic for benchmarking
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Calculate metrics
            end_time = time.time()
            end_memory = self._check_memory_usage()

            latency_ms = (end_time - start_time) * 1000
            memory_usage_gb = end_memory - start_memory
            generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tokens_per_second = (
                generated_tokens / (end_time - start_time)
                if (end_time - start_time) > 0
                else 0
            )

            result = BenchmarkResult(
                model_name=model_name,
                latency_ms=latency_ms,
                memory_usage_gb=memory_usage_gb,
                tokens_per_second=tokens_per_second,
            )

            # Store benchmark result
            self.benchmark_results[model_name] = result

            logger.info(
                f"Benchmarked {model_name}: {latency_ms:.2f}ms,\
                    {tokens_per_second:.2f} tokens/sec"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to benchmark {model_name}: {e}")
            # Return a failed benchmark result
            return BenchmarkResult(
                model_name=model_name,
                latency_ms=float("inf"),
                memory_usage_gb=0,
                tokens_per_second=0,
            )

    def get_model_recommendation(self, priority: str = "balanced") -> str:
        """Get model recommendation based on priority (speed/quality/balanced)."""
        if priority == "speed":
            return self.select_model_for_tier(ModelTier.FAST)
        elif priority == "quality":
            return self.select_model_for_tier(ModelTier.QUALITY)
        else:  # balanced
            return self.select_model_for_tier(ModelTier.BALANCED)

    def get_best_model_for_constraints(
        self,
        max_latency_ms: Optional[float] = None,
        max_memory_gb: Optional[float] = None,
    ) -> str:
        """Get the best model that meets the given constraints."""
        candidates = list(self.model_configs.values())

        if max_latency_ms is not None:
            candidates = [c for c in candidates if c.latency_ms <= max_latency_ms]

        if max_memory_gb is not None:
            candidates = [c for c in candidates if c.memory_gb <= max_memory_gb]

        if not candidates:
            # Return smallest model as fallback
            return min(self.model_configs.values(), key=lambda x: x.memory_gb).name

        # Return highest quality model that meets constraints
        return max(candidates, key=lambda x: x.quality_score).name

    def load_model_with_fallback(
        self,
        model_name: str,
        use_quantization: bool = True,
        quant_type: str = "dynamic",
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model with automatic fallback to smaller\
            models under memory pressure."""
        original_model = model_name

        # Check if we should downgrade
        if self.should_downgrade_model(model_name):
            downgrade_model = self.get_downgrade_model(model_name)
            if downgrade_model:
                logger.warning(
                    f"Memory pressure detected. Downgrading\
                        from {model_name} to {downgrade_model}"
                )
                model_name = downgrade_model
                self.current_tier = self.model_configs[model_name].tier

        try:
            return self.load_model(model_name, use_quantization, quant_type)
        except Exception as e:
            # If loading fails, try progressively smaller models
            logger.error(f"Failed to load {model_name}: {e}")

            available_models = sorted(
                self.model_configs.values(), key=lambda x: x.memory_gb
            )

            for config in available_models:
                # Get current model config or create a dummy one with infinite memory
                current_config = self.model_configs.get(original_model)
                if current_config is None:
                    current_memory_gb = float("inf")
                else:
                    current_memory_gb = current_config.memory_gb

                if config.memory_gb < current_memory_gb:
                    try:
                        logger.info(f"Trying fallback model: {config.name}")
                        return self.load_model(
                            config.name, use_quantization, quant_type
                        )
                    except Exception as fallback_e:
                        logger.error(
                            f"Fallback model {config.name} also failed: {fallback_e}"
                        )
                        continue

            # If all fallbacks fail, raise original error
            raise e


# Global instance for singleton pattern
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Suitable models for M1 Pro PEFT tuning (16GB memory optimized)
SUITABLE_MODELS = {
    "microsoft/DialoGPT-medium": {
        "model": "microsoft/DialoGPT-medium",  # 345M parameters
        "memory": "~2GB (CPU with quantization)",
        "finetuning": "fast convergence, very suitable",
    },
    "microsoft/phi-2": {
        "model": "microsoft/phi-2",  # 2.7B parameters
        "memory": "~5GB (CPU with quantization)",
        "finetuning": "excellent balance, strong reasoning ability",
    },
    "meta-llama/Llama-2-7b-hf": {
        "model": "meta-llama/Llama-2-7b-hf",  # 7B parameters
        "memory": "~8GB (CPU with quantization)",
        "finetuning": "high performance, requires sufficient memory",
    },
    "microsoft/DialoGPT-large": {
        "model": "microsoft/DialoGPT-large",  # 762M parameters
        "memory": "~4GB (CPU with quantization)",
        "finetuning": "excellent dialogue quality",
    },
}
