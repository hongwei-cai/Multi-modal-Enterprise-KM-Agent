"""
Advanced Model Manager for Transformers with caching, quantization, and M1 optimization.
"""
import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import (
    ABTestConfig,
    BenchmarkResult,
    LoRAConfig,
    ModelConfig,
    ModelVersion,
    get_model_name,
)

from ..experiment_tracker import (
    ExperimentConfig,
    MLflowExperimentTracker,
    PerformanceMetrics,
)
from .experiment_manager import ExperimentManager
from .lora_manager import LoRAManager
from .quantization_manager import QuantizationManager

logger = logging.getLogger(__name__)


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
        # Initialize from configs.SUPPORTED_MODELS when present

        self.model_configs = self._initialize_model_configs()
        # ModelTier enum was removed from central config. Use None or
        # environment-driven selection via `configs.model_config.get_model_name()`
        self.current_tier = None

        self.model_versions: Dict[str, ModelVersion] = {}
        self.current_model: Optional[str] = None
        self.config_dir = Path(__file__).parent.parent.parent / "model_configs"
        self.config_dir.mkdir(exist_ok=True)
        self._load_model_versions()

        # Initialize experiment tracking
        self.experiment_tracker = MLflowExperimentTracker()

        # Initialize managers
        adapter_base_dir = Path(
            os.getenv("MODEL_CONFIGS_DIR", str(Path.cwd() / "model_configs"))
        )
        self.lora_manager = LoRAManager(cache_dir=adapter_base_dir, device=self.device)
        self.quantization_manager = QuantizationManager()
        self.experiment_manager = ExperimentManager(
            experiment_tracker=self.experiment_tracker
        )

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
        self, model: AutoModelForCausalLM, quant_type: Optional[str] = "dynamic"
    ) -> AutoModelForCausalLM:
        """Apply PyTorch quantization to the model using torch.ao.quantization."""
        return self.quantization_manager.apply_pytorch_quantization(model, quant_type)

    def load_model(
        self,
        model_name: str,
        use_quantization: bool = True,
        quant_type: Optional[str] = "dynamic",
        use_cache: bool = True,
    ) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model with advanced optimization and caching."""
        # If model_name references a remote provider (format: provider:model_id)
        # then we do not attempt to load a local HuggingFace model here. Return
        # a tuple of (None, None) and let higher-level code (LLMClient) use the
        # provider adapter for generation.
        if ":" in model_name:
            provider_prefix = model_name.split(":", 1)[0].lower()
            if provider_prefix in ("ollama", "dashscope", "aliyun", "aliyun_bailian"):
                logger.info(
                    "Model '%s' is a provider-backed model; skipping local load",
                    model_name,
                )
                return None, None

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

        # Track loading performance
        start_time = time.time()
        memory_before = psutil.virtual_memory().used / (1024**2)  # MB

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=str(self.cache_dir / "tokenizers")
            )

            # Set pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Determine quantization strategy when requested
            if use_quantization and (
                quant_type is None or str(quant_type).lower() in ("auto", "auto_select")
            ):
                quant_type = self.get_optimal_quantization(model_name)

            # Determine device: quantization requires CPU
            model_device = "cpu" if use_quantization else self.device

            # Load model with optimizations
            if use_quantization:
                model_kwargs = {
                    "cache_dir": str(self.cache_dir / "models"),
                    "low_cpu_mem_usage": True,
                    **self.quantization_manager.get_quantization_config(quant_type),
                }
            else:
                model_kwargs = {
                    "cache_dir": str(self.cache_dir / "models"),
                    "low_cpu_mem_usage": True,
                    "dtype": torch.bfloat16,
                    "device_map": {"": model_device},
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

            # Track loading performance
            end_time = time.time()
            memory_after = psutil.virtual_memory().used / (1024**2)  # MB
            loading_time = (end_time - start_time) * 1000  # ms
            memory_delta = memory_after - memory_before

            # Record performance metrics
            try:
                perf_metrics = PerformanceMetrics(
                    latency_ms=loading_time,
                    memory_usage_mb=memory_delta,
                    cpu_usage_percent=psutil.cpu_percent(),
                    throughput_tokens_per_sec=None,  # Not applicable for loading
                    response_quality_score=None,  # Not applicable for loading
                    error_rate=0.0,
                )

                run_id = self.experiment_tracker.start_experiment(
                    ExperimentConfig(
                        experiment_name=f"model_loading_{model_name.replace('/', '_')}",
                        run_name=f"load_{int(time.time())}",
                        model_name=model_name,
                        model_version=self.model_versions.get(
                            model_name,
                            ModelVersion(
                                "unknown",
                                ModelConfig("unknown", 0.0, 0.0, 0.0, "unknown"),
                                time.time(),
                            ),
                        ).version,
                        parameters={
                            "use_quantization": use_quantization,
                            "quant_type": quant_type if use_quantization else None,
                            "device": model_device,
                        },
                    )
                )

                self.experiment_tracker.log_metrics(run_id, perf_metrics)
                self.experiment_tracker.end_experiment(run_id)

            except Exception as e:
                logger.warning(f"Failed to track model loading performance: {e}")

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
        # If model_name references a provider (e.g. "ollama:qwen3-8b"),
        # prefer an HF tokenizer specified by the env var `TOKENIZER_MODEL`
        # or the HF base model for LoRA workflows.
        tokenizer_model = None
        if model_name is None:
            tokenizer_model = os.getenv("TOKENIZER_MODEL")
        else:
            if ":" in model_name:
                tokenizer_model = os.getenv("TOKENIZER_MODEL")
            else:
                tokenizer_model = model_name

        if not tokenizer_model:
            # Fallback to HF base (used for LoRA/finetuning)
            tokenizer_model = get_model_name(prefer_hf=True)

        cache_key = f"tokenizer_{tokenizer_model}"
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]["tokenizer"]

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model, cache_dir=str(self.cache_dir / "tokenizers")
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
        # Allow overriding embedding model via `EMBEDDING_MODEL` env var.
        from typing import Optional

        emb_model: Optional[str] = model_name
        if not emb_model or ":" in str(model_name):
            emb_model = os.getenv("EMBEDDING_MODEL")

        if not emb_model:
            emb_model = "sentence-transformers/all-MiniLM-L6-v2"

        cache_key = f"embedding_{emb_model}"
        if use_cache and cache_key in self.model_cache:
            return self.model_cache[cache_key]["model"]

        model = SentenceTransformer(emb_model, device=self.device)
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
        # Load supported models from central config if available, otherwise
        # fall back to an empty registry.
        try:
            from configs.model_config import SUPPORTED_MODELS

            return {k: v for k, v in SUPPORTED_MODELS.items()}
        except Exception:
            return {}

    # NOTE: `select_model_for_tier` removed — selection is performed directly
    # by `get_model_recommendation` below. This simplifies the codebase and
    # removes the old ModelTier dependency.

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
        return self.experiment_manager.benchmark_model(
            model_name=model_name,
            test_prompt=test_prompt,
            max_tokens=max_tokens,
            model_loader=self.load_model,
        )

    # get_model_recommendation removed — use configs.model_config.get_model_name

    def get_best_model_for_constraints(
        self,
        max_latency_ms: Optional[float] = None,
        max_memory_gb: Optional[float] = None,
    ) -> str:
        # get_best_model_for_constraints removed — use get_optimal_model
        # which prefers models under a memory budget and higher quality.
        return self.get_optimal_model(max_memory_gb=max_memory_gb)

    def load_model_with_fallback(
        self,
        model_name: str,
        use_quantization: bool = True,
        quant_type: str = "dynamic",
    ) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load model with automatic fallback to smaller\
            models under memory pressure."""
        original_model = model_name

        # If provider-backed, don't attempt fallback to other HF models
        if ":" in model_name:
            provider_prefix = model_name.split(":", 1)[0].lower()
            if provider_prefix in ("ollama", "dashscope", "aliyun", "aliyun_bailian"):
                logger.info(
                    "Provider-backed model '%s' requested; skipping fallback logic",
                    model_name,
                )
                return None, None

        # Check if we should downgrade
        if self.should_downgrade_model(model_name):
            downgrade_model = self.get_downgrade_model(model_name)
            if downgrade_model:
                logger.warning(
                    f"Memory pressure detected. Downgrading\
                        from {model_name} to {downgrade_model}"
                )
                model_name = downgrade_model
                # ModelConfig no longer exposes a 'tier' attribute; update current model instead
                self.current_model = model_name

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

    def _load_model_versions(self):
        """Load saved model versions from disk."""
        for config_file in self.config_dir.glob("*.json"):
            with open(config_file, "r") as f:
                data = json.load(f)
                version = ModelVersion(**data)
                self.model_versions[version.name] = version

    def switch_model(self, model_name: str) -> bool:
        """Seamlessly switch to a different model."""
        if model_name not in self.model_versions:
            return False

        # Unload current model if loaded
        if self.current_model and self.current_model in self.model_cache:
            self.unload_model(self.current_model)

        # Load new model
        try:
            model, tokenizer = self.load_model_with_fallback(model_name)
            self.current_model = model_name
            return True
        except Exception:
            return False

    def save_model_version(
        self, name: str, config: ModelConfig, metrics: Optional[Dict[str, float]] = None
    ):
        """Save a model version configuration."""
        version = ModelVersion(
            name=name,
            config=config,
            created_at=time.time(),
            performance_metrics=metrics or {},
        )
        self.model_versions[name] = version

        # Save to disk
        config_path = self.config_dir / f"{name}.json"
        # Serialize the ModelVersion to disk using available ModelConfig fields.
        with open(config_path, "w") as f:
            json.dump(
                {
                    "name": version.name,
                    "config": {
                        "name": getattr(version.config, "name", version.config.name),
                        "memory_gb": getattr(version.config, "memory_gb", None),
                        "latency_ms": getattr(version.config, "latency_ms", None),
                        "quality_score": getattr(version.config, "quality_score", None),
                        "description": getattr(version.config, "description", ""),
                    },
                    "created_at": version.created_at,
                    "performance_metrics": version.performance_metrics,
                },
                f,
                indent=2,
            )

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model version."""
        version = self.model_versions.get(model_name)
        return version.config if version else None

    def list_model_versions(self) -> List[str]:
        """List all available model versions."""
        return list(self.model_versions.keys())

    def start_ab_test(self, config: ABTestConfig):
        """Start an A/B test between two models."""
        self.experiment_manager.start_ab_test(config)

    def get_ab_test_model(self, test_name: str) -> Optional[str]:
        """Get the model to use for A/B testing based on traffic split."""
        return self.experiment_manager.get_ab_test_model(test_name)

    def record_ab_test_result(
        self, test_name: str, model_used: str, metrics: Dict[str, Any]
    ):
        """Record results from A/B test with MLflow tracking."""
        self.experiment_manager.record_ab_test_result(test_name, model_used, metrics)

    def apply_lora_to_model(
        self,
        model_name: str,
        lora_config: Optional[LoRAConfig] = None,
        use_quantization: bool = True,
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """Apply LoRA configuration to a model for efficient fine-tuning."""
        # Load base model
        base_model, tokenizer = self.load_model(
            model_name, use_quantization=use_quantization
        )

        # Ensure base_model and tokenizer are loaded (not provider-backed)
        assert (
            base_model is not None and tokenizer is not None
        ), "LoRA operations require a local HF model and tokenizer; \
            provider-backed models are not supported"

        # Delegate to LoRA manager
        return self.lora_manager.apply_lora_to_model(base_model, tokenizer, lora_config)

    # --- New convenience wrappers for runtime ops ---
    def on_the_fly_quantize(
        self, model: AutoModelForCausalLM, quant_type: str = "dynamic"
    ) -> AutoModelForCausalLM:
        """Apply on-the-fly quantization to an already-loaded model."""
        return self.quantization_manager.on_the_fly_quantize(
            model, quant_type=quant_type
        )

    def prune_model(
        self,
        model: AutoModelForCausalLM,
        amount: float = 0.2,
        method: str = "l1_unstructured",
    ) -> AutoModelForCausalLM:
        """Prune a model using simple heuristics and return the pruned model."""
        return self.quantization_manager.prune_model(
            model, amount=amount, method=method
        )

    def get_optimal_quantization(
        self, model_name: str, constraints: Optional[dict] = None
    ) -> str:
        """Get an optimal quantization strategy for a model using heuristics."""
        return self.quantization_manager.select_quantization_strategy(
            model_name=model_name, constraints=constraints
        )

    def save_lora_adapter(
        self, lora_model: PeftModel, adapter_name: str, model_name: str
    ):
        """Save LoRA adapter weights."""
        self.lora_manager.save_lora_adapter(lora_model, adapter_name, model_name)

    def load_lora_adapter(
        self, model_name: str, adapter_name: str, use_quantization: bool = True
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """Load a saved LoRA adapter onto the base model."""
        # Load base model
        base_model, tokenizer = self.load_model(
            model_name, use_quantization=use_quantization
        )

        # Ensure base_model and tokenizer are loaded (not provider-backed)
        assert (
            base_model is not None and tokenizer is not None
        ), "LoRA adapter loading requires a local HF base model and tokenizer; \
            provider models are not supported"

        # Delegate to LoRA manager
        return self.lora_manager.load_lora_adapter(
            base_model, tokenizer, adapter_name, model_name
        )

    def get_optimal_lora_config(self, model_name: str) -> LoRAConfig:
        """Get optimal LoRA configuration for a specific model and M1 Pro hardware."""
        return self.lora_manager.get_optimal_lora_config(model_name)

    def prepare_model_for_lora_training(
        self, model_name: str, use_quantization: bool = True
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """Prepare a model for LoRA training with optimal settings for M1 Pro."""
        # Load model with quantization for memory efficiency
        model, tokenizer = self.load_model(
            model_name, use_quantization=use_quantization
        )

        # Ensure model and tokenizer are present
        assert (
            model is not None and tokenizer is not None
        ), "Preparing for LoRA training requires a local HF model and tokenizer"

        # Delegate to LoRA manager
        return self.lora_manager.prepare_model_for_lora_training(
            model, tokenizer, model_name, use_quantization
        )

    def list_lora_adapters(self, model_name: Optional[str] = None) -> dict:
        """List all available LoRA adapters."""
        return self.lora_manager.list_available_adapters(model_name)


# Global instance for singleton pattern
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Recommended model for local LoRA development
# We keep this mapping minimal because runtime generation uses provider-backed
# models (Ollama / Dashscope). For LoRA fine-tuning iterations, prefer
# `ollama:qwen3-8b` running on a local Ollama server; use Hugging Face
# model artifacts when packaging adapters for distribution.
SUITABLE_MODELS = {
    "ollama:qwen3-8b": {
        "model": "qwen3-8b (Ollama)",
        "memory": "~8GB (server-side / local Ollama)",
        "finetuning": "Use Ollama for rapid iteration; export LoRA adapters via \
            HF-compatible format",
    }
}
