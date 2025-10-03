"""
Advanced Model Manager for Transformers with caching, quantization, and M1 optimization.
"""
import gc
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

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
