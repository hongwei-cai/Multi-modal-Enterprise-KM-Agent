"""
LoRA Manager for Parameter-Efficient Fine-Tuning

This module handles LoRA (Low-Rank Adaptation) fine-tuning and adapter management
for efficient model adaptation on memory-constrained devices like M1 Pro.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import LoRAConfig

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manager for LoRA fine-tuning and adapter operations."""

    def __init__(self, cache_dir: Path, device: str = "cpu"):
        """
        Initialize LoRA Manager.

        Args:
            cache_dir: Directory for caching LoRA adapters
            device: Target device for model operations
        """
        self.cache_dir = cache_dir
        self.device = device
        self.adapter_base_dir = cache_dir / "lora_adapters"
        self.adapter_base_dir.mkdir(parents=True, exist_ok=True)

    def apply_lora_to_model(
        self,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        lora_config: Optional[LoRAConfig] = None,
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """Apply LoRA configuration to a loaded model for efficient fine-tuning."""
        if lora_config is None:
            lora_config = LoRAConfig()  # Use default config optimized for M1 Pro

        # Create PEFT config
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            inference_mode=lora_config.inference_mode,
        )

        # Apply LoRA
        lora_model = get_peft_model(base_model, peft_config)

        # Move to optimal device
        lora_model.to(self.device)

        logger.info(
            f"LoRA applied. Trainable parameters:\
                {lora_model.print_trainable_parameters()}"
        )

        return lora_model, tokenizer

    def save_lora_adapter(
        self, lora_model: PeftModel, adapter_name: str, model_name: str
    ):
        """Save LoRA adapter weights."""
        adapter_dir = self.adapter_base_dir / model_name / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        lora_model.save_pretrained(str(adapter_dir))
        logger.info(
            f"LoRA adapter '{adapter_name}' saved for\
                model '{model_name}' at {adapter_dir}"
        )

    def load_lora_adapter(
        self,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        adapter_name: str,
        model_name: str,
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """Load a saved LoRA adapter onto the base model."""
        # Adapter path
        adapter_path = self.adapter_base_dir / model_name / adapter_name

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter '{adapter_name}' not found for model '{model_name}'"
            )

        # Load adapter
        lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        lora_model.to(self.device)

        logger.info(f"LoRA adapter '{adapter_name}' loaded for model '{model_name}'")

        return lora_model, tokenizer

    def get_optimal_lora_config(self, model_name: str) -> LoRAConfig:
        """Get optimal LoRA configuration for a specific model and M1 Pro hardware."""
        # Base config optimized for M1 Pro memory constraints
        base_config = LoRAConfig(
            r=8,  # Low rank for memory efficiency
            lora_alpha=16,  # Moderate scaling
            lora_dropout=0.05,  # Light regularization
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],  # Standard attention layers
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=True,
        )

        # Model-specific optimizations
        if "phi-2" in model_name.lower():
            # Phi-2 specific config
            base_config.target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
        elif "llama" in model_name.lower():
            # LLaMA specific config
            base_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gpt" in model_name.lower():
            # GPT-style models
            if "gpt-2" in model_name.lower() or "dialogpt" in model_name.lower():
                # DialoGPT/GPT-2 uses different layer names
                base_config.target_modules = ["c_attn", "c_proj"]
            else:
                base_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        return base_config

    def prepare_model_for_lora_training(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        model_name: str,
        use_quantization: bool = True,
    ) -> Tuple[PeftModel, AutoTokenizer]:
        """Prepare a model for LoRA training with optimal settings for M1 Pro."""
        # Prepare for k-bit training if quantized
        if use_quantization:
            model = prepare_model_for_kbit_training(model)

        # Get optimal LoRA config
        lora_config = self.get_optimal_lora_config(model_name)

        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )

        lora_model = get_peft_model(model, peft_config)
        lora_model.to(self.device)

        logger.info(
            f"Model {model_name} prepared for LoRA training. Trainable parameters:\
                {lora_model.print_trainable_parameters()}"
        )

        return lora_model, tokenizer

    def list_available_adapters(self, model_name: Optional[str] = None) -> dict:
        """List all available LoRA adapters."""
        adapters = {}

        if model_name:
            # List adapters for specific model
            model_dir = self.adapter_base_dir / model_name
            if model_dir.exists():
                adapters[model_name] = [
                    d.name for d in model_dir.iterdir() if d.is_dir()
                ]
            else:
                adapters[model_name] = []
        else:
            # List all adapters
            for model_dir in self.adapter_base_dir.iterdir():
                if model_dir.is_dir():
                    adapters[model_dir.name] = [
                        d.name for d in model_dir.iterdir() if d.is_dir()
                    ]

        return adapters


class LoRAConfigManager:
    """
    High-level manager for LoRA fine-tuning operations.

    This class provides a simplified interface for:
    - Loading base models with LoRA configuration
    - Applying LoRA adapters
    - Saving and loading trained adapters
    - Memory-efficient operations optimized for M1 Pro
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_save_path: str = "./adapters",
        device: Optional[str] = None,
    ):
        """
        Initialize LoRA configuration manager.

        Args:
            base_model_path: Path to the base model directory or HuggingFace model name
            adapter_save_path: Directory to save/load LoRA adapters
            device: Target device ('mps', 'cpu', 'cuda', or None for auto-detection)
        """
        self.base_model_path = Path(base_model_path)
        self.adapter_save_path = Path(adapter_save_path)
        self.adapter_save_path.mkdir(parents=True, exist_ok=True)

        # Device detection with M1 Pro optimization
        if device is None:
            import torch

            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Using device: {self.device}")

        # Default LoRA configuration optimized for M1 Pro
        self.lora_config = LoraConfig(
            r=8,  # Low rank for memory efficiency
            lora_alpha=16,  # Moderate scaling for adaptation strength
            lora_dropout=0.05,  # Light regularization
            bias="none",  # No bias adaptation for efficiency
            task_type="CAUSAL_LM",  # For causal language models
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],  # Standard attention layers
            inference_mode=True,  # Memory optimization for inference
        )

        # Model and tokenizer references
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_lora_applied = False

    def load_base_model(
        self,
        dtype=None,
        trust_remote_code: bool = False,
    ):
        """
        Load the base model and tokenizer with memory optimizations.

        Args:
            dtype: Model dtype (default: float16 for MPS, float32 for CPU)
            trust_remote_code: Whether to trust remote code in model files

        Returns:
            Tuple of (model, tokenizer)
        """
        import torch

        if dtype is None:
            dtype = torch.float16 if self.device == "mps" else torch.float32

        logger.info(f"Loading model from {self.base_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path),
            trust_remote_code=trust_remote_code,
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with memory optimizations
        model_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }

        # Device-specific loading
        if self.device == "mps":
            model_kwargs["device_map"] = {"": "mps"}
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        else:  # CPU
            model_kwargs["device_map"] = {"": "cpu"}

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_path), **model_kwargs
        )

        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model size: {self.model.num_parameters():,} parameters")

        return self.model, self.tokenizer

    def apply_lora(
        self,
        lora_config=None,
        target_modules=None,
    ):
        """
        Apply LoRA configuration to the loaded model.

        Args:
            lora_config: Custom LoRA configuration (uses default if None)
            target_modules: Override target modules for specific model architectures

        Returns:
            The model with LoRA applied
        """
        if self.model is None:
            raise ValueError("Load base model first using load_base_model()")

        if self.is_lora_applied:
            logger.warning("LoRA already applied to model")
            return self.model

        # Use custom config if provided
        config_to_use = lora_config or self.lora_config

        # Override target modules if specified
        if target_modules:
            config_to_use = LoraConfig(**config_to_use.__dict__)
            config_to_use.target_modules = target_modules

        # Apply LoRA
        self.model = get_peft_model(self.model, config_to_use)
        self.model.to(self.device)
        self.is_lora_applied = True

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params

        logger.info("LoRA applied successfully")
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)"
        )
        logger.info(f"Target modules: {config_to_use.target_modules}")

        return self.model

    def save_adapter(self, adapter_name: str, metadata: Optional[dict] = None) -> Path:
        """
        Save the trained LoRA adapter weights.

        Args:
            adapter_name: Name for the adapter
            metadata: Optional metadata to save with the adapter

        Returns:
            Path where adapter was saved
        """
        if self.model is None:
            raise ValueError("No model loaded")
        if not self.is_lora_applied:
            raise ValueError("LoRA not applied to model - nothing to save")

        save_path = self.adapter_save_path / adapter_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the adapter
        self.model.save_pretrained(str(save_path))

        # Save metadata if provided
        if metadata:
            import json

            metadata_path = save_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"LoRA adapter saved to {save_path}")
        return save_path

    def load_adapter(
        self,
        adapter_name: str,
        load_base_model: bool = True,
    ):
        """
        Load a saved LoRA adapter.

        Args:
            adapter_name: Name of the adapter to load
            load_base_model: Whether to load the base model first

        Returns:
            Tuple of (model, tokenizer)
        """
        adapter_path = self.adapter_save_path / adapter_name

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # Load base model if requested and not already loaded
        if load_base_model and self.model is None:
            self.load_base_model()

        if self.model is None:
            raise ValueError("Base model not loaded and load_base_model=False")

        # Load the adapter
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.to(self.device)
        self.is_lora_applied = True

        logger.info(f"LoRA adapter loaded from {adapter_path}")
        return self.model, self.tokenizer

    def get_trainable_parameters(self) -> dict:
        """Get information about trainable parameters."""
        if self.model is None:
            return {}

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "trainable_percentage": 100 * trainable_params / total_params,
        }

    def list_adapters(self) -> list[str]:
        """List all available adapters."""
        if not self.adapter_save_path.exists():
            return []

        return [
            item.name
            for item in self.adapter_save_path.iterdir()
            if item.is_dir() and (item / "adapter_model.bin").exists()
        ]

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_lora_applied = False

        # Force garbage collection
        import gc

        gc.collect()

        import torch

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model unloaded from memory")


# Convenience functions
def create_lora_manager_for_model(
    model_name: str,
    adapter_dir: str = "./adapters",
) -> LoRAConfigManager:
    """
    Convenience function to create a LoRA manager for a specific model.

    Args:
        model_name: HuggingFace model name or local path
        adapter_dir: Directory for adapter storage

    Returns:
        Configured LoRA manager
    """
    return LoRAConfigManager(
        base_model_path=model_name,
        adapter_save_path=adapter_dir,
    )


def quick_apply_lora(
    model_name: str, adapter_name: Optional[str] = None, **lora_kwargs
):
    """
    Quick utility to load a model and apply LoRA in one call.

    Args:
        model_name: Model to load
        adapter_name: Optional adapter to load after applying LoRA
        **lora_kwargs: Additional LoRA configuration options

    Returns:
        Tuple of (model, tokenizer)
    """
    manager = create_lora_manager_for_model(model_name)

    # Load model
    model, tokenizer = manager.load_base_model()

    # Apply LoRA
    model = manager.apply_lora(**lora_kwargs)

    # Load adapter if specified
    if adapter_name:
        model, tokenizer = manager.load_adapter(adapter_name, load_base_model=False)

    return model, tokenizer
