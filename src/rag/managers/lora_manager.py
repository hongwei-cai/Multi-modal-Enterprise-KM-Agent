"""
LoRA Manager for Parameter-Efficient Fine-Tuning

This module handles LoRA (Low-Rank Adaptation) fine-tuning and adapter management
for efficient model adaptation on memory-constrained devices like M1 Pro.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling parameter
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )  # Target attention modules
    lora_dropout: float = 0.05
    bias: str = "none"  # Bias handling
    task_type: str = "CAUSAL_LM"  # Task type for PEFT
    inference_mode: bool = True  # Use inference mode for memory efficiency


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
