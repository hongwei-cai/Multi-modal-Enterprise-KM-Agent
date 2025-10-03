import os
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRAConfigManager:
    def __init__(self, base_model_path: str, adapter_save_path: str = "./adapters"):
        """
        Initialize LoRA configuration for M1 Pro optimization.

        Args:
            base_model_path: Path to the base model (e.g., local quantized model).
            adapter_save_path: Directory to save/load LoRA adapters.
        """
        self.base_model_path = base_model_path
        self.adapter_save_path = adapter_save_path
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Optimal LoRA config for M1 Pro: Low rank for memory efficiency,\
        # moderate alpha for adaptation
        self.lora_config = LoraConfig(
            r=8,  # Low rank to reduce memory usage on M1 Pro
            lora_alpha=16,  # Scaling factor for adaptation strength
            lora_dropout=0.1,  # Dropout for regularization
            bias="none",  # No bias adaptation for efficiency
            task_type="CAUSAL_LM",  # For causal language models
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],  # Common attention layers for LoRA
        )

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def load_base_model(self):
        """Load the base model and tokenizer, optimized for M1 Pro."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency on M1 Pro
            device_map="auto",  # Auto-map to MPS if available
            low_cpu_mem_usage=True,
        )
        print(f"Base model loaded on device: {self.model.device}")

    def apply_lora(self):
        """Apply LoRA configuration to the base model for efficient adaptation."""
        if self.model is None:
            raise ValueError("Load base model first using load_base_model().")

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.to(self.device)
        print(
            f"LoRA applied. Trainable parameters:\
                {self.model.print_trainable_parameters()}"
        )

    def save_adapter(self, adapter_name: str = "lora_adapter"):
        """Save the LoRA adapter weights."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        save_path = os.path.join(self.adapter_save_path, adapter_name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        print(f"LoRA adapter saved to {save_path}")

    def load_adapter(self, adapter_name: str = "lora_adapter"):
        """Load a saved LoRA adapter onto the base model."""
        if self.model is None:
            self.load_base_model()

        if self.model is None:
            raise ValueError("Failed to load base model.")

        adapter_path = os.path.join(self.adapter_save_path, adapter_name)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(self.device)
        print(f"LoRA adapter loaded from {adapter_path}")


# Example usage:
# manager = LoRAConfigManager(base_model_path="./models/local-llama")
# manager.load_base_model()
# manager.apply_lora()
# # After fine-tuning...
# manager.save_adapter("my_fine_tuned_adapter")
# # Later, to load:
# manager.load_adapter("my_fine_tuned_adapter")
