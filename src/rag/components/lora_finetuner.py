"""
LoRA Fine-Tuning Pipeline

This streamlined component orchestrates the complete LoRA fine-tuning workflow,
integrating building blocks for dataset loading, model preparation, training,
and adapter management.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from configs.model_config import LoRAConfig as LoRAConfigModel
from src.rag.managers.lora_manager import LoRAManager

logger = logging.getLogger(__name__)


class LoRAFinetuner:
    """
    Streamlined component for LoRA fine-tuning workflow.

    This component orchestrates:
    - Dataset loading and preprocessing
    - Model preparation with LoRA
    - Training execution
    - Adapter saving
    """

    def __init__(
        self, adapter_save_path: Optional[Union[Path, str]] = None, device: str = "cpu"
    ):
        """
        Initialize the LoRA fine-tuner.

        Args:
            adapter_save_path: Directory for saving LoRA adapters
            (defaults to MODEL_CONFIGS_DIR/)
            device: Target device for training
        """
        self.device = device

        # Set adapter save path - default to MODEL_CONFIGS_DIR environment variable
        if adapter_save_path is None:
            adapter_save_path = os.getenv(
                "MODEL_CONFIGS_DIR", str(Path.cwd() / "model_configs")
            )

        if isinstance(adapter_save_path, str):
            adapter_save_path = Path(adapter_save_path)

        self.adapter_save_path = adapter_save_path
        self.adapter_save_path.mkdir(parents=True, exist_ok=True)

        self.lora_manager = LoRAManager(cache_dir=self.adapter_save_path, device=device)
        self.current_model: Optional[Any] = None
        self.current_tokenizer: Optional[Any] = None
        self.lora_model: Optional[Any] = None

    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load and preprocess dataset for training.

        Args:
            dataset_path: Path to the dataset (JSON format expected)

        Returns:
            Preprocessed dataset ready for training
        """
        logger.info(f"Loading dataset from {dataset_path}")

        # Load from disk if it's a saved dataset
        if Path(dataset_path).is_dir():
            dataset = load_from_disk(dataset_path)
        else:
            # Assume JSON format for now
            import json

            with open(dataset_path, "r") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)

        logger.info(f"Loaded dataset with {len(dataset)} examples")
        return dataset

    def prepare_model(
        self, model_name: str, lora_config: Optional[LoRAConfigModel] = None
    ) -> bool:
        """
        Prepare model for LoRA fine-tuning.

        Args:
            model_name: HuggingFace model name
            lora_config: LoRA configuration (uses optimal config if None)

        Returns:
            Success status
        """
        try:
            logger.info(f"Preparing model {model_name} for LoRA fine-tuning")

            # Load base model and tokenizer
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            self.current_tokenizer = AutoTokenizer.from_pretrained(model_name)
            assert self.current_tokenizer is not None
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

            # Get optimal LoRA config for this model
            if lora_config is None:
                lora_config = self.lora_manager.get_optimal_lora_config(model_name)

            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,  # type: ignore [arg-type]
                task_type=lora_config.task_type,
            )

            self.lora_model = get_peft_model(self.current_model, peft_config)
            assert self.lora_model is not None
            # Only use prepare_model_for_kbit_training for quantized models
            # For full precision models, we don't need this
            # self.lora_model = prepare_model_for_kbit_training(self.lora_model)

            # Ensure LoRA parameters require gradients
            for param in self.lora_model.parameters():
                if param.requires_grad:
                    param.requires_grad_(True)

            logger.info("Model preparation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Model preparation failed: {e}")
            return False

    def tokenize_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        """
        Tokenize dataset for training.

        Args:
            dataset: Raw dataset
            max_length: Maximum sequence length

        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset...")

        def tokenize_function(examples):
            # Handle different dataset formats
            if "instruction" in examples and "response" in examples:
                # Instruction-response format
                texts = [
                    f"Instruction: {instr}\nResponse: {resp}"
                    for instr, resp in zip(
                        examples["instruction"], examples["response"]
                    )
                ]
            elif "text" in examples:
                # Raw text format
                texts = examples["text"]
            else:
                raise ValueError(
                    "Dataset must contain 'instruction'/'response' or 'text' fields"
                )

            tokenized = self.current_tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            # For PyTorch tensors, use .clone() instead of .copy()
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
        )

        logger.info(f"Tokenization completed. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset

    def train(
        self,
        dataset: Dataset,
        output_dir: str = "./lora_output",
        num_epochs: int = 2,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        save_steps: int = 500,
        logging_steps: int = 10,
    ) -> bool:
        """
        Execute LoRA fine-tuning.

        Args:
            dataset: Tokenized dataset for training
            output_dir: Directory to save training outputs
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            save_steps: Steps between checkpoints
            logging_steps: Steps between logging

        Returns:
            Success status
        """
        try:
            logger.info("Starting LoRA fine-tuning...")
            logger.info(
                f"Training parameters: epochs={num_epochs}, "
                f"batch_size={batch_size}, lr={learning_rate}"
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                fp16=torch.cuda.is_available() and self.device == "cuda",
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_total_limit=2,
                eval_strategy="no",  # Skip evaluation for simplicity
                load_best_model_at_end=False,
                report_to=[],  # Disable wandb/tensorboard reporting
            )

            # Create trainer
            trainer = Trainer(
                model=self.lora_model,
                args=training_args,
                train_dataset=dataset,
                processing_class=self.current_tokenizer,  # Use processing_class
                # instead of deprecated tokenizer
            )

            # Train
            trainer.train()

            logger.info("LoRA fine-tuning completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def save_adapter(self, adapter_name: str, model_name: str) -> bool:
        """
        Save the trained LoRA adapter.

        Args:
            adapter_name: Name for the adapter
            model_name: Name of the base model

        Returns:
            Success status
        """
        try:
            if self.lora_model is None:
                logger.error("No trained model available")
                return False

            # Create model-specific directory structure:
            # MODEL_CONFIGS_DIR/model_name/adapter_name
            model_dir = self.adapter_save_path / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            adapter_path = model_dir / adapter_name

            self.lora_model.save_pretrained(str(adapter_path))

            logger.info(f"LoRA adapter saved to {adapter_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save adapter: {e}")
            return False

    def finetune_pipeline(
        self,
        model_name: str,
        dataset_path: str,
        adapter_name: str,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Complete LoRA fine-tuning pipeline.

        Args:
            model_name: HuggingFace model name
            dataset_path: Path to training dataset
            adapter_name: Name for saved adapter
            training_config: Training configuration overrides

        Returns:
            Success status
        """
        # Default training config
        default_config = {
            "num_epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "max_length": 512,
            "output_dir": os.getenv(
                "LORA_OUTPUT_DIR", "./lora_output"
            ),  # Training checkpoints and logs
        }

        if training_config:
            default_config.update(training_config)

        try:
            # Step 1: Load dataset
            dataset = self.load_dataset(dataset_path)

            # Step 2: Prepare model (uses optimal config automatically)
            if not self.prepare_model(model_name):
                return False

            # Step 3: Tokenize dataset
            tokenized_dataset = self.tokenize_dataset(
                dataset, max_length=cast(int, default_config["max_length"])
            )

            # Step 4: Train
            if not self.train(
                dataset=tokenized_dataset,
                output_dir=cast(str, default_config["output_dir"]),
                num_epochs=cast(int, default_config["num_epochs"]),
                batch_size=cast(int, default_config["batch_size"]),
                learning_rate=cast(float, default_config["learning_rate"]),
            ):
                return False

            # Step 5: Save adapter
            if not self.save_adapter(adapter_name, model_name):
                return False

            logger.info(
                f"LoRA fine-tuning pipeline completed successfully! "
                f"Adapter saved as '{adapter_name}'"
            )
            return True

        except Exception as e:
            logger.error(f"LoRA fine-tuning pipeline failed: {e}")
            return False


def get_lora_finetuner(
    adapter_save_path: Optional[Union[Path, str]] = None, device: str = "cpu"
) -> LoRAFinetuner:
    """
    Factory function to get LoRA fine-tuner instance.

    Args:
        adapter_save_path: Directory for saving LoRA adapters
        (defaults to MODEL_CONFIGS_DIR/)
        device: Target device

    Returns:
        LoRAFinetuner instance
    """
    return LoRAFinetuner(adapter_save_path=adapter_save_path, device=device)
