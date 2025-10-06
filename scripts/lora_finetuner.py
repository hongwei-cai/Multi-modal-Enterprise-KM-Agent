#!/usr/bin/env python3
"""
Enhanced LoRA Fine-Tuning Script with Task Control

This script provides flexible LoRA operations including:
- Model preparation and LoRA application
- Adapter training and fine-tuning
- Adapter saving and loading
- Model evaluation and testing

Usage:
    python scripts/lora_finetuner.py --help
    python scripts/lora_finetuner.py --prepare --model microsoft/DialoGPT-medium
    python scripts/lora_finetuner.py --train --model microsoft/DialoGPT-medium \
        --dataset your_dataset
    python scripts/lora_finetuner.py --save-adapter --adapter-name my_adapter
    python scripts/lora_finetuner.py --load-adapter --adapter-name my_adapter --test
"""

import argparse
import json
import logging
import sys
from typing import Optional

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from src.rag.managers.model_manager import get_model_manager


class LoRAFinetuner:
    """Enhanced LoRA fine-tuning controller with task-based operations."""

    def __init__(self, verbose: bool = False):
        """Initialize the LoRA fine-tuner."""
        self.setup_logging(verbose)
        self.manager = get_model_manager()
        self.current_model = None
        self.current_tokenizer = None
        self.lora_model = None

    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def prepare_model(self, model_name: str, use_quantization: bool = True) -> bool:
        """Prepare and apply LoRA to a model."""
        try:
            self.logger.info(f"Preparing model: {model_name}")

            # Get optimal LoRA config
            lora_config = self.manager.get_optimal_lora_config(model_name)
            self.logger.info(
                f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}"
            )
            self.logger.info(f"Target modules: {lora_config.target_modules}")

            # Apply LoRA
            self.lora_model, self.current_tokenizer = self.manager.apply_lora_to_model(
                model_name=model_name,
                lora_config=lora_config,
                use_quantization=use_quantization,
            )

            # Also prepare for training
            self.current_model, _ = self.manager.prepare_model_for_lora_training(
                model_name=model_name, use_quantization=use_quantization
            )

            self.logger.info("Model preparation completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Model preparation failed: {e}")
            return False

    def save_adapter(self, adapter_name: str, model_name: Optional[str] = None) -> bool:
        """Save the current LoRA adapter."""
        if self.lora_model is None:
            self.logger.error("No LoRA model loaded. Run --prepare first.")
            return False

        try:
            model_name = model_name or "default_model"
            self.manager.save_lora_adapter(self.lora_model, adapter_name, model_name)
            self.logger.info(f"Adapter '{adapter_name}' saved successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save adapter: {e}")
            return False

    def load_adapter(
        self, adapter_name: str, model_name: str, use_quantization: bool = True
    ) -> bool:
        """Load a saved LoRA adapter."""
        try:
            self.lora_model, self.current_tokenizer = self.manager.load_lora_adapter(
                model_name=model_name,
                adapter_name=adapter_name,
                use_quantization=use_quantization,
            )
            self.logger.info(f"Adapter '{adapter_name}' loaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load adapter: {e}")
            return False

    def train_model(
        self,
        dataset_path: str,
        output_dir: str = "./lora_output",
        num_epochs: int = 2,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
    ) -> bool:
        """Train the LoRA model on a dataset."""
        if self.current_model is None:
            self.logger.error("No model prepared for training. Run --prepare first.")
            return False

        try:
            self.logger.info(f"Starting training on dataset: {dataset_path}")
            self.logger.info(
                f"Training parameters: epochs={num_epochs}, \
                             batch_size={batch_size}, lr={learning_rate}"
            )

            # Load dataset (you'll need to implement this based on your data format)
            train_dataset = self._load_dataset(dataset_path)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                evaluation_strategy="steps",
                eval_steps=500,
            )

            # Create trainer
            trainer = Trainer(
                model=self.current_model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.current_tokenizer,
            )

            # Train
            self.logger.info("Starting training...")
            trainer.train()

            # Update lora_model with trained weights
            self.lora_model = self.current_model

            self.logger.info("Training completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False

    def test_model(self, test_queries: Optional[list] = None) -> bool:
        """Test the current model with sample queries."""
        if self.lora_model is None:
            self.logger.error("No model loaded. Run --prepare or --load-adapter first.")
            return False

        try:
            self.logger.info("Testing model...")

            # Default test queries
            if test_queries is None:
                test_queries = [
                    "Hello, how are you?",
                    "What is artificial intelligence?",
                    "Tell me about machine learning.",
                ]

            for query in test_queries:
                self.logger.info(f"Query: {query}")
                # Generate response (you'll need to implement this \
                # based on your model type)
                response = self._generate_response(query)
                self.logger.info(f"Response: {response}")
                print(f"Q: {query}")
                print(f"A: {response}")
                print("-" * 50)

            return True

        except Exception as e:
            self.logger.error(f"Testing failed: {e}")
            return False

    def _load_dataset(self, dataset_path: str) -> Dataset:
        """Load dataset for training. Assumes JSON file with list of {'text': '...'}."""
        with open(dataset_path, "r") as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)

        def tokenize_function(example):
            tokenized = self.current_tokenizer(
                example["text"], truncation=True, padding="max_length"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def _generate_response(self, query: str) -> str:
        """Generate response for a query. Implement based on your model type."""
        # Placeholder - implement based on your model architecture
        self.logger.warning(
            "Response generation not implemented. \
                            Please implement _generate_response method."
        )
        return "Response generation not implemented yet."


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced LoRA Fine-Tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare a model for LoRA
  python scripts/lora_finetuner.py --prepare --model microsoft/DialoGPT-medium

  # Train a model
  python scripts/lora_finetuner.py --prepare --train --model \
    microsoft/DialoGPT-medium --dataset ./data/train.json

  # Save an adapter
  python scripts/lora_finetuner.py --prepare --save-adapter \
    --adapter-name my_adapter --model microsoft/DialoGPT-medium

  # Load and test an adapter
  python scripts/lora_finetuner.py --load-adapter --adapter-name my_adapter \
    --model microsoft/DialoGPT-medium --test

  # Full pipeline: prepare, train, save, test
  python scripts/lora_finetuner.py --prepare --train --save-adapter --test \
    --model microsoft/DialoGPT-medium --dataset ./data/train.json \
        --adapter-name trained_adapter
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Model name to use (default: microsoft/DialoGPT-medium)",
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        default=True,
        help="Use quantization (default: True)",
    )

    # Operations
    parser.add_argument(
        "--prepare", action="store_true", help="Prepare model with LoRA configuration"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--save-adapter", action="store_true", help="Save LoRA adapter after training"
    )
    parser.add_argument(
        "--load-adapter", action="store_true", help="Load a saved LoRA adapter"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model with sample queries"
    )

    # Training parameters
    parser.add_argument("--dataset", type=str, help="Path to training dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_output",
        help="Output directory for training (default: ./lora_output)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    # Adapter management
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="default_adapter",
        help="Name for saving/loading adapter (default: default_adapter)",
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any(
        [args.prepare, args.train, args.save_adapter, args.load_adapter, args.test]
    ):
        parser.error(
            "At least one operation must be specified (--prepare, \
                     --train, --save-adapter, --load-adapter, or --test)"
        )

    if args.train and not args.dataset:
        parser.error("--dataset is required when using --train")

    # Initialize fine-tuner
    finetuner = LoRAFinetuner(verbose=args.verbose)

    success = True

    # Execute operations in logical order
    try:
        # Load adapter if requested (must be first)
        if args.load_adapter:
            success &= finetuner.load_adapter(
                args.adapter_name, args.model, args.quantization
            )

        # Prepare model if requested
        if args.prepare:
            success &= finetuner.prepare_model(args.model, args.quantization)

        # Train if requested
        if args.train:
            success &= finetuner.train_model(
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )

        # Save adapter if requested
        if args.save_adapter:
            success &= finetuner.save_adapter(args.adapter_name, args.model)

        # Test if requested
        if args.test:
            success &= finetuner.test_model()

        if success:
            finetuner.logger.info("All operations completed successfully!")
            sys.exit(0)
        else:
            finetuner.logger.error("Some operations failed!")
            sys.exit(1)

    except Exception as e:
        finetuner.logger.error(f"Script execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
