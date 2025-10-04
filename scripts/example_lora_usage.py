#!/usr/bin/env python3
"""
Example script demonstrating LoRA fine-tuning with the ModelManager.
This shows how to apply LoRA to models and manage adapters for efficient fine-tuning.
"""

from src.rag.model_manager import get_model_manager


def main():
    """Demonstrate LoRA functionality with the ModelManager."""

    # Get the model manager instance
    manager = get_model_manager()

    # Example model (using a small model for demonstration)
    model_name = "microsoft/DialoGPT-medium"

    print(f"Setting up LoRA for {model_name}...")

    # Get optimal LoRA config for this model
    lora_config = manager.get_optimal_lora_config(model_name)
    print(f"Optimal LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(f"Target modules: {lora_config.target_modules}")

    # Apply LoRA to the model
    try:
        lora_model, tokenizer = manager.apply_lora_to_model(
            model_name=model_name, lora_config=lora_config, use_quantization=True
        )
        print("LoRA applied successfully!")

        # Example: Save the adapter
        adapter_name = "example_adapter"
        manager.save_lora_adapter(lora_model, adapter_name, model_name)
        print(f"Adapter saved as '{adapter_name}'")

        # Example: Load the adapter back
        loaded_model, loaded_tokenizer = manager.load_lora_adapter(
            model_name=model_name, adapter_name=adapter_name, use_quantization=True
        )
        print("Adapter loaded successfully!")

        # Example: Prepare model for training
        training_model, training_tokenizer = manager.prepare_model_for_lora_training(
            model_name=model_name, use_quantization=True
        )
        print("Model prepared for LoRA training!")

    except Exception as e:
        print(f"Error during LoRA setup: {e}")
        print("Note: This is expected if the model isn't downloaded yet.")
        print("The LoRA methods are ready to use once models are available.")


if __name__ == "__main__":
    main()
