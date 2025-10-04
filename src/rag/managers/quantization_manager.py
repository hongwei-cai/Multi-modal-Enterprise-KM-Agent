"""
Quantization Manager for PyTorch model quantization.
"""
import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QuantizationManager:
    """Manager for PyTorch model quantization operations."""

    def __init__(self):
        # Set quantization engine for ARM (M1)
        torch.backends.quantized.engine = "qnnpack"

    def apply_pytorch_quantization(
        self, model: AutoModelForCausalLM, quant_type: str = "dynamic"
    ) -> AutoModelForCausalLM:
        """Apply PyTorch quantization to the model using torch.ao.quantization."""
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

    def is_quantization_supported(self, model: AutoModelForCausalLM) -> bool:
        """Check if the model supports quantization."""
        # Check if model has quantization-compatible layers
        has_linear_layers = any(
            isinstance(module, torch.nn.Linear) for module in model.modules()
        )
        return has_linear_layers

    def get_quantization_config(self, quant_type: str = "dynamic") -> dict:
        """Get quantization configuration for model loading."""
        if quant_type == "dynamic":
            return {
                "torch_dtype": torch.float32,  # Quantization works with float32
                "device_map": {"": "cpu"},  # Use CPU for quantized models
            }
        elif quant_type == "static":
            return {
                "torch_dtype": torch.float32,
                "device_map": {"": "cpu"},
            }
        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
