"""
Quantization Manager for PyTorch model quantization.
"""
import importlib
import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class QuantizationManager:
    """Manager for PyTorch model quantization operations.

    Supports PyTorch dynamic/static quantization and optional bitsandbytes
    4-bit/8-bit loading (when `bitsandbytes` is available). Also exposes
    simple helpers for calibration and QAT preparation.
    """

    def __init__(self):
        # Set quantization engine for ARM (M1)
        try:
            torch.backends.quantized.engine = "qnnpack"
        except Exception:
            # Some environments may not expose quantized backend
            logger.debug("Could not set torch quantized engine to qnnpack")

        # Detect optional bitsandbytes support
        self._bnb_available = importlib.util.find_spec("bitsandbytes") is not None

    def apply_pytorch_quantization(
        self, model: AutoModelForCausalLM, quant_type: str = "dynamic"
    ) -> AutoModelForCausalLM:
        """Apply PyTorch quantization to the model using torch.ao.quantization.

        quant_type supported values:
        - 'dynamic' : torch.ao.quantization.quantize_dynamic (qint8)
        - 'static'  : prepare + convert (requires calibration)
        - 'bnb_4bit'|'bnb_8bit' : bitsandbytes-backed quantized loading (noop here)
        """
        quant_type = (quant_type or "dynamic").lower()

        if quant_type == "dynamic":
            # Dynamic quantization: quantize weights on-the-fly (8-bit via qint8)
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization (qint8)")

        elif quant_type == "static":
            # Static quantization: requires calibration
            model.eval()
            model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
            torch.ao.quantization.prepare(model, inplace=True)
            # Note: calibration should be performed by calling `calibrate_static`
            torch.ao.quantization.convert(model, inplace=True)
            logger.info("Applied static quantization")

        elif quant_type in ("bnb_4bit", "bnb_8bit"):
            if not self._bnb_available:
                logger.warning(
                    "bitsandbytes not available; falling back to dynamic quantization"
                )
                return self.apply_pytorch_quantization(model, "dynamic")
            # bitsandbytes quantization is handled at load time via HF `from_pretrained`
            # with `load_in_4bit`/`load_in_8bit` flags. Nothing to do here.
            logger.info(f"bitsandbytes quantization requested: {quant_type}")

        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")

        return model

    def is_quantization_supported(self, model: AutoModelForCausalLM) -> bool:
        """Check if the model supports quantization.

        A heuristic check: model contains Linear layers which are quantizable.
        """
        has_linear_layers = any(
            isinstance(module, torch.nn.Linear) for module in model.modules()
        )
        return has_linear_layers

    def get_quantization_config(self, quant_type: str = "dynamic") -> dict:
        """Get quantization configuration for model loading.

        Returns a dictionary of keyword args that can be passed to
        `from_pretrained(..., **config)` when loading a model.
        Supported `quant_type` values: 'dynamic', 'static', 'bnb_4bit', 'bnb_8bit'
        """
        quant_type = (quant_type or "dynamic").lower()

        if quant_type == "dynamic":
            return {
                "dtype": torch.float32,
                "device_map": {"": "cpu"},
            }

        elif quant_type == "static":
            return {
                "dtype": torch.float32,
                "device_map": {"": "cpu"},
            }

        elif quant_type in ("bnb_4bit", "bnb_8bit"):
            # bitsandbytes-backed loading configuration
            if not self._bnb_available:
                raise RuntimeError(
                    "bitsandbytes is not installed; install it to use bnb_4bit/bnb_8bit"
                )

            config: dict[str, Any] = {
                "cache_dir": None,
                # Let HF/bitsandbytes decide device mapping; callers may override.
                "device_map": {"": "auto"},
            }

            if quant_type == "bnb_4bit":
                config.update(
                    {
                        "load_in_4bit": True,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                    }
                )
            else:
                # 8-bit loading via bitsandbytes
                config.update(
                    {"load_in_8bit": True, "bnb_8bit_compute_dtype": torch.float16}
                )

            return config

        else:
            raise ValueError(f"Unsupported quantization type: {quant_type}")

    # --- Simple calibration / QAT helpers ---
    def calibrate_static(
        self,
        model: AutoModelForCausalLM,
        calib_dataloader: Iterable,
        device: Optional[str] = None,
        max_batches: int = 10,
    ) -> None:
        """Run a small calibration loop for static quantization.

        This will run `model` in evaluation mode over `calib_dataloader` to collect
        activation statistics required by `torch.ao.quantization.convert`.
        """
        model.eval()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logger.info("Starting static calibration (max_batches=%d)", max_batches)
        with torch.no_grad():
            for i, batch in enumerate(calib_dataloader):
                if i >= max_batches:
                    break
                # Attempt to move tensors if present
                try:
                    inputs = batch.get("input_ids", None) or batch.get("inputs", None)
                    if inputs is not None:
                        inputs = inputs.to(device)
                        _ = model(inputs)
                except Exception:
                    # Best-effort: some dataloaders yield dicts with different keys
                    try:
                        # If batch is a tensor
                        if torch.is_tensor(batch):
                            _ = model(batch.to(device))
                    except Exception:
                        logger.debug("Calibration batch could not be processed")

        logger.info("Calibration complete")

    def prepare_qat(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Prepare a model for Quantization Aware Training (QAT).

        Returns the prepared model (in training mode). The caller should run
        a normal training loop (fine-tuning) and then call `convert_qat`.
        """
        try:
            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)
            logger.info("Model prepared for QAT")
            return model
        except Exception as e:
            logger.error("Failed to prepare model for QAT: %s", e)
            raise

    def convert_qat(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Convert a QAT-prepared and trained model to a quantized version."""
        try:
            model.eval()
            torch.ao.quantization.convert(model, inplace=True)
            logger.info("Converted QAT-trained model to quantized version")
            return model
        except Exception as e:
            logger.error("Failed to convert QAT model: %s", e)
            raise
