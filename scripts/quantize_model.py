#!/usr/bin/env python3
"""
Utility to load and (optionally) calibrate / convert models with quantization.

Examples:
    python scripts/quantize_model.py --model ollama:qwen3-8b
    --quant_type dynamic
  python scripts/quantize_model.py --model gpt2 --quant_type bnb_4bit
    --output_dir model_configs/quantized/gpt2_4bit

"""
import argparse
import logging
from pathlib import Path

from src.rag.managers.model_manager import get_model_manager


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(description="Quantize a model and save it")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument(
        "--quant_type",
        default="dynamic",
        choices=["dynamic", "static", "bnb_4bit", "bnb_8bit"],
        help="Quantization type",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to save quantized model"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Do not use model cache"
    )

    args = parser.parse_args()
    setup_logging()

    manager = get_model_manager()

    use_quant = True if args.quant_type else False

    logging.info("Loading model %s with quant_type=%s", args.model, args.quant_type)
    model, tokenizer = manager.load_model(
        args.model,
        use_quantization=use_quant,
        quant_type=args.quant_type,
        use_cache=not args.no_cache,
    )

    if model is None or tokenizer is None:
        logging.error(
            "Model '%s' appears to be provider-backed or unavailable locally. "
            "Quantization must be run on a local HF model.",
            args.model,
        )
        raise SystemExit(2)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        logging.info("Saving model to %s", out)
        try:
            model.save_pretrained(str(out))
            tokenizer.save_pretrained(str(out))
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error("Failed to save model: %s", e)


if __name__ == "__main__":
    main()
