#!/usr/bin/env python3
"""Example script: generate with Ollama provider

Usage:
  python scripts/ollama_generate.py --prompt "Hello world"

Environment variables:
  OLLAMA_URL - URL of Ollama server (default: http://127.0.0.1:11434)
  OLLAMA_MODEL - model name (default: qwen3-8b)
"""
import argparse
import logging
import os

from src.rag.providers.ollama_client import OllamaClient


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-8b"))
    parser.add_argument(
        "--url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    )
    args = parser.parse_args()

    setup_logging()

    client = OllamaClient(base_url=args.url, model=args.model)
    try:
        resp = client.generate(args.prompt, max_tokens=256)
        print("=== Response ===")
        print(resp)
    except Exception as e:
        logging.error("Generation failed: %s", e)


if __name__ == "__main__":
    main()
