#!/usr/bin/env python3
"""Example script: generate with Dashscope provider

Usage:
  python scripts/dashscope_generate.py --prompt "Hello world"

Environment variables:
  DASHSCOPE_URL - Dashscope API URL
  DASHSCOPE_API_KEY - API key for Dashscope
  DASHSCOPE_MODEL - model name (default: qwen3-max)
"""
import argparse
import logging
import os

from src.rag.providers.dashscope_client import DashscopeClient


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=os.getenv("DASHSCOPE_MODEL", "qwen3-max"))
    parser.add_argument("--url", default=os.getenv("DASHSCOPE_URL", None))
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY", None))
    args = parser.parse_args()

    setup_logging()

    client = DashscopeClient(api_key=args.api_key, base_url=args.url, model=args.model)
    try:
        resp = client.generate(args.prompt, max_tokens=256)
        print("=== Response ===")
        print(resp)
    except Exception as e:
        logging.error("Generation failed: %s", e)


if __name__ == "__main__":
    main()
