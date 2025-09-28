"""
Startup script for vLLM server with DialoGPT-medium model.
Run: python scripts/start_vllm_server.py
"""
import os
import subprocess
import sys

import torch


def get_model_name():
    """Get model name from env var or auto-detect environment."""
    # Check for explicit env var
    model_name = os.getenv("LLM_MODEL_NAME")
    if model_name:
        return model_name

    # Auto-detect: Use cloud model if CLOUD_ENV is set, else local
    if os.getenv("CLOUD_ENV"):
        return "BAAI/bge-m3"  # Or your preferred cloud model (e.g., GPT-2, Llama)
    else:
        return "microsoft/DialoGPT-medium"  # Local default


def start_vllm_server():
    if not os.getenv("CLOUD_ENV"):
        print("vLLM is for cloud deployment. For local, use LLMClient directly.")
        return

    """Start vLLM server with selected model."""
    model_name = get_model_name()
    host = os.getenv("VLLM_HOST", "0.0.0.0")
    port = int(os.getenv("VLLM_PORT", "8000"))
    gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY", "0.8"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "1024"))

    # Detect device: MPS for M1, CUDA for GPU, CPU otherwise
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Command to run vLLM server
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--device",
        device,  # Enable MPS acceleration
    ]

    print(f"Starting vLLM server with model: {model_name} on device: {device}")
    print(f"Server will be available at http://{host}:{port}")
    print("Press Ctrl+C to stop.")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting vLLM server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server stopped.")


if __name__ == "__main__":
    start_vllm_server()
