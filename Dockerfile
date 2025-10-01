# =============================================
# Stage 1: Builder - Install Python dependencies
# =============================================

FROM python:3.10-slim AS builder

# ---- Environment Variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# ---- Install Minimal System Dependencies ----
# Only what's needed at build time: e.g., for torch, spacy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Setup Virtual Environment ----
RUN python -m venv /opt/venv
# Ensure venv is on PATH in this stage
ENV PATH="/opt/venv/bin:$PATH"

# ---- Copy Only Requirements First for Caching ----
WORKDIR /app
COPY requirements.txt .

# ---- Install Python Dependencies ----
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# =============================================
# Stage 2: Runtime - Minimal Production Image
# =============================================

FROM python:3.10-slim AS runtime

# ---- Environment Variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PORT=8000

# ---- Install Runtime-only System Deps (e.g., libgomp1 for torch) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy Virtual Environment from Builder ----
COPY --from=builder /opt/venv /opt/venv

# ---- Create and Set Working Directory ----
WORKDIR /app

# ---- Copy Application Code ----
# Order matters: copy least frequently changed files first for caching
COPY .env .
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# ---- Expose Port (Documentation only; not enforced) ----
EXPOSE 8000

# ---- Health Check (Optional but Recommended) ----
# NOTE: Ensure `/health` endpoint exists in your FastAPI app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ---- Default Command ----
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
