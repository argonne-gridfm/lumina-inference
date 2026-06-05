# =============================================================================
# lumina-inference Docker image
# =============================================================================
# Multi-stage build for a lean inference-only container.
#
# Build:
#   docker build -t lumina-inference .
#
# Build with GPU support (requires NVIDIA base image):
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn9-runtime -t lumina-inference:gpu .
#
# Run (CPU):
#   docker run --rm -it lumina-inference python -c "from lumina_inference import Modeler; print('OK')"
#
# Run (GPU):
#   docker run --rm --gpus all -it lumina-inference:gpu python examples/huggingface_inference.py
#
# Run tests:
#   docker run --rm lumina-inference pytest tests/test_flexible_ingestion.py -v
# =============================================================================

ARG BASE_IMAGE=python:3.11-slim

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies and package
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

WORKDIR /build

# System dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU by default; GPU users override BASE_IMAGE)
# Pin to a known-good CPU wheel index for reproducibility
RUN pip install --no-cache-dir \
        torch>=2.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and its dependencies
RUN pip install --no-cache-dir \
        torch_geometric>=2.4

# Copy full source (pyproject.toml uses src-layout; setuptools needs
# src/ present even to resolve packages.find).
COPY . .

# Install the package with the [release] extra so the resulting image
# can both run inference and exercise the uploader/test suite.
# Also install pytest for the test target.
RUN pip install --no-cache-dir ".[release]" pytest

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image with only installed packages
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (examples, tests, docs)
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY README.md LICENSE ./

# Create non-root user for security
RUN groupadd -r lumina && useradd -r -g lumina -d /app -s /sbin/nologin lumina \
    && chown -R lumina:lumina /app
USER lumina

# Default: show package version
CMD ["python", "-c", "import lumina_inference; print(f'lumina-inference v{lumina_inference.__version__}')"]
