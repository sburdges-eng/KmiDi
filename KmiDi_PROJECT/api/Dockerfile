# KmiDi FastAPI Production Service - Multi-stage Docker Build
# ============================================================
# Optimized for production: minimal image size, security best practices

# Stage 1: Builder - Install dependencies and build packages
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies (only what's needed for compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
# Try production requirements first, fallback to API requirements
COPY requirements-production.txt* /build/ 2>/dev/null || true
COPY api/requirements.txt /build/api-requirements.txt

# Install Python dependencies (use production requirements if available, else API requirements)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    if [ -f /build/requirements-production.txt ]; then \
        echo "Using production requirements"; \
        pip install --no-cache-dir -r /build/requirements-production.txt; \
    else \
        echo "Using API requirements"; \
        pip install --no-cache-dir -r /build/api-requirements.txt; \
    fi

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim

# Metadata labels
LABEL maintainer="KmiDi Team"
LABEL description="KmiDi Music Generation API - Production FastAPI Service"
LABEL version="1.0.0"

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash kmidi && \
    mkdir -p /app /data /models && \
    chown -R kmidi:kmidi /app /data /models

WORKDIR /app

# Copy installed Python packages from builder (only runtime)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (as non-root user)
COPY --chown=kmidi:kmidi api/ ./api/
COPY --chown=kmidi:kmidi music_brain/ ./music_brain/
COPY --chown=kmidi:kmidi pyproject.toml ./
COPY --chown=kmidi:kmidi data/ ./data/ 2>/dev/null || true

# Switch to non-root user
USER kmidi

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    KELLY_AUDIO_DATA_ROOT=/data \
    LOG_LEVEL=INFO

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
