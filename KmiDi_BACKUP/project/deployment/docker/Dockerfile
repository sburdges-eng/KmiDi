# DAiW Music-Brain - Multi-stage Docker Build
# =============================================
# Supports development and production environments

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY music_brain/ ./music_brain/
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all]"

# Stage 2: Runtime
FROM python:3.11-slim

# Set labels
LABEL maintainer="Sean Burdges <seanblariat@gmail.com>"
LABEL description="DAiW Music-Brain - Music production intelligence toolkit"
LABEL version="0.2.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash daiw

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=daiw:daiw music_brain/ ./music_brain/
COPY --chown=daiw:daiw pyproject.toml README.md ./

# Switch to non-root user
USER daiw

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for Streamlit UI
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import music_brain; print('OK')" || exit 1

# Default command
CMD ["daiw", "--help"]
