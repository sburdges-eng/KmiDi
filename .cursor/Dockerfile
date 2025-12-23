FROM python:3.11-slim

# Create non-root user (best practice for cloud agents)
RUN useradd -m -s /bin/bash ubuntu

# Install system dependencies (minimal for iDAW/CLI tooling + JUCE GUI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libasound2-dev \
        libsndfile1-dev \
        libffi-dev \
        ninja-build \
        pkg-config \
        curl \
        # JUCE GUI dependencies (for Linux)
        libfreetype6-dev \
        libx11-dev \
        libxrandr-dev \
        libxcursor-dev \
        libxinerama-dev \
        libgl1-mesa-dev \
        libxext-dev \
        && rm -rf /var/lib/apt/lists/*

# Install Node.js (LTS) for iDAWi frontend development
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install common Python packages (as root, before switching users)
RUN pip install --upgrade pip setuptools wheel

# Switch to non-root user
USER ubuntu

# Set working directory to user's home (as per cloud agent best practices)
WORKDIR /home/ubuntu

# Add user's local bin to PATH (for pip --user installations)
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Default command - launch bash shell
CMD ["/bin/bash"]
