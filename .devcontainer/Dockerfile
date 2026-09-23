FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    pkg-config \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    portaudio19-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    sudo \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /workspace
