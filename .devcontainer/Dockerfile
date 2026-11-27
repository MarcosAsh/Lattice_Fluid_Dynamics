FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    clang-format \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libsdl2-dev \
    mesa-utils \
    xvfb \
    xauth \
    ffmpeg \
    curl \
    ca-certificates \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Node for the website
ARG NODE_MAJOR=26
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# uv for the Python tools
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV MESA_GL_VERSION_OVERRIDE=4.3
ENV LIBGL_ALWAYS_SOFTWARE=1
