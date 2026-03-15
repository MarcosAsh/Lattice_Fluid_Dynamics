FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    libgl1-mesa-dev \
    libglew-dev \
    libglu1-mesa-dev \
    libsdl2-dev \
    libsdl2-ttf-dev \
    libosmesa6-dev \
    mesa-utils \
    xvfb \
    x11-utils \
    ffmpeg \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 for the website
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV MESA_GL_VERSION_OVERRIDE=4.3
ENV LIBGL_ALWAYS_SOFTWARE=1
