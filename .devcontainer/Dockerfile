# Use Python 3.9 with CUDA support as base image
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry
RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project false \
    && poetry config virtualenvs.path /opt/poetry/venvs

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Install dependencies with Poetry
RUN poetry install --with dev --extras all

# Create codespace user
RUN useradd -m -s /bin/bash codespace \
    && usermod -aG docker codespace

# Switch to codespace user
USER codespace
WORKDIR /home/codespace

# Add Poetry to codespace user PATH
ENV PATH="/home/codespace/.local/bin:$PATH"

# Install Poetry for codespace user
RUN curl -sSL https://install.python-poetry.org | python3 -

# Configure Poetry for codespace user
RUN /home/codespace/.local/bin/poetry config virtualenvs.create true \
    && /home/codespace/.local/bin/poetry config virtualenvs.in-project false \
    && /home/codespace/.local/bin/poetry config virtualenvs.path /home/codespace/.cache/pypoetry/virtualenvs

# Set workspace directory
WORKDIR /workspace