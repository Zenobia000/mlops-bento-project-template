# CLAUDE.md - Project Configuration

This file contains project-specific information for Claude Code to help with development tasks.

## Project Overview
MLOps Template for GPU-enabled machine learning projects. This template includes PyTorch, TensorFlow, Hugging Face, and various MLOps tools.

## Common Commands

### Setup & Installation
```bash
make install        # Install dependencies with Poetry (recommended)
make install-dev    # Minimal setup: Install basic dev tools with pip
poetry shell        # Activate Poetry virtual environment
```

### Testing & Quality
```bash
make test          # Run pytest with coverage
make lint          # Run pylint on Python files
make format        # Format code with black
make refactor      # Format and lint together
make clean         # Clean build artifacts
```

### GPU Verification
```bash
make checkgpu      # Check GPU availability for PyTorch and TensorFlow
nvidia-smi -l 1    # Monitor GPU usage
poetry run python shared/utils/gpu_utils/verify_cuda_pytorch.py    # Test PyTorch CUDA
poetry run python domain/models/deep_learning/pytorch/quickstart_pytorch.py     # PyTorch training test
poetry run python domain/models/deep_learning/tensorflow/quickstart_tf2.py         # TensorFlow training test
```

### Docker & Containers
```bash
make container-lint    # Lint Dockerfile with hadolint
# TensorFlow GPU container:
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu
```

### ML Tools
```bash
# Zero-shot classification
poetry run python domain/models/generative/llm/zero_shot_classification.py classify

# Keyword extraction
poetry run python domain/models/specialized/nlp/kw_extract.py

# Whisper transcription
bash domain/models/specialized/nlp/transcribe-whisper.sh

# Hugging Face fine-tuning
poetry run python domain/models/generative/llm/hf_fine_tune_hello_world.py

# BentoML service
make run           # Start BentoML inference service
```

## Project Structure (System Architecture)
- `domain/` - Domain layer (data, models, experiments)
- `application/` - Application layer (training, inference, validation)
- `infrastructure/` - Infrastructure layer (deployment, monitoring, CI/CD)
- `shared/` - Shared utilities and configurations
- `tests/` - Test suite
- `docs/` - Documentation
- `examples/` - Example scripts and industry cases
- `.devcontainer/` - Dev container with Poetry support

## Key Files
- `pyproject.toml` - Poetry configuration and dependencies (replaces requirements.txt)
- `Makefile` - Build automation with Poetry support
- `tests/` - Test suite with pytest
- `.devcontainer/` - VS Code dev container with GPU + Poetry

## Development Notes
- Uses Poetry for dependency management (pip fallback available)
- System architecture with clear layer separation (Domain/Application/Infrastructure)
- GPU support for PyTorch, TensorFlow with CUDA
- Dev container with Poetry and GPU support
- CI/CD with GitHub Actions
- BentoML for model serving and deployment