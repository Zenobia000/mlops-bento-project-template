# CLAUDE.md - Project Configuration

This file contains project-specific information for Claude Code to help with development tasks.

## Project Overview
MLOps Template for GPU-enabled machine learning projects. This template includes PyTorch, TensorFlow, Hugging Face, and various MLOps tools.

## Common Commands

### Setup & Installation
```bash
make install        # Install dependencies and upgrade pip
```

### Testing & Quality
```bash
make test          # Run pytest with coverage
make lint          # Run pylint on Python files
make format        # Format code with black
make refactor      # Format and lint together
```

### GPU Verification
```bash
make checkgpu      # Check GPU availability for PyTorch and TensorFlow
nvidia-smi -l 1    # Monitor GPU usage
python utils/verify_cuda_pytorch.py    # Test PyTorch CUDA
python utils/quickstart_pytorch.py     # PyTorch training test
python utils/quickstart_tf2.py         # TensorFlow training test
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
python hugging-face/zero_shot_classification.py classify

# Keyword extraction
python utils/kw_extract.py

# Whisper transcription
./utils/transcribe-whisper.sh

# Hugging Face fine-tuning
python hugging-face/hf_fine_tune_hello_world.py
```

## Project Structure
- `hugging-face/` - Hugging Face models and utilities
- `utils/` - Utility scripts for GPU verification and ML tasks
- `mylib/` - Custom Python library
- `bentoml/` - BentoML deployment configurations
- `examples/` - Example scripts and notebooks
- `.devcontainer/` - Dev container configuration
- `.github/` - GitHub Actions workflows

## Key Files
- `requirements.txt` - Python dependencies
- `tf-requirements.txt` - TensorFlow-specific requirements
- `Makefile` - Build automation
- `test_main.py` - Test suite
- `main.py` - Main application entry point

## Development Notes
- Uses pip and virtualenv (no conda required)
- GPU support for both PyTorch and TensorFlow
- CI/CD with GitHub Actions
- Docker support with GPU access
- Supports GitHub Codespaces with GPU