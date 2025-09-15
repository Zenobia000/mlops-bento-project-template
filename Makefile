.PHONY: install install-pip test format lint checkgpu bento-build containerize run deploy clean help all

# Poetry-based installation (recommended)
install:
	@echo "🚀 Installing with Poetry..."
	poetry install --with dev --extras all
	@echo "🎯 Installing GPU-specific packages..."
	poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	poetry run pip install tensorflow[and-cuda]
	poetry run pip install git+https://github.com/openai/whisper.git
	@echo "✅ Installation completed!"

# Development installation without Poetry (not recommended)
install-dev:
	@echo "📦 Development installation..."
	@echo "⚠️  Poetry is the recommended way to install dependencies"
	@echo "⚠️  Installing basic development tools only..."
	pip install --upgrade pip
	pip install black pylint pytest jupyter
	@echo "✅ Basic dev tools installed! Use 'poetry install' for full setup"

# Run tests
test:
	@echo "🧪 Running tests..."
	poetry run python -m pytest -vvs --cov=shared --cov=domain --cov=application tests/

# Format code with black
format:
	@echo "🎨 Formatting code with black..."
	poetry run black shared/ domain/ application/ tests/ scripts/

# Lint code with pylint
lint:
	@echo "🔍 Linting code with pylint..."
	poetry run pylint --disable=R,C shared/ domain/ application/

# Check GPU setup
checkgpu:
	@echo "🎯 Checking GPU setup..."
	poetry run python shared/utils/gpu_utils/verify_cuda_pytorch.py
	poetry run python shared/utils/gpu_utils/verify_tf.py

# Train model
train:
	@echo "🏃 Training model..."
	poetry run python application/training/pipelines/train.py

# Build BentoML service
bento-build:
	@echo "📦 Building BentoML service..."
	cd application/inference/services && poetry run bentoml build

# Containerize BentoML service
containerize:
	@echo "🐳 Containerizing BentoML service..."
	poetry run bentoml containerize iris_classifier:latest

# Run BentoML service locally
run:
	@echo "🚀 Starting BentoML service..."
	cd application/inference/services && poetry run bentoml serve service.py:IrisClassifier --reload

# Deploy (placeholder)
deploy:
	@echo "🚀 Deployment target is a placeholder."
	@echo "Configure your deployment target here."

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ .coverage

# Quick development setup
refactor: format lint

# Show help
help:
	@echo "📚 MLOps Template Makefile Commands:"
	@echo ""
	@echo "🔧 Setup & Installation:"
	@echo "  make install       - Install dependencies with Poetry (recommended)"
	@echo "  make install-dev   - Install basic dev tools with pip (minimal setup)"
	@echo ""
	@echo "🧪 Development:"
	@echo "  make test         - Run tests with pytest"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with pylint"
	@echo "  make refactor     - Format and lint code"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "🎯 GPU & ML:"
	@echo "  make checkgpu     - Check GPU setup (PyTorch & TensorFlow)"
	@echo "  make train        - Train ML model"
	@echo ""
	@echo "🚀 Deployment:"
	@echo "  make bento-build  - Build BentoML service"
	@echo "  make containerize - Create Docker container"
	@echo "  make run          - Run BentoML service locally"
	@echo "  make deploy       - Deploy service (configure target)"
	@echo ""
	@echo "📖 Usage Examples:"
	@echo "  make install && make checkgpu  # Setup and verify"
	@echo "  make refactor && make test     # Code quality check"
	@echo "  make run                       # Start local service"

# Complete pipeline
all: install format lint test checkgpu

# Default target
.DEFAULT_GOAL := help