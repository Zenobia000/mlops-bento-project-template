#!/bin/bash

# MLOps Template Poetry Setup Script
echo "🚀 Setting up MLOps Template with Poetry..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Configure Poetry
echo "⚙️ Configuring Poetry..."
poetry config virtualenvs.create true
poetry config virtualenvs.in-project false
poetry config virtualenvs.path ~/.cache/pypoetry/virtualenvs

# Prepare project structure
echo "📁 Preparing project structure..."
mkdir -p domain
touch domain/__init__.py
echo "✅ Project structure prepared"

# Install dependencies
echo "📚 Installing project dependencies..."
if [ -f "pyproject.toml" ]; then
    # Install all dependencies including optional extras
    poetry install --with dev --extras all

    echo "🎯 Installing GPU-specific packages..."
    # Install PyTorch with CUDA support (separate due to platform requirements)
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install TensorFlow GPU support
    poetry run pip install tensorflow[and-cuda]

    # Install OpenAI Whisper
    poetry run pip install git+https://github.com/openai/whisper.git

    echo "✅ Dependencies installed successfully!"
else
    echo "❌ pyproject.toml not found. Please ensure you're in the project root directory."
    exit 1
fi

# Verify installations
echo "🔍 Verifying installations..."
poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
poetry run python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"
poetry run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    poetry run pre-commit install
fi

# Display helpful information
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📝 Useful Poetry commands:"
echo "  poetry shell                 # Activate virtual environment"
echo "  poetry install               # Install dependencies"
echo "  poetry add <package>         # Add new dependency"
echo "  poetry run <command>         # Run command in virtual environment"
echo "  poetry show                  # Show installed packages"
echo ""
echo "🧪 Test your setup:"
echo "  poetry run python shared/utils/gpu_utils/verify_cuda_pytorch.py"
echo "  poetry run python domain/models/generative/llm/zero_shot_classification.py classify"
echo ""
echo "🚀 Start developing:"
echo "  poetry shell"
echo "  cd domain/models/traditional/sklearn"
echo "  jupyter lab"