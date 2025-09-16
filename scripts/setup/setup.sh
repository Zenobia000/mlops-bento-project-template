#!/usr/bin/env bash

# MLOps Template Setup for Codespace
echo "🚀 Setting up MLOps Template for Codespace..."

# Activate virtual environment
if [ -f "/home/codespace/venv/bin/activate" ]; then
    source /home/codespace/venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found at /home/codespace/venv/bin/activate"
    echo "   This might be expected if using Poetry instead"
fi

# Add to bashrc for persistence
echo 'source /home/codespace/venv/bin/activate' >> ~/.bashrc 2>/dev/null || true

# Prepare project structure (for Poetry compatibility)
if [ -f "pyproject.toml" ]; then
    echo "📁 Preparing project structure..."
    mkdir -p domain
    touch domain/__init__.py
    echo "✅ Project structure prepared"
fi

echo "🎉 Codespace setup completed!"
echo "💡 Next steps:"
echo "   - Run: poetry install --with dev --extras all"
echo "   - Run: make install"
echo "   - Run: bash scripts/quickstart.sh"
