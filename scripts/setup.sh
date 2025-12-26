#!/bin/bash
# Setup script for the project
# This script sets up the development environment

set -e

echo "==================================="
echo "Crypto DQN OPS - Setup Script"
echo "==================================="

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Installing dependencies with uv..."
uv sync

echo "Setting up pre-commit hooks..."
uv run pre-commit install

echo "Initializing DVC..."
if [ ! -d ".dvc" ]; then
    uv run dvc init
fi

echo "Creating necessary directories..."
mkdir -p data
mkdir -p trained_models
mkdir -p plots

echo ""
echo "==================================="
echo "Setup completed successfully!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Place your data file in data/crypto_data.pkl"
echo "3. Run: uv run python scripts/setup_data.py"
echo "4. Start training: uv run python commands.py train"
echo ""
echo "For more information, see README.md"
