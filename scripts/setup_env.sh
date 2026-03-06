#!/usr/bin/env bash
# Setup script for fine-tune-forge
# Creates virtualenv, installs deps, copies .env template

set -e

echo "=== fine-tune-forge setup ==="

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED="3.10"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "✓ Python $PYTHON_VERSION"
else
    echo "✗ Python >= $REQUIRED required (found $PYTHON_VERSION)"
    exit 1
fi

# Create virtualenv if not present
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi

echo "Activating .venv..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing fine-tune-forge..."
pip install --upgrade pip --quiet
pip install -e ".[dev]" --quiet
echo "✓ Dependencies installed"

# Copy env files if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env created from .env.example — add your API keys"
fi

if [ ! -f ".env.ctx" ]; then
    cp .env.ctx.example .env.ctx
    echo "✓ .env.ctx created from .env.ctx.example — configure for ctx tools"
fi

# Create data/output dirs
mkdir -p data/raw data/processed outputs/checkpoints outputs/models logs
echo "✓ data/, outputs/, logs/ directories created"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your GEMINI_API_KEY and HF_TOKEN"
echo "  2. source .venv/bin/activate"
echo "  3. forge list-domains"
echo "  4. FORCE_CPU=true forge run --dominio restaurant_booking --modello Qwen/Qwen3-0.6B"
