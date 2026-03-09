#!/bin/bash

# Script de démarrage pour Projet-QMKL-Finance

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=================================================="
echo "  Quantum Multiple Kernel Learning - Startup"
echo "=================================================="
echo ""

# Check venv
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found."
    echo "   Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
echo "✓ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check ipykernel is registered
echo "✓ Registering Jupyter kernel..."
python -m ipykernel install --user --name qmkl-finance --display-name "QMKL-Finance" 2>/dev/null || true

echo ""
echo "=================================================="
echo "  Ready to use!"
echo "=================================================="
echo ""
echo "Available commands:"
echo ""
echo "  1. Launch Jupyter (notebooks):"
echo "     jupyter notebook"
echo ""
echo "  2. Run experiment:"
echo "     python scripts/run_experiment.py --config config/default.yaml"
echo ""
echo "  3. Run tests:"
echo "     pytest tests/ -v"
echo ""
echo "  4. List available kernels:"
echo "     jupyter kernelspec list"
echo ""
echo "=================================================="
echo ""
