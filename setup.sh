#!/bin/bash
# Quick setup script for Linux/Mac

echo "Setting up Qwen3-0.6B Local Chat..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To start the chat interface:"
echo "  python src/app.py"
echo ""
echo "To start the API server (optional):"
echo "  python src/api.py"
echo ""
