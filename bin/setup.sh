#!/bin/bash
# Setup script for CS336 Assignment 5 environment

set -e  # Exit on error

echo "Installing system packages..."
apt update && apt install -y rsync vim

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Use copy mode since cache and venv may be on different filesystems (common on cloud pods)
export UV_LINK_MODE=copy

echo "Running uv sync..."
uv sync

echo "Setup complete!"
