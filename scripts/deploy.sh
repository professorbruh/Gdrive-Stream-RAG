#!/bin/bash
# Deployment script executed on the GCP Compute Engine instance by GitHub Actions

set -e

APP_DIR="$HOME/rag-mcp"
echo "Starting deployment for DriveStream RAG-MCP..."

# 1. Ensure directory exists and navigate to it
if [ ! -d "$APP_DIR" ]; then
    echo "Directory $APP_DIR does not exist. Please clone the repository first."
    exit 1
fi
cd "$APP_DIR"

# 2. Pull the latest code from GitHub
echo "Pulling latest code from main branch..."
git pull origin main

# 3. Setup virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# 4. Install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

# 5. Restart the systemd service
echo "Restarting the rag-mcp systemd service..."
sudo systemctl restart rag-mcp

echo "Deployment complete! Service status:"
sudo systemctl status rag-mcp --no-pager
