#!/bin/bash
# Deployment script executed on the Oracle Cloud instance by GitHub Actions

set -e

APP_DIR="$HOME/Gdrive-Stream-RAG"
echo "Starting deployment for DriveStream RAG-MCP..."

# 1. Ensure directory exists and navigate to it
if [ ! -d "$APP_DIR" ]; then
    echo "Directory $APP_DIR does not exist. Please clone the repository first."
    exit 1
fi
cd "$APP_DIR"

# 2. Pull the latest code from GitHub
echo "Pulling latest code from master branch..."
git pull origin master

# 3. Setup virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# 4. Install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
# Pre-install CPU-only PyTorch so it doesn't download 4GB of NVIDIA CUDA libraries on the cloud
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 5. Setup Caddy (Reverse Proxy) if not installed
if ! command -v caddy &> /dev/null; then
    echo "Installing Caddy..."
    sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
    sudo apt update
    sudo apt install caddy -y
fi

# 6. Apply Caddy configuration
echo "Applying Caddy configuration for domain: $APP_DOMAIN"
sed "s/YOUR_DOMAIN_PLACEHOLDER/$APP_DOMAIN/g" Caddyfile | sudo tee /etc/caddy/Caddyfile > /dev/null
sudo systemctl reload caddy

# 7. Restart the systemd service
echo "Restarting the rag-mcp systemd service..."
sudo systemctl restart rag-mcp

echo "Deployment complete! Service status:"
sudo systemctl status rag-mcp --no-pager
