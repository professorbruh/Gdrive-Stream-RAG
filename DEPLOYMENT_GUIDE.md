# 🚀 Oracle Cloud (Free Tier) Deployment Guide

This guide covers the complete setup required to host the DriveStream RAG-MCP web application on **Oracle Cloud Infrastructure (OCI)** with **HTTPS (Valid SSL via Caddy)**, **Automated CI/CD (GitHub Actions)**, and **Split GPU Architecture**.

Oracle Cloud's "Always Free" tier provides an incredible **Ampere A1 Compute ARM instance** with up to 4 Cores and 24GB of RAM, making it the absolute best platform for hosting the non-GPU parts of this architecture for free!

---

## Step 1: Create SSH Keys for GitHub Actions
We need a dedicated SSH key for GitHub Actions to securely connect to your new Oracle VM.
1. On your local PC, open a terminal and generate a new key pair:
   ```bash
   ssh-keygen -t ed25519 -C "github-actions-deploy" -f ./oci_deploy_key
   ```
   *(Leave the passphrase empty by pressing Enter twice).*
2. This creates two files:
   - `oci_deploy_key` (The Private Key — Keep this secret! We will put this in GitHub).
   - `oci_deploy_key.pub` (The Public Key — We will give this to Oracle Cloud).

---

## Step 2: Create the Oracle Cloud VM
1. Go to your [Oracle Cloud Console](https://cloud.oracle.com).
2. Click **Create a VM instance**.
   - **Name**: `rag-mcp-server`
   - **Placement**: Keep the default Domain.
   - **Image and shape**: 
     - Click Edit. Change the Image to **Ubuntu 22.04 or 24.04**.
     - Change the Shape to **Ampere -> VM.Standard.A1.Flex** (Check the "Always Free Eligible" tag). Set OCPUs to **2** and Memory to **12GB** (or max it out at 4/24GB if you want!).
   - **Networking**: Leave defaults to create a new VCN. **CRITICAL**: Ensure the Subnet is set to **Public Subnet** and check the box that says **Assign a public IPv4 address**.
   - **Add SSH keys**: Select **Paste public keys**. Open your `oci_deploy_key.pub` file and paste the entire contents into the box.
   - **Boot volume**: Leave default (50GB).
3. Click **Create**. Note down the **Public IP Address** and the default **Username** (`ubuntu`) once the VM is running.

---

## Step 3: Configure Oracle VCN Firewall (Ingress Rules)
Oracle Cloud blocks all ports by default at the network level.
1. On your VM's details page, click on the **Subnet** link under Primary VNIC.
2. Click on the **Security List** (usually named `Default Security List for...`).
3. Click **Add Ingress Rules**.
   - **Source CIDR**: `0.0.0.0/0`
   - **IP Protocol**: `TCP`
   - **Destination Port Range**: `80,443`
4. Click **Add Ingress Rules**.

---

## Step 4: Open the Machine Firewall (iptables)
Unlike GCP, Oracle Cloud Ubuntu images have strict `iptables` rules enabled inside the OS itself that block HTTP/HTTPS, even if the VCN allows it.
1. SSH into your instance from your local PC using the key you created:
   ```bash
   ssh -i ./oci_deploy_key ubuntu@YOUR_ORACLE_IP
   ```
2. Once inside, run these exact commands to open the ports and save the rules:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo netfilter-persistent save
   ```

---

## Step 5: Register your Free Domain (DuckDNS)
1. Go to [duckdns.org](https://www.duckdns.org) and log in.
2. In the "domains" box, type a name (e.g., `drivestream-rag`) and click **add domain**.
3. In the domain list, find the **current ip** column.
4. Replace it with the **Public IP Address** of your Oracle VM and click **update ip**.

---

## Step 6: Configure GitHub Secrets
1. Go to your repository on GitHub.
2. Click **Settings** -> **Secrets and variables** -> **Actions**.
3. Add the following four **New repository secrets**:
   - `APP_DOMAIN`: Your exact DuckDNS domain (e.g. `drivestream-rag.duckdns.org`).
   - `OCI_HOST_IP`: The public IP address of your Oracle VM.
   - `OCI_USERNAME`: `ubuntu`
   - `OCI_SSH_KEY`: Open the `oci_deploy_key` (the private key, no `.pub`) in a text editor and paste the ENTIRE contents here (including the `BEGIN` and `END` lines).

---

## Step 7: Initial VM Bootstrap (One-time setup)
You need to clone the repo on the VM so the deploy script knows where it is.
1. SSH back into your VM (if you closed it).
2. Run these commands:
   ```bash
   sudo apt update
   sudo apt install git python3-venv -y
   
   # Clone the repo exactly into ~/Gdrive-Stream-RAG
   git clone https://github.com/YOUR_USERNAME/Gdrive-Stream-RAG.git
   cd Gdrive-Stream-RAG
   
   # Create your .env file
   cp .env.example .env
   nano .env
   ```
3. Inside the `.env` editor, configure it for remote GPU execution:
   ```env
   LLM_MODE=remote
   LLM_REMOTE_URL=https://your-ngrok-url.ngrok-free.app/generate
   LLM_API_KEY=your_super_secret_password_here
   ```
   *(Save and exit by pressing `Ctrl+X`, then `Y`, then `Enter`).*

---

## Step 8: Trigger the Deployment!
Now everything is wired up.
1. Make a small change to any file locally (or just run `git commit --allow-empty -m "Trigger deployment"`).
2. Run `git push origin master`.
3. Go to the **Actions** tab on your GitHub repository.
4. Watch the pipeline run! It will SSH into your VM, install Caddy, pull the code, install python dependencies, and restart the server.

Once it's green, go to `https://your-domain.duckdns.org` in your browser. You will see a beautiful padlock and your RAG app!

---

## Step 9: Start your Local GPU (The Brains)
Because you set `LLM_MODE=remote` on the cloud, the Oracle Cloud server expects your local PC to generate the answers securely.
1. On your local GPU PC, ensure you have the heavy GPU dependencies installed:
   ```bash
   pip install -r requirements-gpu.txt
   ```
2. Set the exact same API key in your environment variables:
   - On Windows (PowerShell): `$env:LLM_API_KEY="your_super_secret_password_here"`
2. Start the Local GPU server:
   ```bash
   python run.py llm-server
   ```
3. Expose the server to the internet using Ngrok:
   ```bash
   ngrok http 8080
   ```
4. Take the HTTPS URL Ngrok gives you, SSH into your Oracle Cloud VM, and update the `LLM_REMOTE_URL` in your `.env` file to match it (then restart the service: `sudo systemctl restart rag-mcp`).

**Congratulations! You now have a fully automated, HTTPS-secured Oracle Cloud Web App powered securely by your Local GPU!**
