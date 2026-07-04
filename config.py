"""
Centralized configuration for the DriveStream RAG + MCP system.

All settings can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # d:\kafka-google
SOURCE_DIR = PROJECT_ROOT / "src" / "main" / "java" / "com" / "drivestream"
DOCS_DIR = PROJECT_ROOT  # README.md, SETUP_GUIDE.md live here
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

# ── Embedding Model ─────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIMENSION = 384

# ── LLM ──────────────────────────────────────────────────────────────

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct"
)
# Modes: "local" (GPU), "remote" (Oracle -> Local via Cloudflare), "hf_api" (HuggingFace Cloud)
LLM_MODE = os.getenv("LLM_MODE", "hf_api").lower()
LLM_REMOTE_URL = os.getenv("LLM_REMOTE_URL", "http://localhost:8080/generate")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Quantization: 4-bit for local GPU inference (fits in ~6GB VRAM)
LLM_LOAD_IN_4BIT = os.getenv("LLM_LOAD_IN_4BIT", "false").lower() == "true"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# ── Retrieval ────────────────────────────────────────────────────────

CHROMA_COLLECTION_NAME = "drivestream_code"
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))

# ── Chunking ─────────────────────────────────────────────────────────

CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

# ── Web Server ───────────────────────────────────────────────────────

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))

# ── MCP Server ───────────────────────────────────────────────────────

MCP_SERVER_NAME = "drivestream-rag"
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # "stdio" or "sse"
MCP_SSE_PORT = int(os.getenv("MCP_SSE_PORT", "8001"))

# ── Observability (OpenTelemetry) ────────────────────────────────────

OCI_APM_ENDPOINT = os.getenv("OCI_APM_ENDPOINT", "")
OCI_APM_DATA_KEY = os.getenv("OCI_APM_DATA_KEY", "")
