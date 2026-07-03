"""
FastAPI web gateway for the DriveStream RAG system.

Serves:
  - POST /api/ask         — Ask a question (full RAG)
  - POST /api/search      — Search code snippets
  - POST /api/explain     — Explain a class
  - GET  /api/health      — Health check
  - GET  /                — Chat UI (serves static files)

Usage:
    python -m web.app
    # or
    uvicorn web.app:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Ensure rag-mcp is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config

# ── Pydantic Models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class ExplainRequest(BaseModel):
    class_name: str = Field(..., min_length=1, max_length=100)


# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="DriveStream RAG API",
    description="Ask questions about the DriveStream codebase using RAG",
    version="1.0.0",
)

# Lazy-loaded RAG engine
_rag_engine = None

def _get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        from retrieval.rag_engine import RAGEngine
        _rag_engine = RAGEngine()
    return _rag_engine


# ── API Endpoints ────────────────────────────────────────────────────

@app.post("/api/ask")
async def ask(request: AskRequest):
    """Ask a question about the DriveStream codebase."""
    try:
        engine = _get_rag_engine()
        response = engine.ask(request.question, top_k=request.top_k)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "chunks_used": response.chunks_used,
            "model": response.model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search(request: SearchRequest):
    """Search for relevant code snippets."""
    try:
        engine = _get_rag_engine()
        results = engine.search_code(request.query, top_k=request.top_k)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain")
async def explain(request: ExplainRequest):
    """Get a detailed explanation of a specific class."""
    try:
        engine = _get_rag_engine()
        response = engine.explain_class(request.class_name)
        return {
            "explanation": response.answer,
            "sources": response.sources,
            "chunks_used": response.chunks_used,
            "model": response.model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    engine = _get_rag_engine()
    return {
        "status": "ok",
        "vectors": engine.vector_store.count(),
        "model": engine.llm.model_name,
    }


# ── Static Files (Chat UI) ──────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_ui():
    """Serves the chat UI."""
    return FileResponse(str(static_dir / "index.html"))


# ── CLI Launcher ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"\n  🚀 Starting DriveStream RAG Web UI on http://localhost:{config.WEB_PORT}\n")
    uvicorn.run(
        "web.app:app",
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        reload=False,
    )
