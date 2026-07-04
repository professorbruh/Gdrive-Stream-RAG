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
    llm_mode: str = Field(default=None)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)

class ExplainRequest(BaseModel):
    class_name: str = Field(..., min_length=1, max_length=100)
    llm_mode: str = Field(default=None)


# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="DriveStream RAG API",
    description="Ask questions about the DriveStream codebase using RAG",
    version="1.0.0",
)

# ── OpenTelemetry (OCI APM) ──────────────────────────────────────────
if config.OCI_APM_ENDPOINT and config.OCI_APM_DATA_KEY:
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

        # 1. Set the service name
        resource = Resource(attributes={SERVICE_NAME: "drivestream-rag-web"})
        
        # Determine endpoints correctly for Oracle Cloud APM
        base_url = config.OCI_APM_ENDPOINT.split("/20200101/")[0] if "/20200101/" in config.OCI_APM_ENDPOINT else config.OCI_APM_ENDPOINT.rstrip("/")
        
        trace_endpoint = config.OCI_APM_ENDPOINT
        if not trace_endpoint.endswith("/v1/traces") and "dataFormat=otlp" not in trace_endpoint:
            trace_endpoint = f"{base_url}/20200101/opentelemetry/public/v1/traces"
            
        # OCI APM requires a very specific query parameter format for OTLP metrics
        metric_endpoint = f"{base_url}/20200101/observations/metric?dataFormat=otlp-metric&dataFormatVersion=1"

        headers = {"Authorization": f"dataKey {config.OCI_APM_DATA_KEY}"}

        # 2. Configure the Tracer Provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # 3. Configure the OTLP HTTP Trace Exporter
        span_exporter = OTLPSpanExporter(endpoint=trace_endpoint, headers=headers)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        
        # 4. Configure the Meter Provider for Metrics
        metric_exporter = OTLPMetricExporter(endpoint=metric_endpoint, headers=headers)
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        # 5. Auto-instrument FastAPI routes
        FastAPIInstrumentor().instrument_app(app)

        # 6. Gather System Metrics (RAM, CPU)
        SystemMetricsInstrumentor().instrument()
        
        print(f"✓ OpenTelemetry configured for OCI APM (Traces & Metrics)")
    except ImportError as e:
        print(f"⚠️ OpenTelemetry dependencies missing, skipping APM config: {e}")
    except Exception as e:
        print(f"⚠️ OpenTelemetry configuration failed: {e}")

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
        response = engine.ask(request.question, top_k=request.top_k, llm_mode=request.llm_mode)
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
        response = engine.explain_class(request.class_name, llm_mode=request.llm_mode)
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
        "model": engine.default_llm.model_name,
        "mode": engine.default_llm.mode,
        "llm_ready": engine.default_llm.ping(),
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
    print(f"\n  Starting DriveStream RAG Web UI on http://localhost:{config.WEB_PORT}\n")
    uvicorn.run(
        "web.app:app",
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        reload=False,
    )
