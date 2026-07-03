"""
CLI entrypoint for the ingestion pipeline.
Parses the DriveStream codebase → embeds chunks → stores in ChromaDB.

Usage:
    python -m ingestion.ingest
"""

import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors with Rich
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure rag-mcp is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

import config
from ingestion.code_parser import parse_codebase, CodeChunk
from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore

console = Console()


def run_ingestion():
    """Runs the full ingestion pipeline."""
    console.print("\n[bold cyan]╔══════════════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║    DriveStream RAG — Ingestion Pipeline          ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════╝[/]\n")

    # ── Step 1: Parse source files ────────────────────────────
    console.print("[bold yellow]Step 1:[/] Parsing codebase...")
    t0 = time.time()

    chunks = parse_codebase(config.SOURCE_DIR, config.PROJECT_ROOT)
    parse_time = time.time() - t0

    console.print(f"  [green]✓[/] Parsed [bold]{len(chunks)}[/] chunks in {parse_time:.2f}s\n")

    # Show summary table
    table = Table(title="Chunk Summary", show_lines=False)
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Example ID", style="dim")

    type_counts: dict[str, list[CodeChunk]] = {}
    for chunk in chunks:
        type_counts.setdefault(chunk.chunk_type, []).append(chunk)

    for chunk_type, items in type_counts.items():
        table.add_row(chunk_type, str(len(items)), items[0].id)

    console.print(table)
    console.print()

    # ── Step 2: Generate embeddings ───────────────────────────
    console.print("[bold yellow]Step 2:[/] Generating embeddings...")
    t1 = time.time()

    embedder = Embedder()
    texts = [chunk.to_embedding_text() for chunk in chunks]
    embeddings = embedder.embed_texts(texts)
    embed_time = time.time() - t1

    console.print(f"  [green]✓[/] Generated {len(embeddings)} embeddings in {embed_time:.2f}s\n")

    # ── Step 3: Store in ChromaDB ─────────────────────────────
    console.print("[bold yellow]Step 3:[/] Storing in ChromaDB...")
    t2 = time.time()

    store = VectorStore()
    store.clear()  # Fresh start

    # De-duplicate IDs (Java method overloads create duplicates like send, send, send)
    ids = []
    id_counts: dict[str, int] = {}
    for chunk in chunks:
        base_id = chunk.id
        id_counts[base_id] = id_counts.get(base_id, 0) + 1
        if id_counts[base_id] > 1:
            ids.append(f"{base_id}_{id_counts[base_id]}")
        else:
            ids.append(base_id)

    documents = [chunk.to_embedding_text() for chunk in chunks]
    metadatas = [chunk.to_dict() for chunk in chunks]

    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    store_time = time.time() - t2

    console.print(f"  [green]✓[/] Stored {len(ids)} vectors in ChromaDB in {store_time:.2f}s")
    console.print(f"  [dim]  Path: {config.CHROMA_DB_DIR}[/]\n")

    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - t0
    console.print(f"[bold green]✅ Ingestion complete![/] Total time: {total_time:.2f}s")
    console.print(f"   Chunks: {len(chunks)} | Vectors: {len(embeddings)} | DB: {config.CHROMA_COLLECTION_NAME}\n")


if __name__ == "__main__":
    run_ingestion()
