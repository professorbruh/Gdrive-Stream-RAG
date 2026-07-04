"""
Unified CLI launcher for the DriveStream RAG + MCP system.

Usage:
    python run.py ingest     # Ingest the codebase into ChromaDB
    python run.py web        # Start the web chat UI (FastAPI)
    python run.py mcp        # Start the MCP server (stdio)
    python run.py mcp-sse    # Start the MCP server (SSE transport)
    python run.py ask "..."  # Ask a question from the command line
"""

import sys
import os
import argparse
from logger_setup import get_logger

logger = get_logger(__name__)

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure we're working from the rag-mcp directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_usage():
    print("""
  ⚡ DriveStream RAG + MCP System
  ═══════════════════════════════════════════════

  Usage:  python run.py <command> [args]

  Commands:
    ingest          Parse & embed the codebase into ChromaDB
    web             Start the web chat UI (http://localhost:8000)
    mcp             Start the MCP server (stdio transport)
    mcp-sse         Start the MCP server (SSE transport)
    ask "question"  Ask a question from the command line

  Setup:
    1. pip install -r requirements.txt
    2. Copy .env.example to .env and set HF_TOKEN
    3. python run.py ingest
    4. python run.py web   (or)   python run.py mcp
    """)


def cmd_ingest():
    from ingestion.ingest import run_ingestion
    run_ingestion()


def cmd_web():
    import uvicorn
    import config
    logger.info(f"Starting DriveStream RAG Web UI on http://localhost:{config.WEB_PORT}")
    uvicorn.run(
        "web.app:app",
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        reload=False,
    )


def cmd_mcp(args):
    """Starts the MCP Server for external AI integration."""
    from mcp_server.server import main as mcp_main

    if args.transport == "sse":
        os.environ["MCP_TRANSPORT"] = "sse"
    else:
        os.environ["MCP_TRANSPORT"] = "stdio"
    
    mcp_main(args.transport, args.port)


def cmd_llm_server(args):
    """Starts the standalone Local GPU LLM server (for remote mode)."""
    import uvicorn
    from llm.server import app
    logger.info("Starting DriveStream Local GPU Server on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)


def cmd_ask(question: str):
    from retrieval.rag_engine import RAGEngine

    print("\n  ⚡ DriveStream RAG — CLI Query\n")
    print(f"  Question: {question}\n")
    print("  Initializing...")

    engine = RAGEngine()

    print("  Searching & generating answer...\n")
    response = engine.ask(question)

    print("  ═══ ANSWER ═══\n")
    print(f"  {response.answer}\n")

    if response.sources:
        print("  ═══ SOURCES ═══\n")
        for s in response.sources:
            label = f"{s.get('class', '')}.{s.get('method', '')}()" if s.get('method') else s.get('class', s.get('file', ''))
            print(f"    • {label}  ({int(s.get('relevance', 0) * 100)}% match)")
        print()


def main():
    parser = argparse.ArgumentParser(description="DriveStream RAG System")
    subparsers = parser.add_subparsers(dest="command")

    # --- INGEST COMMAND ---
    subparsers.add_parser("ingest", help="Ingest codebase")

    # --- WEB COMMAND ---
    subparsers.add_parser("web", help="Start Web UI")

    # --- MCP COMMAND ---
    parser_mcp = subparsers.add_parser(
        "mcp", help="Start the MCP Server for AI Assistants"
    )
    parser_mcp.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser_mcp.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for SSE transport (default: 8001)",
    )
    
    # --- LLM SERVER COMMAND ---
    subparsers.add_parser(
        "llm-server", help="Start the standalone Local GPU Server for remote execution"
    )

    # --- ASK COMMAND ---
    parser_ask = subparsers.add_parser("ask", help="Ask a question")
    parser_ask.add_argument("question", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest()
    elif args.command == "web":
        cmd_web()
    elif args.command == "mcp":
        cmd_mcp(args)
    elif args.command == "llm-server":
        cmd_llm_server(args)
    elif args.command == "ask":
        if not args.question:
            print("  Error: Please provide a question.")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        cmd_ask(question)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
