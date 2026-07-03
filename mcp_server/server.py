"""
MCP (Model Context Protocol) server for the DriveStream RAG system.

Exposes three tools that any MCP-compatible client (Claude Desktop,
Cursor, custom agents) can call:

  1. ask_codebase  — Ask any question about the DriveStream codebase
  2. search_code   — Search for code snippets by semantic similarity
  3. explain_class — Get a detailed explanation of a specific Java class

Supports both stdio and SSE transports.

Usage:
    # stdio (for Claude Desktop / direct MCP clients)
    python -m mcp_server.server

    # SSE (for web-based access)
    MCP_TRANSPORT=sse python -m mcp_server.server
"""

import sys
import json
import asyncio
from pathlib import Path

# Ensure rag-mcp is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import config

# ── Initialize MCP Server ────────────────────────────────────────────

server = Server(config.MCP_SERVER_NAME)

# Lazy-loaded RAG engine (loaded on first tool call)
_rag_engine = None


def _get_rag_engine():
    """Lazily initializes the RAG engine."""
    global _rag_engine
    if _rag_engine is None:
        from retrieval.rag_engine import RAGEngine
        _rag_engine = RAGEngine()
    return _rag_engine


# ── Tool Definitions ─────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Registers the available tools with the MCP client."""
    return [
        Tool(
            name="ask_codebase",
            description=(
                "Ask any question about the DriveStream codebase — a Kafka-inspired "
                "event streaming platform built with Java 21 that uses Google Drive "
                "as storage. The tool retrieves relevant source code and generates "
                "a detailed answer. Examples: 'How does partition routing work?', "
                "'What happens when a consumer commits offsets?', "
                "'How is the Google Drive API authenticated?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question about the DriveStream codebase",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="search_code",
            description=(
                "Search the DriveStream codebase for relevant code snippets by "
                "semantic similarity. Returns matching code chunks with file paths "
                "and relevance scores. Faster than ask_codebase since it doesn't "
                "generate an LLM response."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language or code-like)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="explain_class",
            description=(
                "Get a detailed explanation of a specific Java class in the "
                "DriveStream codebase. Covers purpose, fields, methods, interactions "
                "with other classes, and design patterns. Available classes: "
                "EventStreamEngine, TopicManager, PartitionManager, OffsetManager, "
                "DriveStreamProducer, DriveStreamConsumer, GoogleDriveStorageService, "
                "GoogleDriveConfig, Event, TopicMetadata, ConsumerOffsets, TopicPartition, "
                "DemoApp"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "Name of the Java class to explain",
                    },
                },
                "required": ["class_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatches tool calls to the RAG engine."""
    engine = _get_rag_engine()

    if name == "ask_codebase":
        question = arguments["question"]
        response = engine.ask(question)
        result = {
            "answer": response.answer,
            "sources": response.sources,
            "chunks_used": response.chunks_used,
            "model": response.model_name,
        }
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    elif name == "search_code":
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        results = engine.search_code(query, top_k=top_k)
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2),
        )]

    elif name == "explain_class":
        class_name = arguments["class_name"]
        response = engine.explain_class(class_name)
        result = {
            "explanation": response.answer,
            "sources": response.sources,
            "chunks_used": response.chunks_used,
            "model": response.model_name,
        }
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}),
        )]


# ── Server Launcher ──────────────────────────────────────────────────

async def run_stdio():
    """Runs the MCP server with stdio transport."""
    print(f"Starting MCP server '{config.MCP_SERVER_NAME}' (stdio transport)...", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse():
    """Runs the MCP server with SSE transport."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=sse.handle_post_message, methods=["POST"]),
    ])

    print(f"Starting MCP server '{config.MCP_SERVER_NAME}' (SSE on port {config.MCP_SSE_PORT})...", file=sys.stderr)
    uv_config = uvicorn.Config(app, host="0.0.0.0", port=config.MCP_SSE_PORT)
    uv_server = uvicorn.Server(uv_config)
    await uv_server.serve()


def main():
    transport = config.MCP_TRANSPORT.lower()
    if transport == "sse":
        asyncio.run(run_sse())
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
