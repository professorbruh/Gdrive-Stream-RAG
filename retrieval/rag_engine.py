"""
RAG Engine: orchestrates retrieval → context building → LLM generation.
This is the central interface that the MCP server and web API call.
"""

from dataclasses import dataclass, field

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever, RetrievedChunk
from retrieval.context_builder import (
    build_context,
    build_context_for_class_explanation,
    format_sources_for_response,
)
from llm.hf_model import HuggingFaceModel
from config import RETRIEVAL_TOP_K


@dataclass
class RAGResponse:
    """Response from the RAG engine including answer and source citations."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    chunks_used: int = 0
    model_name: str = ""


class RAGEngine:
    """
    Main RAG orchestrator.

    Usage:
        engine = RAGEngine()
        response = engine.ask("How does DriveStream create topics?")
        print(response.answer)
        print(response.sources)
    """

    def __init__(self, llm: HuggingFaceModel = None):
        print("Initializing RAG Engine...")
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.retriever = Retriever(embedder=self.embedder, vector_store=self.vector_store)
        self.llm = llm or HuggingFaceModel()
        print(f"  ✓ RAG Engine ready — {self.vector_store.count()} vectors in store")

    def ask(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResponse:
        """
        Answers a question about the DriveStream codebase using RAG.

        1. Retrieves top-K relevant code chunks
        2. Builds a context-rich prompt
        3. Generates an answer using the LLM
        4. Returns the answer with source citations
        """
        # Retrieve relevant chunks
        chunks = self.retriever.search(question, top_k=top_k)

        if not chunks:
            return RAGResponse(
                answer="I couldn't find any relevant code in the DriveStream codebase for your question.",
                sources=[],
                chunks_used=0,
                model_name=self.llm.model_name,
            )

        # Build prompt with context
        prompt = build_context(question, chunks)

        # Generate answer
        answer = self.llm.generate(prompt)

        # Format source citations
        sources = format_sources_for_response(chunks)

        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            model_name=self.llm.model_name,
        )

    def search_code(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Searches for relevant code snippets by semantic similarity.
        Returns raw chunks without LLM generation (faster).
        """
        chunks = self.retriever.search(query, top_k=top_k)
        results = []
        for chunk in chunks:
            results.append({
                "id": chunk.id,
                "file": chunk.file_path,
                "class": chunk.class_name,
                "method": chunk.method_name,
                "type": chunk.chunk_type,
                "content": chunk.content[:500],
                "relevance": round(1.0 - chunk.score, 3),
            })
        return results

    def explain_class(self, class_name: str) -> RAGResponse:
        """
        Provides a detailed explanation of a specific class.
        Retrieves all chunks for the class and generates a comprehensive explanation.
        """
        # First try metadata filter, then fall back to semantic search
        chunks = self.retriever.search_by_class(class_name, top_k=10)
        if not chunks:
            chunks = self.retriever.search(
                f"class {class_name} in DriveStream",
                top_k=8,
            )

        if not chunks:
            return RAGResponse(
                answer=f"I couldn't find a class named '{class_name}' in the DriveStream codebase.",
                sources=[],
                chunks_used=0,
                model_name=self.llm.model_name,
            )

        prompt = build_context_for_class_explanation(class_name, chunks)
        answer = self.llm.generate(prompt)
        sources = format_sources_for_response(chunks)

        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            model_name=self.llm.model_name,
        )
