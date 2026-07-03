"""
Retriever: embeds a user query and performs top-K similarity search
against the ChromaDB vector store.
"""

from dataclasses import dataclass

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from config import RETRIEVAL_TOP_K


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with its similarity score."""

    id: str
    content: str
    metadata: dict
    score: float  # cosine distance (lower = more similar)

    @property
    def file_path(self) -> str:
        return self.metadata.get("file_path", "")

    @property
    def chunk_type(self) -> str:
        return self.metadata.get("chunk_type", "")

    @property
    def class_name(self) -> str:
        return self.metadata.get("class_name", "")

    @property
    def method_name(self) -> str:
        return self.metadata.get("method_name", "")

    @property
    def docstring(self) -> str:
        return self.metadata.get("docstring", "")


class Retriever:
    """Retrieves relevant code chunks for a given query."""

    def __init__(self, embedder: Embedder = None, vector_store: VectorStore = None):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K, where: dict = None) -> list[RetrievedChunk]:
        """
        Embeds the query and returns the top-K most similar chunks.

        Args:
            query: The user's natural language question.
            top_k: Number of results to return.
            where: Optional metadata filter (e.g., {"chunk_type": "method"}).

        Returns:
            List of RetrievedChunk objects sorted by relevance.
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        chunks = []
        for i in range(len(results["ids"])):
            chunks.append(
                RetrievedChunk(
                    id=results["ids"][i],
                    content=results["documents"][i],
                    metadata=results["metadatas"][i],
                    score=results["distances"][i],
                )
            )

        return chunks

    def search_by_class(self, class_name: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Retrieves all chunks related to a specific class."""
        return self.search(
            query=f"class {class_name}",
            top_k=top_k,
            where={"class_name": class_name},
        )
