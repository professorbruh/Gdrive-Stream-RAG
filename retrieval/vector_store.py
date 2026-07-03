"""
ChromaDB vector store interface.
Handles persistent storage of code embeddings and similarity search.
"""

import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


class VectorStore:
    """Persistent ChromaDB vector store for code embeddings."""

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        persist_dir = persist_dir or str(CHROMA_DB_DIR)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """Adds vectors to the collection. Handles ChromaDB batch limits."""
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict = None,
    ) -> dict:
        """
        Queries the collection by embedding similarity.

        Returns:
            dict with keys: ids, documents, metadatas, distances
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def count(self) -> int:
        """Returns the number of vectors in the collection."""
        return self.collection.count()

    def clear(self):
        """Deletes and recreates the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
