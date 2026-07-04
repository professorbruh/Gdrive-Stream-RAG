"""Tests for retrieval/vector_store.py — ChromaDB operations using ephemeral storage."""

import pytest
from retrieval.vector_store import VectorStore


@pytest.fixture
def vector_store(temp_chroma_dir):
    """Creates a VectorStore backed by a temporary directory."""
    return VectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="test_collection",
    )


@pytest.fixture
def sample_embeddings():
    """Returns sample 384-dim embeddings (matching all-MiniLM-L6-v2)."""
    import random
    random.seed(42)
    return [[random.uniform(-1, 1) for _ in range(384)] for _ in range(5)]


class TestVectorStoreBasics:
    """Basic CRUD operations on the vector store."""

    def test_starts_empty(self, vector_store):
        assert vector_store.count() == 0

    def test_add_and_count(self, vector_store, sample_embeddings):
        vector_store.add(
            ids=["id_1", "id_2", "id_3"],
            embeddings=sample_embeddings[:3],
            documents=["doc one", "doc two", "doc three"],
            metadatas=[
                {"chunk_type": "class", "class_name": "Foo"},
                {"chunk_type": "method", "class_name": "Bar"},
                {"chunk_type": "class", "class_name": "Baz"},
            ],
        )
        assert vector_store.count() == 3

    def test_clear_resets_count(self, vector_store, sample_embeddings):
        vector_store.add(
            ids=["id_1"],
            embeddings=[sample_embeddings[0]],
            documents=["doc one"],
            metadatas=[{"chunk_type": "class"}],
        )
        assert vector_store.count() == 1
        vector_store.clear()
        assert vector_store.count() == 0


class TestVectorStoreQuery:
    """Query operations."""

    def test_query_returns_correct_structure(self, vector_store, sample_embeddings):
        vector_store.add(
            ids=["id_1", "id_2"],
            embeddings=sample_embeddings[:2],
            documents=["class EventStreamEngine", "class TopicManager"],
            metadatas=[
                {"chunk_type": "class", "class_name": "EventStreamEngine"},
                {"chunk_type": "class", "class_name": "TopicManager"},
            ],
        )
        results = vector_store.query(
            query_embedding=sample_embeddings[0],
            top_k=2,
        )
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results

    def test_query_returns_requested_count(self, vector_store, sample_embeddings):
        vector_store.add(
            ids=[f"id_{i}" for i in range(5)],
            embeddings=sample_embeddings,
            documents=[f"doc {i}" for i in range(5)],
            metadatas=[{"chunk_type": "class"} for _ in range(5)],
        )
        results = vector_store.query(
            query_embedding=sample_embeddings[0],
            top_k=3,
        )
        assert len(results["ids"]) == 3

    def test_query_most_similar_first(self, vector_store, sample_embeddings):
        """The query embedding itself should be the closest match."""
        vector_store.add(
            ids=["exact_match", "other"],
            embeddings=[sample_embeddings[0], sample_embeddings[4]],
            documents=["exact match doc", "other doc"],
            metadatas=[{"chunk_type": "class"}, {"chunk_type": "method"}],
        )
        results = vector_store.query(
            query_embedding=sample_embeddings[0],
            top_k=2,
        )
        assert results["ids"][0] == "exact_match"

    def test_query_with_metadata_filter(self, vector_store, sample_embeddings):
        vector_store.add(
            ids=["class_chunk", "method_chunk"],
            embeddings=sample_embeddings[:2],
            documents=["a class", "a method"],
            metadatas=[
                {"chunk_type": "class", "class_name": "Foo"},
                {"chunk_type": "method", "class_name": "Bar"},
            ],
        )
        results = vector_store.query(
            query_embedding=sample_embeddings[0],
            top_k=5,
            where={"chunk_type": "method"},
        )
        assert len(results["ids"]) == 1
        assert results["metadatas"][0]["chunk_type"] == "method"

    def test_query_empty_collection(self, vector_store, sample_embeddings):
        results = vector_store.query(
            query_embedding=sample_embeddings[0],
            top_k=5,
        )
        assert results["ids"] == []
        assert results["documents"] == []


class TestVectorStoreBatching:
    """Test batch insert handling."""

    def test_large_batch_insert(self, vector_store):
        """Inserts more than the batch_size (5000) to test chunking."""
        import random
        random.seed(99)
        n = 100  # Keep small for CI speed, but tests the add() loop
        ids = [f"id_{i}" for i in range(n)]
        embeddings = [[random.uniform(-1, 1) for _ in range(384)] for _ in range(n)]
        documents = [f"document {i}" for i in range(n)]
        metadatas = [{"chunk_type": "class"} for _ in range(n)]

        vector_store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        assert vector_store.count() == n
