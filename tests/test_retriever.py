"""Tests for retrieval/retriever.py — RetrievedChunk and search operations."""

import pytest
from unittest.mock import MagicMock
from retrieval.retriever import Retriever, RetrievedChunk


class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass properties."""

    def test_file_path_property(self):
        chunk = RetrievedChunk(
            id="test",
            content="code",
            metadata={"file_path": "src/Foo.java"},
            score=0.1,
        )
        assert chunk.file_path == "src/Foo.java"

    def test_class_name_property(self):
        chunk = RetrievedChunk(
            id="test",
            content="code",
            metadata={"class_name": "EventStreamEngine"},
            score=0.1,
        )
        assert chunk.class_name == "EventStreamEngine"

    def test_method_name_property(self):
        chunk = RetrievedChunk(
            id="test",
            content="code",
            metadata={"method_name": "createTopic"},
            score=0.1,
        )
        assert chunk.method_name == "createTopic"

    def test_docstring_property(self):
        chunk = RetrievedChunk(
            id="test",
            content="code",
            metadata={"docstring": "Creates a topic"},
            score=0.1,
        )
        assert chunk.docstring == "Creates a topic"

    def test_missing_metadata_returns_empty_string(self):
        chunk = RetrievedChunk(
            id="test",
            content="code",
            metadata={},
            score=0.1,
        )
        assert chunk.file_path == ""
        assert chunk.class_name == ""
        assert chunk.method_name == ""
        assert chunk.docstring == ""


class TestRetriever:
    """Tests for Retriever search operations using mocked dependencies."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.1] * 384
        return embedder

    @pytest.fixture
    def mock_vector_store(self):
        store = MagicMock()
        store.query.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "documents": ["class Foo {}", "void bar() {}"],
            "metadatas": [
                {"file_path": "Foo.java", "chunk_type": "class", "class_name": "Foo", "method_name": "", "docstring": ""},
                {"file_path": "Bar.java", "chunk_type": "method", "class_name": "Bar", "method_name": "bar", "docstring": ""},
            ],
            "distances": [0.1, 0.3],
        }
        return store

    def test_search_returns_retrieved_chunks(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        results = retriever.search("How does Foo work?")
        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_search_embeds_query(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        retriever.search("test query")
        mock_embedder.embed_query.assert_called_once_with("test query")

    def test_search_queries_store_with_embedding(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        retriever.search("test query", top_k=3)
        mock_vector_store.query.assert_called_once()
        call_kwargs = mock_vector_store.query.call_args
        assert call_kwargs.kwargs["top_k"] == 3

    def test_search_preserves_metadata(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        results = retriever.search("test")
        assert results[0].file_path == "Foo.java"
        assert results[0].class_name == "Foo"
        assert results[1].method_name == "bar"

    def test_search_preserves_scores(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        results = retriever.search("test")
        assert results[0].score == 0.1
        assert results[1].score == 0.3

    def test_search_by_class_passes_filter(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        retriever.search_by_class("EventStreamEngine")
        call_kwargs = mock_vector_store.query.call_args
        assert call_kwargs.kwargs["where"] == {"class_name": "EventStreamEngine"}

    def test_search_empty_results(self, mock_embedder):
        empty_store = MagicMock()
        empty_store.query.return_value = {
            "ids": [], "documents": [], "metadatas": [], "distances": [],
        }
        retriever = Retriever(embedder=mock_embedder, vector_store=empty_store)
        results = retriever.search("nothing here")
        assert results == []
