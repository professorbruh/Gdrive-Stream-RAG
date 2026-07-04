"""Tests for web/app.py — FastAPI API endpoints using TestClient."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_rag_engine(mock_llm):
    """Creates a mock RAG engine for API testing."""
    from retrieval.rag_engine import RAGResponse

    engine = MagicMock()
    engine.vector_store = MagicMock()
    engine.vector_store.count.return_value = 42
    engine.default_llm = mock_llm
    engine.default_llm.model_name = "test-model/mock-7b"
    engine.default_llm.mode = "hf_api"
    engine.default_llm.ping.return_value = True

    # Mock ask()
    engine.ask.return_value = RAGResponse(
        answer="DriveStream uses TopicManager to create topics.",
        sources=[{"file": "TopicManager.java", "class": "TopicManager", "method": "createTopic", "type": "method", "relevance": 0.85}],
        chunks_used=3,
        model_name="test-model/mock-7b",
    )

    # Mock search_code()
    engine.search_code.return_value = [
        {"id": "chunk_1", "file": "TopicManager.java", "class": "TopicManager",
         "method": "createTopic", "type": "method", "content": "public void createTopic...",
         "relevance": 0.85},
    ]

    # Mock explain_class()
    engine.explain_class.return_value = RAGResponse(
        answer="EventStreamEngine is the main orchestrator...",
        sources=[{"file": "EventStreamEngine.java", "class": "EventStreamEngine", "method": "", "type": "class", "relevance": 0.9}],
        chunks_used=5,
        model_name="test-model/mock-7b",
    )

    return engine


@pytest.fixture
def client(mock_rag_engine):
    """Creates a FastAPI TestClient with the RAG engine mocked out."""
    with patch("web.app._get_rag_engine", return_value=mock_rag_engine):
        from web.app import app
        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_returns_correct_shape(self, client):
        resp = client.get("/api/health")
        data = resp.json()
        assert "status" in data
        assert "vectors" in data
        assert "model" in data
        assert "mode" in data
        assert "llm_ready" in data

    def test_health_status_is_ok(self, client):
        resp = client.get("/api/health")
        assert resp.json()["status"] == "ok"

    def test_health_vectors_count(self, client):
        resp = client.get("/api/health")
        assert resp.json()["vectors"] == 42

    def test_health_llm_ready(self, client):
        resp = client.get("/api/health")
        assert resp.json()["llm_ready"] is True


class TestAskEndpoint:
    """Tests for POST /api/ask."""

    def test_ask_returns_200(self, client):
        resp = client.post("/api/ask", json={"question": "How do topics work?"})
        assert resp.status_code == 200

    def test_ask_returns_answer(self, client):
        resp = client.post("/api/ask", json={"question": "How do topics work?"})
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_ask_returns_sources(self, client):
        resp = client.post("/api/ask", json={"question": "How do topics work?"})
        data = resp.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_ask_returns_model_name(self, client):
        resp = client.post("/api/ask", json={"question": "test"})
        data = resp.json()
        assert "model" in data

    def test_ask_returns_chunks_used(self, client):
        resp = client.post("/api/ask", json={"question": "test"})
        data = resp.json()
        assert "chunks_used" in data
        assert data["chunks_used"] == 3

    def test_ask_empty_question_returns_422(self, client):
        resp = client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_ask_missing_question_returns_422(self, client):
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 422

    def test_ask_custom_top_k(self, client, mock_rag_engine):
        client.post("/api/ask", json={"question": "test", "top_k": 3})
        mock_rag_engine.ask.assert_called_once_with("test", top_k=3, llm_mode=None)


class TestSearchEndpoint:
    """Tests for POST /api/search."""

    def test_search_returns_200(self, client):
        resp = client.post("/api/search", json={"query": "partition routing"})
        assert resp.status_code == 200

    def test_search_returns_results(self, client):
        resp = client.post("/api/search", json={"query": "partition routing"})
        data = resp.json()
        assert "results" in data
        assert "count" in data
        assert isinstance(data["results"], list)

    def test_search_empty_query_returns_422(self, client):
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422


class TestExplainEndpoint:
    """Tests for POST /api/explain."""

    def test_explain_returns_200(self, client):
        resp = client.post("/api/explain", json={"class_name": "EventStreamEngine"})
        assert resp.status_code == 200

    def test_explain_returns_explanation(self, client):
        resp = client.post("/api/explain", json={"class_name": "EventStreamEngine"})
        data = resp.json()
        assert "explanation" in data
        assert len(data["explanation"]) > 0

    def test_explain_empty_class_returns_422(self, client):
        resp = client.post("/api/explain", json={"class_name": ""})
        assert resp.status_code == 422


class TestStaticFiles:
    """Tests for static file serving."""

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_root_contains_drivestream(self, client):
        resp = client.get("/")
        assert "DriveStream" in resp.text
