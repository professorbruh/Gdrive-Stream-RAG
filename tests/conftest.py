"""
Shared pytest fixtures for the DriveStream RAG test suite.

All fixtures are designed to work WITHOUT a GPU, .env file, or live ChromaDB.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class MockRetrievedChunk:
    """Lightweight mock of RetrievedChunk for tests that don't need the real class."""
    id: str = "chunk_001"
    content: str = "public class EventStreamEngine { ... }"
    metadata: dict = None
    score: float = 0.15

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                "file_path": "src/main/java/com/drivestream/EventStreamEngine.java",
                "chunk_type": "class",
                "class_name": "EventStreamEngine",
                "method_name": "",
                "docstring": "Main engine for event streaming",
            }

    @property
    def file_path(self):
        return self.metadata.get("file_path", "")

    @property
    def chunk_type(self):
        return self.metadata.get("chunk_type", "")

    @property
    def class_name(self):
        return self.metadata.get("class_name", "")

    @property
    def method_name(self):
        return self.metadata.get("method_name", "")

    @property
    def docstring(self):
        return self.metadata.get("docstring", "")


@pytest.fixture
def sample_chunks():
    """Returns a list of mock retrieved chunks for testing."""
    return [
        MockRetrievedChunk(
            id="chunk_001",
            content="public class EventStreamEngine {\n    private TopicManager topicManager;\n}",
            metadata={
                "file_path": "src/main/java/com/drivestream/EventStreamEngine.java",
                "chunk_type": "class",
                "class_name": "EventStreamEngine",
                "method_name": "",
                "docstring": "Main engine for event streaming",
            },
            score=0.10,
        ),
        MockRetrievedChunk(
            id="chunk_002",
            content="public void createTopic(String name, int partitions) { ... }",
            metadata={
                "file_path": "src/main/java/com/drivestream/TopicManager.java",
                "chunk_type": "method",
                "class_name": "TopicManager",
                "method_name": "createTopic",
                "docstring": "Creates a new topic with partitions",
            },
            score=0.25,
        ),
        MockRetrievedChunk(
            id="chunk_003",
            content="public void commitOffset(String groupId, int partition, long offset) { ... }",
            metadata={
                "file_path": "src/main/java/com/drivestream/OffsetManager.java",
                "chunk_type": "method",
                "class_name": "OffsetManager",
                "method_name": "commitOffset",
                "docstring": "",
            },
            score=0.35,
        ),
    ]


@pytest.fixture
def mock_llm():
    """Returns a mock LLM that doesn't require GPU or network."""
    llm = MagicMock()
    llm.model_name = "test-model/mock-7b"
    llm.generate.return_value = "This is a mock LLM response about DriveStream."
    llm.ping.return_value = True
    llm.mode = "local"
    return llm


@pytest.fixture
def temp_chroma_dir():
    """Creates a temporary directory for an ephemeral ChromaDB instance."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir
