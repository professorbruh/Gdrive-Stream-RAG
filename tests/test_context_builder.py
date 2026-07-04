"""Tests for retrieval/context_builder.py — prompt construction and source formatting."""

import pytest
from retrieval.context_builder import (
    build_context,
    build_context_for_class_explanation,
    format_sources_for_response,
    _format_source,
)


class TestBuildContext:
    """Tests for the main prompt builder."""

    def test_returns_string(self, sample_chunks):
        result = build_context("How does DriveStream create topics?", sample_chunks)
        assert isinstance(result, str)

    def test_contains_question(self, sample_chunks):
        question = "How does DriveStream create topics?"
        result = build_context(question, sample_chunks)
        assert question in result

    def test_contains_source_labels(self, sample_chunks):
        result = build_context("test question", sample_chunks)
        assert "Source 1:" in result
        assert "Source 2:" in result
        assert "Source 3:" in result

    def test_contains_chunk_content(self, sample_chunks):
        result = build_context("test question", sample_chunks)
        for chunk in sample_chunks:
            assert chunk.content in result

    def test_contains_system_instructions(self, sample_chunks):
        result = build_context("test question", sample_chunks)
        assert "DriveStream" in result
        assert "YOUR ANSWER" in result

    def test_single_chunk(self, sample_chunks):
        result = build_context("test question", [sample_chunks[0]])
        assert "Source 1:" in result
        assert "Source 2:" not in result


class TestBuildContextForClassExplanation:
    """Tests for class-specific prompt builder."""

    def test_contains_class_name(self, sample_chunks):
        result = build_context_for_class_explanation("EventStreamEngine", sample_chunks)
        assert "EventStreamEngine" in result

    def test_contains_explanation_instructions(self, sample_chunks):
        result = build_context_for_class_explanation("EventStreamEngine", sample_chunks)
        assert "purpose" in result.lower() or "role" in result.lower()
        assert "method" in result.lower()

    def test_contains_source_content(self, sample_chunks):
        result = build_context_for_class_explanation("EventStreamEngine", sample_chunks)
        for chunk in sample_chunks:
            assert chunk.content in result


class TestFormatSourcesForResponse:
    """Tests for source citation formatting."""

    def test_returns_list(self, sample_chunks):
        result = format_sources_for_response(sample_chunks)
        assert isinstance(result, list)

    def test_source_has_required_keys(self, sample_chunks):
        result = format_sources_for_response(sample_chunks)
        for source in result:
            assert "file" in source
            assert "class" in source
            assert "method" in source
            assert "type" in source
            assert "relevance" in source

    def test_relevance_is_float(self, sample_chunks):
        result = format_sources_for_response(sample_chunks)
        for source in result:
            assert isinstance(source["relevance"], float)

    def test_deduplicates_identical_sources(self, sample_chunks):
        # Add a duplicate chunk with same file/class/method
        from tests.conftest import MockRetrievedChunk
        duplicate = MockRetrievedChunk(
            id="chunk_dup",
            content="duplicate content",
            metadata=sample_chunks[0].metadata.copy(),
            score=0.5,
        )
        chunks_with_dup = sample_chunks + [duplicate]
        result = format_sources_for_response(chunks_with_dup)
        # Should be 3 unique sources, not 4
        assert len(result) == 3

    def test_relevance_calculation(self, sample_chunks):
        """Relevance = 1.0 - distance score."""
        result = format_sources_for_response(sample_chunks)
        first_source = result[0]
        expected_relevance = round(1.0 - sample_chunks[0].score, 3)
        assert first_source["relevance"] == expected_relevance


class TestFormatSource:
    """Tests for the internal _format_source helper."""

    def test_includes_filename(self, sample_chunks):
        result = _format_source(sample_chunks[0])
        assert "EventStreamEngine.java" in result

    def test_includes_class_name(self, sample_chunks):
        result = _format_source(sample_chunks[0])
        assert "EventStreamEngine" in result

    def test_includes_method_name(self, sample_chunks):
        result = _format_source(sample_chunks[1])
        assert "createTopic" in result

    def test_includes_docstring_snippet(self, sample_chunks):
        result = _format_source(sample_chunks[0])
        assert "Main engine" in result
