"""Tests for LLM health check (ping) functionality."""

import pytest
from unittest.mock import MagicMock, patch
import importlib


class TestPingLocal:
    """Tests for ping() when LLM is in local or hf_api mode."""

    def test_ping_returns_true_for_local_mode(self):
        llm = MagicMock()
        llm.mode = "local"
        # Import the real ping method and bind it
        from llm.hf_model import HuggingFaceModel
        assert HuggingFaceModel.ping(llm) is True

    def test_ping_returns_true_for_hf_api_mode(self):
        llm = MagicMock()
        llm.mode = "hf_api"
        from llm.hf_model import HuggingFaceModel
        assert HuggingFaceModel.ping(llm) is True


class TestPingRemote:
    """Tests for ping() when LLM is in remote mode."""

    def test_ping_returns_true_on_successful_response(self):
        llm = MagicMock()
        llm.mode = "remote"

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            from llm.hf_model import HuggingFaceModel
            result = HuggingFaceModel.ping(llm)
            assert result is True
            mock_get.assert_called_once()

    def test_ping_returns_false_on_502(self):
        llm = MagicMock()
        llm.mode = "remote"

        mock_response = MagicMock()
        mock_response.status_code = 502

        with patch("requests.get", return_value=mock_response):
            from llm.hf_model import HuggingFaceModel
            result = HuggingFaceModel.ping(llm)
            assert result is False

    def test_ping_returns_false_on_connection_error(self):
        llm = MagicMock()
        llm.mode = "remote"

        with patch("requests.get", side_effect=ConnectionError("Connection refused")):
            from llm.hf_model import HuggingFaceModel
            result = HuggingFaceModel.ping(llm)
            assert result is False

    def test_ping_returns_false_on_timeout(self):
        llm = MagicMock()
        llm.mode = "remote"

        import requests
        with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
            from llm.hf_model import HuggingFaceModel
            result = HuggingFaceModel.ping(llm)
            assert result is False

    def test_ping_uses_health_endpoint(self):
        """Verifies that ping() calls /health, not /generate."""
        llm = MagicMock()
        llm.mode = "remote"

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch("config.LLM_REMOTE_URL", "https://example.ngrok.dev/generate"):
            from llm.hf_model import HuggingFaceModel
            HuggingFaceModel.ping(llm)
            call_args = mock_get.call_args
            assert "/health" in call_args.args[0]
            assert "/generate" not in call_args.args[0]

    def test_ping_sends_ngrok_header(self):
        """Verifies that the ngrok-skip-browser-warning header is sent."""
        llm = MagicMock()
        llm.mode = "remote"

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            from llm.hf_model import HuggingFaceModel
            HuggingFaceModel.ping(llm)
            call_kwargs = mock_get.call_args.kwargs
            assert "headers" in call_kwargs
            assert "ngrok-skip-browser-warning" in call_kwargs["headers"]
