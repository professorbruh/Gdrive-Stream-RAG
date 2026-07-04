"""Tests for config.py — verifies defaults and env var overrides."""

import os
import importlib
import pytest


class TestConfigDefaults:
    """Verify that config loads sane defaults without any .env file."""

    def test_embedding_model_default(self):
        import config
        assert config.EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_dimension(self):
        import config
        assert config.EMBEDDING_DIMENSION == 384

    def test_llm_model_default(self):
        import config
        assert config.LLM_MODEL_NAME == "Qwen/Qwen2.5-Coder-3B-Instruct"

    def test_llm_mode_default(self):
        import config
        assert config.LLM_MODE in ("local", "remote", "hf_api")

    def test_web_port_default(self):
        import config
        assert config.WEB_PORT == 8000

    def test_retrieval_top_k_default(self):
        import config
        assert config.RETRIEVAL_TOP_K == 6

    def test_chroma_collection_name(self):
        import config
        assert config.CHROMA_COLLECTION_NAME == "drivestream_code"

    def test_chunk_max_tokens_default(self):
        import config
        assert config.CHUNK_MAX_TOKENS == 512

    def test_llm_max_new_tokens_default(self):
        import config
        assert config.LLM_MAX_NEW_TOKENS == 1024

    def test_llm_temperature_default(self):
        import config
        assert config.LLM_TEMPERATURE == 0.3

    def test_base_dir_exists(self):
        import config
        assert config.BASE_DIR.exists()


class TestConfigEnvOverrides:
    """Verify env var overrides are respected when config is reloaded."""

    def test_web_port_override(self, monkeypatch):
        monkeypatch.setenv("WEB_PORT", "9999")
        import config
        importlib.reload(config)
        assert config.WEB_PORT == 9999
        # Reset
        monkeypatch.delenv("WEB_PORT", raising=False)
        importlib.reload(config)

    def test_retrieval_top_k_override(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        import config
        importlib.reload(config)
        assert config.RETRIEVAL_TOP_K == 10
        # Reset
        monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
        importlib.reload(config)

    def test_llm_mode_override(self, monkeypatch):
        monkeypatch.setenv("LLM_MODE", "remote")
        import config
        importlib.reload(config)
        assert config.LLM_MODE == "remote"
        # Reset
        monkeypatch.delenv("LLM_MODE", raising=False)
        importlib.reload(config)
