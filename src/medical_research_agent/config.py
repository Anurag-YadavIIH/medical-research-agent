"""Centralised, typed application configuration.

All configuration is loaded from environment variables (or a local ``.env``)
via ``pydantic-settings``. Nothing else in the codebase should read
``os.environ`` directly — inject :func:`get_settings` instead. This keeps
configuration testable and makes the full surface of required secrets explicit.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "groq", "gemini"]
EmbeddingProvider = Literal["openai", "gemini"]


class Settings(BaseSettings):
    """Strongly-typed runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- App ---------------------------------------------------------------
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # --- LLM ---------------------------------------------------------------
    default_llm_provider: Provider = "groq"
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_default_model: str = "gpt-4o-mini"
    groq_default_model: str = "llama-3.3-70b-versatile"
    gemini_default_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3

    # --- Embeddings (project chat / RAG retrieval) --------------------------
    # Separate from default_llm_provider since Groq has no embeddings API —
    # chat can stay on Groq while embeddings use OpenAI/Gemini.
    embedding_provider: EmbeddingProvider = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"

    # --- NCBI / PubMed -----------------------------------------------------
    ncbi_tool: str = "medical-research-agent"
    ncbi_email: str = "you@example.com"
    ncbi_api_key: str | None = None

    # --- CrossRef ----------------------------------------------------------
    crossref_mailto: str = "you@example.com"

    # --- Persistence -------------------------------------------------------
    database_url: str = "postgresql+asyncpg://mra:mra@localhost:5432/mra"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86_400

    # --- Observability -----------------------------------------------------
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "medical-research-agent"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # --- API ---------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_url: str = "http://localhost:8000"

    def model_for(self, provider: Provider) -> str:
        """Return the configured default model name for a provider."""
        models: dict[Provider, str] = {
            "openai": self.openai_default_model,
            "groq": self.groq_default_model,
            "gemini": self.gemini_default_model,
        }
        return models[provider]

    def api_key_for(self, provider: Provider) -> str | None:
        """Return the configured API key for a provider."""
        keys: dict[Provider, str | None] = {
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
        }
        return keys[provider]

    def embedding_model_for(self, provider: EmbeddingProvider) -> str:
        """Return the configured embedding model name for a provider."""
        models: dict[EmbeddingProvider, str] = {
            "openai": self.openai_embedding_model,
            "gemini": self.gemini_embedding_model,
        }
        return models[provider]


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (one process-wide load)."""
    return Settings()
