"""Provider-agnostic LLM factory.

The rest of the system never imports a vendor SDK directly. Agents request a
chat model through :func:`get_chat_model`, which returns a LangChain
``BaseChatModel`` so it plugs straight into LangGraph nodes. Swapping OpenAI for
Groq (or adding a new provider) is a one-place change.

Vendor packages are imported lazily so that importing this module — or the
FastAPI app — does not pull the entire LangChain stack until a model is actually
constructed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from medical_research_agent.config import EmbeddingProvider, Provider, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


class LLMConfigurationError(RuntimeError):
    """Raised when a provider is requested without the required credentials."""


def get_chat_model(
    provider: Provider | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Build a configured chat model for the requested provider.

    Args:
        provider: ``"openai"``, ``"groq"``, or ``"gemini"``. Defaults to the
            configured ``DEFAULT_LLM_PROVIDER``.
        model: Override the provider's default model name.
        temperature: Override the configured sampling temperature.

    Raises:
        LLMConfigurationError: If the provider's API key is not configured.
        ValueError: If the provider name is unknown.
    """
    settings = get_settings()
    provider = provider or settings.default_llm_provider
    model = model or settings.model_for(provider)
    temperature = settings.llm_temperature if temperature is None else temperature

    api_key = settings.api_key_for(provider)
    if not api_key:
        key_names: dict[Provider, str] = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        raise LLMConfigurationError(
            f"No API key configured for provider '{provider}'. "
            f"Set {key_names[provider]} in your .env."
        )

    common = {
        "model": model,
        "temperature": temperature,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=api_key, **common)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(api_key=api_key, **common)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(google_api_key=api_key, **common)

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_embeddings_model(provider: EmbeddingProvider | None = None) -> Embeddings:
    """Build a configured embeddings model for the requested provider.

    Args:
        provider: ``"openai"`` or ``"gemini"`` (Groq has no embeddings API).
            Defaults to the configured ``EMBEDDING_PROVIDER``.

    Raises:
        LLMConfigurationError: If the provider's API key is not configured.
        ValueError: If the provider name is unknown.
    """
    settings = get_settings()
    provider = provider or settings.embedding_provider
    model = settings.embedding_model_for(provider)

    api_key = settings.api_key_for(provider)
    if not api_key:
        key_names: dict[EmbeddingProvider, str] = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        raise LLMConfigurationError(
            f"No API key configured for embedding provider '{provider}'. "
            f"Set {key_names[provider]} in your .env."
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(openai_api_key=api_key, model=model)
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model)

    raise ValueError(f"Unknown embedding provider: {provider!r}")
