"""Tests for the provider-agnostic LLM factory — construction only, no live calls."""

from __future__ import annotations

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from medical_research_agent.config import Settings
from medical_research_agent.llm.factory import LLMConfigurationError, get_chat_model


def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr("medical_research_agent.llm.factory.get_settings", lambda: settings)


def test_get_chat_model_openai_constructs_chatopenai(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, Settings(default_llm_provider="openai", openai_api_key="fake-key"))

    model = get_chat_model("openai")

    assert isinstance(model, ChatOpenAI)


def test_get_chat_model_groq_constructs_chatgroq(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, Settings(default_llm_provider="groq", groq_api_key="fake-key"))

    model = get_chat_model("groq")

    assert isinstance(model, ChatGroq)


def test_get_chat_model_gemini_constructs_chatgooglegenerativeai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, Settings(default_llm_provider="gemini", gemini_api_key="fake-key"))

    model = get_chat_model("gemini")

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-2.0-flash")


def test_get_chat_model_respects_temperature_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(
        monkeypatch,
        Settings(
            default_llm_provider="gemini",
            gemini_api_key="fake-key",
            llm_temperature=0.7,
            llm_max_retries=5,
        ),
    )

    model = get_chat_model("gemini")

    assert model.temperature == 0.7
    assert model.max_retries == 5


def test_get_chat_model_missing_gemini_key_raises_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, Settings(default_llm_provider="gemini", gemini_api_key=None))

    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        get_chat_model("gemini")


def test_get_chat_model_missing_openai_key_raises_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, Settings(default_llm_provider="openai", openai_api_key=None))

    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        get_chat_model("openai")
