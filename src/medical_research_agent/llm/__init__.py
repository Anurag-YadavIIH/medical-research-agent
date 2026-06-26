"""LLM provider abstraction layer (OpenAI + Groq via a single factory)."""

from medical_research_agent.llm.factory import get_chat_model

__all__ = ["get_chat_model"]
