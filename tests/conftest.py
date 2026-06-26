"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage

from medical_research_agent.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


class FakeStructuredRunnable:
    """Stands in for ``model.with_structured_output(Schema)``.

    ``results`` is consumed one call at a time so a test can return different
    structured outputs (or raise) across successive ``ainvoke`` calls, e.g. once
    per study in a per-item agent loop.
    """

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)

    async def ainvoke(self, messages: list[BaseMessage]) -> Any:
        if not self._results:
            raise AssertionError("FakeStructuredRunnable called more times than expected")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeChatModel:
    """Stands in for ``get_chat_model()`` — never makes a real LLM call."""

    def __init__(self, results: list[Any]) -> None:
        self._results = results

    def with_structured_output(self, schema: type[Any]) -> FakeStructuredRunnable:
        return FakeStructuredRunnable(self._results)


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that patches ``get_chat_model`` in the given agent module."""

    def _patch(module: str, *results: Any) -> FakeChatModel:
        fake = FakeChatModel(list(results))
        monkeypatch.setattr(f"{module}.get_chat_model", lambda *a, **kw: fake)
        return fake

    return _patch
