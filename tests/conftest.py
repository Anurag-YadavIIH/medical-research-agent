"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from medical_research_agent.api.main import create_app
from medical_research_agent.api.rate_limit import reset_rate_limits
from medical_research_agent.database.models import Base
from medical_research_agent.database.session import get_session

# Postgres in production; SQLite (via aiosqlite) in tests — see
# database/models.py's _JSONType for the JSONB/JSON dialect split this relies on.
# A real Postgres testcontainer was the alternative, but it would need Docker in
# CI for every test run just to exercise ORM wiring that doesn't depend on any
# Postgres-only behavior, so dialect-portable columns are the better trade-off here.


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session

    await engine.dispose()


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    # The /research rate limiter keys off client IP in module-level state, so
    # without a reset, request counts would leak across unrelated tests sharing
    # TestClient's fixed host and trip false 429s.
    reset_rate_limits()


@pytest.fixture
def client(db_session: AsyncSession) -> TestClient:
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    return TestClient(app)


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


class _FakeAIMessage:
    """Stands in for an ``AIMessage`` — just enough for ``.content`` access."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """Stands in for ``get_chat_model()`` — never makes a real LLM call.

    Supports both call shapes used in this codebase: structured-output agents
    (``.with_structured_output(Schema).ainvoke(...)``) and plain chat
    (``.ainvoke(...)`` directly, e.g. the project chat service), sharing one
    results queue either way.
    """

    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self.call_count = 0

    def with_structured_output(self, schema: type[Any]) -> FakeStructuredRunnable:
        return FakeStructuredRunnable(self._results)

    async def ainvoke(self, messages: list[BaseMessage]) -> _FakeAIMessage:
        self.call_count += 1
        if not self._results:
            raise AssertionError("FakeChatModel called more times than expected")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return _FakeAIMessage(result)


@pytest.fixture
def fake_chat_model(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that patches ``get_chat_model`` in the given agent module."""

    def _patch(module: str, *results: Any) -> FakeChatModel:
        fake = FakeChatModel(list(results))
        monkeypatch.setattr(f"{module}.get_chat_model", lambda *a, **kw: fake)
        return fake

    return _patch


class FakeEmbeddingsModel:
    """Stands in for ``get_embeddings_model()`` — never makes a real API call."""

    def __init__(
        self, query_vector: list[float], document_vectors: list[list[float]] | None
    ) -> None:
        self._query_vector = query_vector
        self._document_vectors = document_vectors

    async def aembed_query(self, text: str) -> list[float]:
        return self._query_vector

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._document_vectors is not None:
            return self._document_vectors
        return [self._query_vector for _ in texts]


@pytest.fixture
def fake_embeddings_model(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that patches ``get_embeddings_model`` in the given module."""

    def _patch(
        module: str,
        query_vector: list[float] | None = None,
        document_vectors: list[list[float]] | None = None,
    ) -> FakeEmbeddingsModel:
        fake = FakeEmbeddingsModel(query_vector or [1.0, 0.0], document_vectors)
        monkeypatch.setattr(f"{module}.get_embeddings_model", lambda *a, **kw: fake)
        return fake

    return _patch
