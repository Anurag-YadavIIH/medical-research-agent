"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medical_research_agent.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
