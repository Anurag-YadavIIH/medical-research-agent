"""Smoke tests for the application skeleton."""

from __future__ import annotations

from medical_research_agent import __version__


def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert "redis" in body["checks"]


def test_openapi_exposes_required_endpoints(client) -> None:
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    assert "/research" in paths
    assert "/studies/{query_id}" in paths
    assert "/health" in paths


def test_disclaimer_in_openapi(client) -> None:
    spec = client.get("/openapi.json").json()
    assert "research and educational purposes only" in spec["info"]["description"].lower()
