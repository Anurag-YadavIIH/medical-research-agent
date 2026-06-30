"""Minimal in-process rate limiter for POST /research.

This is a local, in-memory sliding-window limiter — state lives in a module-level
dict, so it resets on process restart and is NOT shared across replicas. It exists
to stop a single client from trivially burning LLM/NCBI quota against a
single-instance demo deployment. It is not a substitute for real infra-level rate
limiting (an API gateway, a Redis-backed limiter, etc.) in a multi-replica
production deployment — see the README's Security notes section.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

WINDOW_SECONDS = 60.0
MAX_REQUESTS_PER_WINDOW = 10

_requests_by_client: dict[str, deque[float]] = defaultdict(deque)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: raise 429 once a client exceeds the per-window cap."""
    now = time.monotonic()
    bucket = _requests_by_client[_client_key(request)]
    while bucket and now - bucket[0] > WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: max {MAX_REQUESTS_PER_WINDOW} requests per "
                f"{int(WINDOW_SECONDS)}s per client. Please slow down and retry shortly."
            ),
        )
    bucket.append(now)


def reset_rate_limits() -> None:
    """Test-only: clear all tracked request buckets."""
    _requests_by_client.clear()
