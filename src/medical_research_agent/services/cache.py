"""Redis-backed JSON cache with graceful degradation.

If Redis is unavailable, cache operations no-op rather than raising, so a missing
cache never takes down a request.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis

from medical_research_agent.config import Settings, get_settings
from medical_research_agent.logging_config import get_logger

log = get_logger("cache")


class Cache:
    """Minimal async JSON cache."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            log.warning("cache.get_failed", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            await self._redis.set(key, json.dumps(value), ex=ttl or self.settings.cache_ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            log.warning("cache.set_failed", key=key, error=str(exc))

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001
            return False

    async def aclose(self) -> None:
        await self._redis.aclose()
