"""
Redis-backed registry for SandboxSession metadata.

Keyed as `sandbox:session:{session_id}`. This is deliberately separate from
the E2B/Docker backend classes so both backends share one registry
implementation and one TTL-reaping code path.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from src.sandbox.manager import SessionNotFoundError
from src.sandbox.models import SandboxSession

logger = logging.getLogger(__name__)

_KEY_PREFIX = "sandbox:session:"
_INDEX_KEY = "sandbox:session_index"


class SessionRegistry:
    """Thin async wrapper around Redis for SandboxSession CRUD."""

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    async def save(self, session: SandboxSession) -> None:
        await self._redis.set(
            self._key(session.session_id),
            session.model_dump_json(),
            ex=session.ttl_seconds + 60,  # grace period beyond app-level TTL
        )
        await self._redis.sadd(_INDEX_KEY, session.session_id)

    async def get(self, session_id: str) -> SandboxSession:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            raise SessionNotFoundError(f"No sandbox session found for id={session_id!r}")
        return SandboxSession.model_validate_json(raw)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
        await self._redis.srem(_INDEX_KEY, session_id)

    async def list_all(self) -> list[SandboxSession]:
        ids = await self._redis.smembers(_INDEX_KEY)
        sessions: list[SandboxSession] = []
        for raw_id in ids:
            session_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            try:
                sessions.append(await self.get(session_id))
            except SessionNotFoundError:
                # Key expired via Redis TTL between SMEMBERS and GET — prune the
                # dangling index entry rather than surfacing an error.
                await self._redis.srem(_INDEX_KEY, session_id)
        return sessions
