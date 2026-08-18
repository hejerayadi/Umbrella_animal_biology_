from __future__ import annotations

import json
import secrets
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum

from redis.asyncio import Redis

from .config import Settings


class SessionState(StrEnum):
    AUTHENTICATED = "AUTHENTICATED"
    MFA_REQUIRED = "MFA_REQUIRED"
    MFA_ENROLLMENT_REQUIRED = "MFA_ENROLLMENT_REQUIRED"
    PASSWORD_CHANGE_REQUIRED = "PASSWORD_CHANGE_REQUIRED"


@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    state: str
    csrf_token: str
    created_at: int
    last_seen_at: int


class SessionStore:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings

    @staticmethod
    def _key(session_id: str) -> str:
        return f"auth:session:{session_id}"

    @staticmethod
    def _user_key(user_id: str) -> str:
        return f"auth:user-sessions:{user_id}"

    async def create(self, user_id: uuid.UUID | str, state: SessionState) -> SessionRecord:
        now = int(time.time())
        record = SessionRecord(
            session_id=secrets.token_urlsafe(32),
            user_id=str(user_id),
            state=state.value,
            csrf_token=secrets.token_urlsafe(24),
            created_at=now,
            last_seen_at=now,
        )
        await self._write(record)
        return record

    async def _write(self, record: SessionRecord) -> None:
        ttl = (
            self.settings.session_idle_ttl_seconds
            if record.state == SessionState.AUTHENTICATED.value
            else self.settings.restricted_session_ttl_seconds
        )
        pipeline = self.redis.pipeline()
        pipeline.set(self._key(record.session_id), json.dumps(asdict(record)), ex=ttl)
        pipeline.sadd(self._user_key(record.user_id), record.session_id)
        pipeline.expire(self._user_key(record.user_id), self.settings.session_absolute_ttl_seconds)
        await pipeline.execute()

    async def load(self, session_id: str | None) -> SessionRecord | None:
        if not session_id:
            return None
        raw = await self.redis.get(self._key(session_id))
        if not raw:
            return None
        payload = json.loads(raw)
        record = SessionRecord(**payload)
        now = int(time.time())
        if now - record.created_at >= self.settings.session_absolute_ttl_seconds:
            await self.delete(record)
            return None
        record.last_seen_at = now
        await self._write(record)
        return record

    async def rotate(self, record: SessionRecord, state: SessionState) -> SessionRecord:
        await self.delete(record)
        return await self.create(record.user_id, state)

    async def delete(self, record: SessionRecord) -> None:
        pipeline = self.redis.pipeline()
        pipeline.delete(self._key(record.session_id))
        pipeline.srem(self._user_key(record.user_id), record.session_id)
        await pipeline.execute()

    async def revoke_user(self, user_id: uuid.UUID | str, *, except_id: str | None = None) -> None:
        key = self._user_key(str(user_id))
        session_ids = await self.redis.smembers(key)
        pipeline = self.redis.pipeline()
        for raw_id in session_ids:
            session_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            if session_id != except_id:
                pipeline.delete(self._key(session_id))
                pipeline.srem(key, session_id)
        await pipeline.execute()

    async def rate_limit(self, bucket: str, limit: int, window_seconds: int) -> bool:
        key = f"auth:rate:{bucket}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, window_seconds)
        return count <= limit

    async def reset_rate_limit(self, bucket: str) -> None:
        await self.redis.delete(f"auth:rate:{bucket}")
