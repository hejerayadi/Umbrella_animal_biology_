from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import Request, Response
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.security import new_token, token_digest
from ...core.sessions import SessionRecord, SessionState, SessionStore
from ...models import AuthToken, TokenPurpose


def set_session_cookie(request: Request, response: Response, record: SessionRecord) -> None:
    settings = request.app.state.settings
    response.set_cookie(
        settings.session_cookie_name,
        record.session_id,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
        max_age=settings.session_absolute_ttl_seconds,
    )


def clear_session_cookie(request: Request, response: Response) -> None:
    settings = request.app.state.settings
    response.delete_cookie(
        settings.session_cookie_name,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


async def rotate_session(
    request: Request,
    response: Response,
    sessions: SessionStore,
    record: SessionRecord,
    state: SessionState,
) -> SessionRecord:
    rotated = await sessions.rotate(record, state)
    set_session_cookie(request, response, rotated)
    return rotated


async def issue_auth_token(
    db: AsyncSession,
    user_id: object,
    purpose: TokenPurpose,
    ttl: timedelta,
) -> str:
    now = datetime.now(UTC)
    await db.execute(
        update(AuthToken)
        .where(
            AuthToken.user_id == user_id,
            AuthToken.purpose == purpose,
            AuthToken.used_at.is_(None),
        )
        .values(used_at=now)
    )
    raw = new_token()
    db.add(
        AuthToken(
            user_id=user_id,
            purpose=purpose,
            token_hash=token_digest(raw),
            expires_at=now + ttl,
        )
    )
    return raw

