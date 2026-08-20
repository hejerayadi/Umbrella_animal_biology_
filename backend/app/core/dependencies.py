from __future__ import annotations

import hmac
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..contracts import ApiProblem
from ..models import User, UserRole, UserStatus
from .sessions import SessionRecord, SessionState, SessionStore


@dataclass
class Identity:
    user: User
    session: SessionRecord


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


async def get_identity(
    request: Request,
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> Identity:
    cookie_name = request.app.state.settings.session_cookie_name
    record = await sessions.load(request.cookies.get(cookie_name))
    if record is None:
        raise ApiProblem(status.HTTP_401_UNAUTHORIZED, "AUTH_REQUIRED", "Authentication required", "Sign in to continue.")
    user = await db.get(User, uuid.UUID(record.user_id))
    if user is None or user.status == UserStatus.DISABLED:
        await sessions.delete(record)
        raise ApiProblem(status.HTTP_401_UNAUTHORIZED, "SESSION_INVALID", "Session invalid", "This session is no longer valid.")
    request.state.identity = Identity(user=user, session=record)
    return request.state.identity


async def require_user(identity: Identity = Depends(get_identity)) -> Identity:
    if identity.session.state != SessionState.AUTHENTICATED.value or identity.user.status != UserStatus.ACTIVE:
        code = identity.session.state if identity.session.state != SessionState.AUTHENTICATED.value else "ACCOUNT_NOT_ACTIVE"
        raise ApiProblem(status.HTTP_403_FORBIDDEN, code, "Additional action required", "Complete the required authentication step.")
    return identity


async def require_admin(identity: Identity = Depends(require_user)) -> Identity:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(status.HTTP_403_FORBIDDEN, "ADMIN_REQUIRED", "Administrator required", "You do not have permission to perform this action.")
    return identity


async def require_csrf(request: Request, identity: Identity = Depends(get_identity)) -> Identity:
    supplied = request.headers.get("X-CSRF-Token", "")
    if not supplied or not hmac.compare_digest(supplied, identity.session.csrf_token):
        raise ApiProblem(status.HTTP_403_FORBIDDEN, "CSRF_INVALID", "CSRF validation failed", "Refresh the page and try again.")
    return identity


async def require_user_csrf(
    request: Request, identity: Identity = Depends(require_user)
) -> Identity:
    supplied = request.headers.get("X-CSRF-Token", "")
    if not supplied or not hmac.compare_digest(supplied, identity.session.csrf_token):
        raise ApiProblem(status.HTTP_403_FORBIDDEN, "CSRF_INVALID", "CSRF validation failed", "Refresh the page and try again.")
    return identity
