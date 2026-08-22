from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Request
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import AuditEvent


def audit(
    db: AsyncSession,
    request: Request,
    event_type: str,
    outcome: str,
    *,
    actor_id: uuid.UUID | None = None,
    target_id: uuid.UUID | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    db.add(
        AuditEvent(
            actor_user_id=actor_id,
            target_user_id=target_id,
            event_type=event_type,
            outcome=outcome,
            ip_address=request.client.host if request.client else None,
            request_id=getattr(request.state, "request_id", None),
            details=details or {},
        )
    )


async def purge_old_audit_events(db: AsyncSession, days: int) -> None:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    await db.execute(delete(AuditEvent).where(AuditEvent.created_at < cutoff))
    await db.commit()

