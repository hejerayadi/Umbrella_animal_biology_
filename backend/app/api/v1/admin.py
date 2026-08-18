from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.image_store import IMAGE_STORE

from ...contracts import ApiProblem, success
from ...core.dependencies import (
    Identity,
    get_db,
    get_session_store,
    require_admin,
    require_user_csrf as require_csrf,
)
from ...core.security import hash_password, normalize_email, temporary_password
from ...core.sessions import SessionStore
from ...models import (
    AdminNotification, AuditEvent, BiologistProfile, Invitation, MfaFactor,
    MfaRecoveryCode, TokenPurpose, User, UserRole, UserStatus,
)
from ...schemas import BulkUserRequest, InvitationRequest, RejectRequest, user_view
from ...services.audit import audit
from .helpers import issue_auth_token

router = APIRouter(prefix="/admin", tags=["administration"])


async def _target(db: AsyncSession, user_id: uuid.UUID) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise ApiProblem(404, "USER_NOT_FOUND", "User not found", "The requested user does not exist.")
    return user


async def _active_admin_count(db: AsyncSession) -> int:
    return int(await db.scalar(select(func.count()).select_from(User).where(
        User.role == UserRole.ADMIN, User.status == UserStatus.ACTIVE
    )) or 0)


async def _bulk_targets(db: AsyncSession, user_ids: list[uuid.UUID]) -> list[User]:
    unique_ids = list(dict.fromkeys(user_ids))
    users = list((await db.scalars(select(User).where(User.id.in_(unique_ids)))).all())
    if len(users) != len(unique_ids):
        found_ids = {user.id for user in users}
        missing = [str(user_id) for user_id in unique_ids if user_id not in found_ids]
        raise ApiProblem(
            404,
            "USERS_NOT_FOUND",
            "Some users were not found",
            f"Missing user IDs: {', '.join(missing)}",
        )
    return users


async def _delete_targets(
    db: AsyncSession,
    request: Request,
    sessions: SessionStore,
    actor: User,
    targets: list[User],
) -> tuple[list[str], int]:
    if any(target.id == actor.id for target in targets):
        raise ApiProblem(
            409,
            "SELF_DELETE_FORBIDDEN",
            "Cannot delete your account",
            "Remove your own account from the selection.",
        )
    active_admin_targets = sum(
        target.role == UserRole.ADMIN and target.status == UserStatus.ACTIVE
        for target in targets
    )
    if active_admin_targets and await _active_admin_count(db) - active_admin_targets < 1:
        raise ApiProblem(
            409,
            "LAST_ADMIN",
            "Last administrator protected",
            "At least one active administrator must remain outside the selection.",
        )

    deleted_ids = [str(target.id) for target in targets]
    for target in targets:
        await sessions.revoke_user(target.id)
        audit(
            db,
            request,
            "admin.user.deleted",
            "SUCCESS",
            actor_id=actor.id,
            target_id=target.id,
            details={"role": target.role.value, "status": target.status.value},
        )
        await db.delete(target)
    await db.commit()
    removed_images = sum(IMAGE_STORE.delete_for_owner(user_id) for user_id in deleted_ids)
    return deleted_ids, removed_images


@router.get("/stats")
async def admin_stats(
    request: Request,
    identity: Identity = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    status_rows = (await db.execute(
        select(User.status, func.count()).group_by(User.status)
    )).all()
    role_rows = (await db.execute(
        select(User.role, func.count()).group_by(User.role)
    )).all()
    status_counts = {status.value: int(count) for status, count in status_rows}
    role_counts = {role.value: int(count) for role, count in role_rows}
    unread_notifications = int(await db.scalar(
        select(func.count()).select_from(AdminNotification).where(
            AdminNotification.recipient_admin_id == identity.user.id,
            AdminNotification.read_at.is_(None),
        )
    ) or 0)
    total_notifications = int(await db.scalar(
        select(func.count()).select_from(AdminNotification).where(
            AdminNotification.recipient_admin_id == identity.user.id
        )
    ) or 0)
    total_audit_events = int(await db.scalar(
        select(func.count()).select_from(AuditEvent)
    ) or 0)
    failed_audit_events = int(await db.scalar(
        select(func.count()).select_from(AuditEvent).where(AuditEvent.outcome != "SUCCESS")
    ) or 0)
    active_biologists = int(await db.scalar(
        select(func.count()).select_from(User).where(
            User.role == UserRole.BIOLOGIST, User.status == UserStatus.ACTIVE
        )
    ) or 0)
    return success(request, {
        "total_users": sum(status_counts.values()),
        "active_users": status_counts.get(UserStatus.ACTIVE.value, 0),
        "active_biologists": active_biologists,
        "pending_applications": status_counts.get(UserStatus.PENDING_APPROVAL.value, 0),
        "rejected_applications": status_counts.get(UserStatus.REJECTED.value, 0),
        "invited_users": status_counts.get(UserStatus.INVITED.value, 0),
        "disabled_users": status_counts.get(UserStatus.DISABLED.value, 0),
        "status_counts": status_counts,
        "role_counts": role_counts,
        "unread_notifications": unread_notifications,
        "total_notifications": total_notifications,
        "total_audit_events": total_audit_events,
        "failed_audit_events": failed_audit_events,
    })


@router.get("/applications")
async def applications(
    request: Request,
    state: UserStatus | None = Query(default=None),
    search: str = Query(default="", max_length=200),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    _: Identity = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    states = [UserStatus.PENDING_APPROVAL, UserStatus.REJECTED]
    conditions = [User.role == UserRole.BIOLOGIST]
    conditions.append(User.status == state if state else User.status.in_(states))
    if search.strip():
        term = f"%{search.strip()}%"
        conditions.append(or_(
            User.email.ilike(term),
            BiologistProfile.full_name.ilike(term),
            BiologistProfile.institution.ilike(term),
        ))
    total = int(await db.scalar(
        select(func.count()).select_from(User).outerjoin(BiologistProfile).where(*conditions)
    ) or 0)
    users = list((await db.scalars(
        select(User).outerjoin(BiologistProfile).where(*conditions).order_by(User.created_at.desc())
        .offset((page - 1) * page_size).limit(page_size)
    )).all())
    return success(request, [user_view(user) for user in users], pagination={
        "page": page, "page_size": page_size, "total": total,
    })


@router.post("/applications/{user_id}/approve")
async def approve_application(
    user_id: uuid.UUID,
    request: Request,
    background: BackgroundTasks,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    user = await _target(db, user_id)
    if user.status != UserStatus.PENDING_APPROVAL:
        raise ApiProblem(409, "INVALID_STATUS", "Application cannot be approved", "Only pending applications can be approved.")
    user.status = UserStatus.ACTIVE
    audit(db, request, "admin.application.approved", "SUCCESS", actor_id=identity.user.id, target_id=user.id)
    await db.commit()
    background.add_task(
        request.app.state.email_service.send, user.email, "Your Umbrella application was approved",
        "Your account is active. You can now sign in to Umbrella.",
    )
    return success(request, user_view(user))


@router.post("/applications/{user_id}/reject")
async def reject_application(
    user_id: uuid.UUID,
    payload: RejectRequest,
    request: Request,
    background: BackgroundTasks,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    user = await _target(db, user_id)
    if user.status != UserStatus.PENDING_APPROVAL:
        raise ApiProblem(409, "INVALID_STATUS", "Application cannot be rejected", "Only pending applications can be rejected.")
    user.status = UserStatus.REJECTED
    audit(db, request, "admin.application.rejected", "SUCCESS", actor_id=identity.user.id, target_id=user.id, details={"reason": payload.reason})
    await db.commit()
    background.add_task(
        request.app.state.email_service.send, user.email, "Your Umbrella application",
        f"Your application was not approved. Reason: {payload.reason}",
    )
    return success(request, user_view(user))


@router.post("/applications/{user_id}/reopen")
async def reopen_application(
    user_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    user = await _target(db, user_id)
    if user.status != UserStatus.REJECTED:
        raise ApiProblem(409, "INVALID_STATUS", "Application cannot be reopened", "Only rejected applications can be reopened.")
    user.status = UserStatus.PENDING_APPROVAL
    audit(db, request, "admin.application.reopened", "SUCCESS", actor_id=identity.user.id, target_id=user.id)
    await db.commit()
    return success(request, user_view(user))


@router.post("/invitations", status_code=201)
async def invite_biologist(
    payload: InvitationRequest,
    request: Request,
    background: BackgroundTasks,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    email = normalize_email(str(payload.email))
    if await db.scalar(select(User).where(User.email == email)) is not None:
        raise ApiProblem(409, "EMAIL_EXISTS", "Email already registered", "Manage the existing account instead.")
    user = User(
        email=email, password_hash=None, role=UserRole.BIOLOGIST, status=UserStatus.INVITED,
        profile=BiologistProfile(
            full_name=payload.full_name.strip(), institution=payload.institution.strip(),
            professional_title=payload.professional_title.strip(), country=payload.country.upper(),
            orcid=payload.orcid, motivation="Invited by an Umbrella administrator.",
            specialties=payload.specialties,
        ),
    )
    db.add(user)
    await db.flush()
    raw_password = temporary_password()
    invitation = Invitation(
        user_id=user.id, invited_by_id=identity.user.id,
        temporary_password_hash=hash_password(raw_password),
        expires_at=datetime.now(UTC) + timedelta(hours=request.app.state.settings.invitation_ttl_hours),
    )
    db.add(invitation)
    audit(db, request, "admin.invitation.created", "SUCCESS", actor_id=identity.user.id, target_id=user.id)
    await db.commit()
    background.add_task(request.app.state.email_service.invitation, user.email, raw_password)
    return success(request, user_view(user))


@router.post("/invitations/{user_id}/resend")
async def resend_invitation(
    user_id: uuid.UUID,
    request: Request,
    background: BackgroundTasks,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    user = await _target(db, user_id)
    if user.status != UserStatus.INVITED:
        raise ApiProblem(409, "INVALID_STATUS", "Invitation cannot be resent", "This user is not awaiting an invitation.")
    now = datetime.now(UTC)
    old = list((await db.scalars(select(Invitation).where(
        Invitation.user_id == user.id, Invitation.consumed_at.is_(None)
    ))).all())
    for invitation in old:
        invitation.consumed_at = now
    raw_password = temporary_password()
    db.add(Invitation(
        user_id=user.id, invited_by_id=identity.user.id,
        temporary_password_hash=hash_password(raw_password),
        expires_at=now + timedelta(hours=request.app.state.settings.invitation_ttl_hours),
    ))
    audit(db, request, "admin.invitation.resent", "SUCCESS", actor_id=identity.user.id, target_id=user.id)
    await db.commit()
    background.add_task(request.app.state.email_service.invitation, user.email, raw_password)
    return success(request, {"resent": True})


@router.get("/users")
async def list_users(
    request: Request,
    role: UserRole | None = Query(default=None),
    state: UserStatus | None = Query(default=None),
    search: str = Query(default="", max_length=200),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    _: Identity = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    conditions = []
    if role:
        conditions.append(User.role == role)
    if state:
        conditions.append(User.status == state)
    if search.strip():
        term = f"%{search.strip()}%"
        conditions.append(or_(
            User.email.ilike(term),
            BiologistProfile.full_name.ilike(term),
            BiologistProfile.institution.ilike(term),
        ))
    total = int(await db.scalar(
        select(func.count()).select_from(User).outerjoin(BiologistProfile).where(*conditions)
    ) or 0)
    users = list((await db.scalars(
        select(User).outerjoin(BiologistProfile).where(*conditions).order_by(User.created_at.desc())
        .offset((page - 1) * page_size).limit(page_size)
    )).all())
    return success(request, [user_view(user) for user in users], pagination={
        "page": page, "page_size": page_size, "total": total,
    })


@router.get("/users/{user_id}")
async def get_user(
    user_id: uuid.UUID,
    request: Request,
    _: Identity = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    return success(request, user_view(await _target(db, user_id)))


@router.post("/users/{user_id}/disable")
async def disable_user(
    user_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    target = await _target(db, user_id)
    if target.id == identity.user.id:
        raise ApiProblem(409, "SELF_DISABLE_FORBIDDEN", "Cannot disable your account", "Ask another administrator.")
    if target.role == UserRole.ADMIN and await _active_admin_count(db) <= 1:
        raise ApiProblem(409, "LAST_ADMIN", "Last administrator protected", "At least one active administrator is required.")
    target.status = UserStatus.DISABLED
    target.disabled_at = datetime.now(UTC)
    await sessions.revoke_user(target.id)
    audit(db, request, "admin.user.disabled", "SUCCESS", actor_id=identity.user.id, target_id=target.id)
    await db.commit()
    return success(request, user_view(target))


@router.post("/users/{user_id}/reactivate")
async def reactivate_user(
    user_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    target = await _target(db, user_id)
    if target.status != UserStatus.DISABLED:
        raise ApiProblem(409, "INVALID_STATUS", "Account cannot be reactivated", "Only disabled accounts can be reactivated.")
    target.status = UserStatus.ACTIVE
    target.disabled_at = None
    audit(db, request, "admin.user.reactivated", "SUCCESS", actor_id=identity.user.id, target_id=target.id)
    await db.commit()
    return success(request, user_view(target))


@router.post("/users/{user_id}/revoke-sessions")
async def revoke_sessions(
    user_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    target = await _target(db, user_id)
    await sessions.revoke_user(target.id)
    audit(db, request, "admin.sessions.revoked", "SUCCESS", actor_id=identity.user.id, target_id=target.id)
    await db.commit()
    return success(request, {"revoked": True})


@router.post("/users/{user_id}/force-password-reset")
async def force_password_reset(
    user_id: uuid.UUID,
    request: Request,
    background: BackgroundTasks,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    target = await _target(db, user_id)
    if target.status != UserStatus.ACTIVE:
        raise ApiProblem(409, "INVALID_STATUS", "Reset unavailable", "Only active users can reset their password.")
    token = await issue_auth_token(
        db, target.id, TokenPurpose.PASSWORD_RESET,
        timedelta(minutes=request.app.state.settings.password_reset_ttl_minutes),
    )
    await sessions.revoke_user(target.id)
    audit(db, request, "admin.password.reset_forced", "SUCCESS", actor_id=identity.user.id, target_id=target.id)
    await db.commit()
    background.add_task(request.app.state.email_service.password_reset, target.email, token)
    return success(request, {"reset_required": True})


@router.post("/users/{user_id}/reset-mfa")
async def reset_mfa(
    user_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    target = await _target(db, user_id)
    if target.role != UserRole.ADMIN:
        raise ApiProblem(409, "MFA_NOT_APPLICABLE", "MFA reset unavailable", "MFA is mandatory only for administrators.")
    await db.execute(delete(MfaRecoveryCode).where(MfaRecoveryCode.user_id == target.id))
    await db.execute(delete(MfaFactor).where(MfaFactor.user_id == target.id))
    await sessions.revoke_user(target.id)
    audit(db, request, "admin.mfa.reset", "SUCCESS", actor_id=identity.user.id, target_id=target.id)
    await db.commit()
    return success(request, {"mfa_reset": True})


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    target = await _target(db, user_id)
    deleted_ids, removed_images = await _delete_targets(
        db, request, sessions, identity.user, [target]
    )
    return success(
        request,
        {"deleted": True, "user_id": deleted_ids[0], "removed_images": removed_images},
    )


@router.post("/bulk-users/delete")
async def bulk_delete_users(
    payload: BulkUserRequest,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    targets = await _bulk_targets(db, payload.user_ids)
    deleted_ids, removed_images = await _delete_targets(
        db, request, sessions, identity.user, targets
    )
    return success(request, {
        "deleted": len(deleted_ids),
        "user_ids": deleted_ids,
        "removed_images": removed_images,
    })


@router.post("/bulk-users/reactivate")
async def bulk_reactivate_users(
    payload: BulkUserRequest,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    targets = await _bulk_targets(db, payload.user_ids)
    invalid = [target for target in targets if target.status != UserStatus.DISABLED]
    if invalid:
        raise ApiProblem(
            409,
            "BULK_INVALID_STATUS",
            "Some accounts cannot be reactivated",
            "Only disabled accounts can be reactivated. Adjust the selection and try again.",
        )
    for target in targets:
        target.status = UserStatus.ACTIVE
        target.disabled_at = None
        audit(
            db,
            request,
            "admin.user.reactivated",
            "SUCCESS",
            actor_id=identity.user.id,
            target_id=target.id,
            details={"bulk": True},
        )
    await db.commit()
    return success(request, {
        "reactivated": len(targets),
        "user_ids": [str(target.id) for target in targets],
    })


@router.get("/notifications")
async def notifications(
    request: Request,
    unread_only: bool = False,
    identity: Identity = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    conditions = [AdminNotification.recipient_admin_id == identity.user.id]
    if unread_only:
        conditions.append(AdminNotification.read_at.is_(None))
    rows = list((await db.scalars(
        select(AdminNotification).where(*conditions)
        .order_by(AdminNotification.created_at.desc()).limit(100)
    )).all())
    return success(request, [{
        "id": str(item.id), "kind": item.kind, "title": item.title,
        "message": item.message, "subject_user_id": str(item.subject_user_id)
        if item.subject_user_id else None,
        "read_at": item.read_at.isoformat() if item.read_at else None,
        "created_at": item.created_at.isoformat(),
    } for item in rows])


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: uuid.UUID,
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    notification = await db.get(AdminNotification, notification_id)
    if notification is None or notification.recipient_admin_id != identity.user.id:
        raise ApiProblem(404, "NOTIFICATION_NOT_FOUND", "Notification not found", "The notification does not exist.")
    notification.read_at = datetime.now(UTC)
    await db.commit()
    return success(request, {"read": True})


@router.post("/notifications/read-all")
async def mark_all_notifications_read(
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if identity.user.role != UserRole.ADMIN:
        raise ApiProblem(403, "ADMIN_REQUIRED", "Administrator required", "You do not have permission.")
    rows = list((await db.scalars(select(AdminNotification).where(
        AdminNotification.recipient_admin_id == identity.user.id,
        AdminNotification.read_at.is_(None),
    ))).all())
    now = datetime.now(UTC)
    for notification in rows:
        notification.read_at = now
    audit(
        db,
        request,
        "admin.notifications.read_all",
        "SUCCESS",
        actor_id=identity.user.id,
        details={"count": len(rows)},
    )
    await db.commit()
    return success(request, {"read": len(rows)})


@router.get("/audit-events")
async def audit_events(
    request: Request,
    event_type: str = Query(default="", max_length=96),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    _: Identity = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    conditions = [AuditEvent.event_type == event_type] if event_type else []
    total = int(await db.scalar(select(func.count()).select_from(AuditEvent).where(*conditions)) or 0)
    rows = list((await db.scalars(
        select(AuditEvent).where(*conditions).order_by(AuditEvent.created_at.desc())
        .offset((page - 1) * page_size).limit(page_size)
    )).all())
    return success(request, [{
        "id": str(item.id), "actor_user_id": str(item.actor_user_id) if item.actor_user_id else None,
        "target_user_id": str(item.target_user_id) if item.target_user_id else None,
        "event_type": item.event_type, "outcome": item.outcome,
        "ip_address": item.ip_address, "request_id": item.request_id,
        "details": item.details, "created_at": item.created_at.isoformat(),
    } for item in rows], pagination={"page": page, "page_size": page_size, "total": total})
