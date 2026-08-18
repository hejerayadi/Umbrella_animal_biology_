from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pyotp
from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...contracts import ApiProblem, success
from ...core.dependencies import Identity, get_db, get_identity, get_session_store, require_csrf
from ...core.security import (
    decrypt_secret, encrypt_secret, hash_password, new_totp_secret, normalize_email,
    recovery_codes, token_digest, validate_password, verify_password, verify_totp,
)
from ...core.sessions import SessionState, SessionStore
from ...models import (
    AdminNotification, AuthToken, BiologistProfile, Invitation, MfaFactor,
    MfaRecoveryCode, TokenPurpose, User, UserRole, UserStatus,
)
from ...schemas import (
    EmailRequest, LoginRequest, MfaCodeRequest, PasswordChangeRequest,
    PasswordResetRequest, RegistrationRequest, TokenRequest, user_view,
)
from ...services.audit import audit
from .helpers import clear_session_cookie, issue_auth_token, rotate_session, set_session_cookie

router = APIRouter(prefix="/auth", tags=["authentication"])


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def _public_limit(request: Request, sessions: SessionStore, name: str, identity: str = "") -> None:
    settings = request.app.state.settings
    discriminator = token_digest(identity)[:16] if identity else _client_key(request)
    allowed = await sessions.rate_limit(
        f"{name}:{_client_key(request)}:{discriminator}",
        settings.public_rate_limit,
        settings.public_rate_window_seconds,
    )
    if not allowed:
        raise ApiProblem(429, "RATE_LIMITED", "Too many requests", "Try again later.")


@router.post("/register", status_code=status.HTTP_202_ACCEPTED)
async def register(
    payload: RegistrationRequest,
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    email = normalize_email(str(payload.email))
    await _public_limit(request, sessions, "register", email)
    try:
        validate_password(payload.password, email=email, name=payload.full_name)
    except ValueError as exc:
        raise ApiProblem(422, "WEAK_PASSWORD", "Password rejected", str(exc)) from exc
    existing = await db.scalar(select(User).where(User.email == email))
    if existing is None:
        user = User(
            email=email,
            password_hash=hash_password(payload.password),
            role=UserRole.BIOLOGIST,
            status=UserStatus.PENDING_EMAIL,
            profile=BiologistProfile(
                full_name=payload.full_name.strip(),
                institution=payload.institution.strip(),
                professional_title=payload.professional_title.strip(),
                country=payload.country,
                orcid=payload.orcid,
                motivation=payload.motivation.strip(),
                specialties=payload.specialties,
            ),
        )
        db.add(user)
        await db.flush()
        token = await issue_auth_token(
            db, user.id, TokenPurpose.EMAIL_VERIFICATION,
            timedelta(minutes=request.app.state.settings.email_verification_ttl_minutes),
        )
        audit(db, request, "auth.registration.created", "SUCCESS", target_id=user.id)
        await db.commit()
        background.add_task(request.app.state.email_service.verification, email, token)
    elif existing.status == UserStatus.PENDING_EMAIL:
        # The address was registered but never verified - usually because the
        # first verification email never arrived. Re-issue the link instead of
        # silently doing nothing, which would strand the applicant with no way
        # to ever complete signup. The response below is identical either way,
        # so this still reveals nothing about whether the address exists, and
        # the endpoint is rate limited exactly as before.
        # The stored password and profile are deliberately left untouched: a
        # second registration must never be able to overwrite a pending account.
        token = await issue_auth_token(
            db, existing.id, TokenPurpose.EMAIL_VERIFICATION,
            timedelta(minutes=request.app.state.settings.email_verification_ttl_minutes),
        )
        audit(db, request, "auth.registration.reissued", "SUCCESS", target_id=existing.id)
        await db.commit()
        background.add_task(request.app.state.email_service.verification, email, token)
    return success(request, {"message": "If the address can be registered, a verification email has been sent."})


@router.post("/email-verification/confirm")
async def confirm_email(
    payload: TokenRequest,
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    now = datetime.now(UTC)
    token = await db.scalar(select(AuthToken).where(
        AuthToken.token_hash == token_digest(payload.token),
        AuthToken.purpose == TokenPurpose.EMAIL_VERIFICATION,
        AuthToken.used_at.is_(None), AuthToken.expires_at > now,
    ))
    if token is None:
        raise ApiProblem(400, "TOKEN_INVALID", "Invalid verification link", "This link is invalid or expired.")
    user = await db.get(User, token.user_id)
    if user is None or user.status != UserStatus.PENDING_EMAIL:
        raise ApiProblem(400, "TOKEN_INVALID", "Invalid verification link", "This link is invalid or expired.")
    token.used_at = now
    user.email_verified_at = now
    user.status = UserStatus.PENDING_APPROVAL
    admins = list((await db.scalars(select(User).where(
        User.role == UserRole.ADMIN, User.status == UserStatus.ACTIVE
    ))).all())
    for admin in admins:
        db.add(AdminNotification(
            recipient_admin_id=admin.id,
            subject_user_id=user.id,
            kind="BIOLOGIST_APPLICATION",
            title="New biologist application",
            message=f"{user.profile.full_name if user.profile else user.email} is awaiting approval.",
        ))
        background.add_task(
            request.app.state.email_service.send, admin.email,
            "New Umbrella biologist application",
            f"A verified application from {user.email} is waiting in the admin dashboard.",
        )
    audit(db, request, "auth.email.verified", "SUCCESS", target_id=user.id)
    await db.commit()
    return success(request, {"status": UserStatus.PENDING_APPROVAL.value})


@router.post("/email-verification/resend", status_code=status.HTTP_202_ACCEPTED)
async def resend_verification(
    payload: EmailRequest,
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    email = normalize_email(str(payload.email))
    await _public_limit(request, sessions, "verify", email)
    user = await db.scalar(select(User).where(User.email == email))
    if user is not None and user.status == UserStatus.PENDING_EMAIL:
        token = await issue_auth_token(
            db, user.id, TokenPurpose.EMAIL_VERIFICATION,
            timedelta(minutes=request.app.state.settings.email_verification_ttl_minutes),
        )
        await db.commit()
        background.add_task(request.app.state.email_service.verification, email, token)
    return success(request, {"message": "If verification is available, an email has been sent."})


@router.post("/login")
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    email = normalize_email(str(payload.email))
    settings = request.app.state.settings
    login_bucket = f"login:{_client_key(request)}:{token_digest(email)[:16]}"
    allowed = await sessions.rate_limit(
        login_bucket,
        settings.login_rate_limit,
        settings.login_rate_window_seconds,
    )
    if not allowed:
        raise ApiProblem(429, "RATE_LIMITED", "Too many login attempts", "Try again later.")
    user = await db.scalar(select(User).where(User.email == email))
    valid = False
    invitation = None
    if user is not None and user.status == UserStatus.INVITED:
        invitation = await db.scalar(
            select(Invitation).where(
                Invitation.user_id == user.id,
                Invitation.consumed_at.is_(None),
                Invitation.expires_at > datetime.now(UTC),
            ).order_by(Invitation.created_at.desc())
        )
        valid = invitation is not None and verify_password(
            payload.password, invitation.temporary_password_hash
        )
    elif user is not None:
        valid = verify_password(payload.password, user.password_hash)
    if user is None or not valid:
        audit(db, request, "auth.login", "FAILURE", target_id=user.id if user else None)
        await db.commit()
        raise ApiProblem(401, "INVALID_CREDENTIALS", "Sign in failed", "Email or password is incorrect.")
    await sessions.reset_rate_limit(login_bucket)
    if user.status in {
        UserStatus.PENDING_EMAIL, UserStatus.PENDING_APPROVAL,
        UserStatus.REJECTED, UserStatus.DISABLED,
    }:
        audit(db, request, "auth.login.blocked", "FAILURE", target_id=user.id, details={"status": user.status.value})
        await db.commit()
        raise ApiProblem(403, user.status.value, "Account unavailable", "This account is not active.")
    if invitation is not None:
        invitation.consumed_at = datetime.now(UTC)
        next_state = SessionState.PASSWORD_CHANGE_REQUIRED
    elif user.role == UserRole.ADMIN and (
        user.mfa_factor is None or user.mfa_factor.confirmed_at is None
    ):
        next_state = SessionState.MFA_ENROLLMENT_REQUIRED
    elif user.role == UserRole.ADMIN:
        next_state = SessionState.MFA_REQUIRED
    else:
        next_state = SessionState.AUTHENTICATED
    record = await sessions.create(user.id, next_state)
    set_session_cookie(request, response, record)
    user.last_login_at = datetime.now(UTC)
    audit(db, request, "auth.login", "SUCCESS", target_id=user.id, details={"next_step": next_state.value})
    await db.commit()
    data: dict[str, object] = {"next_step": next_state.value, "csrf_token": record.csrf_token}
    if next_state == SessionState.AUTHENTICATED:
        data["user"] = user_view(user)
    return success(request, data)


@router.get("/session")
async def current_session(request: Request, identity: Identity = Depends(get_identity)) -> dict:
    return success(request, {
        "next_step": identity.session.state,
        "csrf_token": identity.session.csrf_token,
        "user": user_view(identity.user)
        if identity.session.state == SessionState.AUTHENTICATED.value else None,
    })


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    identity: Identity = Depends(require_csrf),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    await sessions.delete(identity.session)
    clear_session_cookie(request, response)
    return success(request, {"signed_out": True})


@router.post("/password/change")
async def change_password(
    payload: PasswordChangeRequest,
    request: Request,
    response: Response,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    user = identity.user
    if identity.session.state == SessionState.AUTHENTICATED.value:
        if not payload.current_password or not verify_password(payload.current_password, user.password_hash):
            raise ApiProblem(401, "INVALID_CREDENTIALS", "Password not changed", "Current password is incorrect.")
    elif identity.session.state != SessionState.PASSWORD_CHANGE_REQUIRED.value:
        raise ApiProblem(403, "PASSWORD_CHANGE_NOT_ALLOWED", "Password change unavailable", "Complete the current authentication step first.")
    try:
        validate_password(
            payload.new_password,
            email=user.email,
            name=user.profile.full_name if user.profile else "",
        )
    except ValueError as exc:
        raise ApiProblem(422, "WEAK_PASSWORD", "Password rejected", str(exc)) from exc
    user.password_hash = hash_password(payload.new_password)
    if user.status == UserStatus.INVITED:
        user.status = UserStatus.ACTIVE
        user.email_verified_at = datetime.now(UTC)
    await sessions.revoke_user(user.id, except_id=identity.session.session_id)
    next_state = SessionState.MFA_REQUIRED if user.role == UserRole.ADMIN else SessionState.AUTHENTICATED
    rotated = await rotate_session(request, response, sessions, identity.session, next_state)
    audit(db, request, "auth.password.changed", "SUCCESS", actor_id=user.id, target_id=user.id)
    await db.commit()
    return success(request, {
        "next_step": next_state.value,
        "csrf_token": rotated.csrf_token,
        "user": user_view(user) if next_state == SessionState.AUTHENTICATED else None,
    })


@router.post("/password/forgot", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    payload: EmailRequest,
    request: Request,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    email = normalize_email(str(payload.email))
    await _public_limit(request, sessions, "forgot", email)
    user = await db.scalar(select(User).where(User.email == email))
    if user is not None and user.status == UserStatus.ACTIVE and user.password_hash:
        token = await issue_auth_token(
            db, user.id, TokenPurpose.PASSWORD_RESET,
            timedelta(minutes=request.app.state.settings.password_reset_ttl_minutes),
        )
        audit(db, request, "auth.password.reset.requested", "SUCCESS", target_id=user.id)
        await db.commit()
        background.add_task(request.app.state.email_service.password_reset, email, token)
    return success(request, {"message": "If the account is eligible, a reset email has been sent."})


@router.post("/password/reset")
async def reset_password(
    payload: PasswordResetRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    now = datetime.now(UTC)
    token = await db.scalar(select(AuthToken).where(
        AuthToken.token_hash == token_digest(payload.token),
        AuthToken.purpose == TokenPurpose.PASSWORD_RESET,
        AuthToken.used_at.is_(None), AuthToken.expires_at > now,
    ))
    if token is None:
        raise ApiProblem(400, "TOKEN_INVALID", "Invalid reset link", "This link is invalid or expired.")
    user = await db.get(User, token.user_id)
    if user is None or user.status != UserStatus.ACTIVE:
        raise ApiProblem(400, "TOKEN_INVALID", "Invalid reset link", "This link is invalid or expired.")
    try:
        validate_password(
            payload.new_password,
            email=user.email,
            name=user.profile.full_name if user.profile else "",
        )
    except ValueError as exc:
        raise ApiProblem(422, "WEAK_PASSWORD", "Password rejected", str(exc)) from exc
    token.used_at = now
    user.password_hash = hash_password(payload.new_password)
    await sessions.revoke_user(user.id)
    audit(db, request, "auth.password.reset", "SUCCESS", target_id=user.id)
    await db.commit()
    return success(request, {"password_reset": True})


@router.post("/mfa/enrollment")
async def begin_mfa_enrollment(
    request: Request,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if (
        identity.user.role != UserRole.ADMIN
        or identity.session.state != SessionState.MFA_ENROLLMENT_REQUIRED.value
    ):
        raise ApiProblem(403, "MFA_ENROLLMENT_NOT_ALLOWED", "MFA enrollment unavailable", "This session cannot enroll MFA.")
    secret = new_totp_secret()
    factor = await db.get(MfaFactor, identity.user.id)
    if factor is None:
        factor = MfaFactor(user_id=identity.user.id, secret_encrypted=encrypt_secret(secret))
        db.add(factor)
    else:
        factor.secret_encrypted = encrypt_secret(secret)
        factor.confirmed_at = None
    await db.commit()
    uri = pyotp.TOTP(secret).provisioning_uri(name=identity.user.email, issuer_name="Umbrella")
    return success(request, {"secret": secret, "provisioning_uri": uri})


@router.post("/mfa/enrollment/confirm")
async def confirm_mfa_enrollment(
    payload: MfaCodeRequest,
    request: Request,
    response: Response,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    factor = await db.get(MfaFactor, identity.user.id)
    if identity.session.state != SessionState.MFA_ENROLLMENT_REQUIRED.value or factor is None:
        raise ApiProblem(403, "MFA_ENROLLMENT_NOT_ALLOWED", "MFA enrollment unavailable", "Start enrollment first.")
    if not verify_totp(decrypt_secret(factor.secret_encrypted), payload.code):
        raise ApiProblem(401, "MFA_INVALID", "Invalid authentication code", "Check your authenticator and try again.")
    factor.confirmed_at = datetime.now(UTC)
    await db.execute(delete(MfaRecoveryCode).where(MfaRecoveryCode.user_id == identity.user.id))
    codes = recovery_codes()
    for code in codes:
        db.add(MfaRecoveryCode(user_id=identity.user.id, code_hash=token_digest(code)))
    rotated = await rotate_session(
        request, response, sessions, identity.session, SessionState.AUTHENTICATED
    )
    audit(db, request, "auth.mfa.enrolled", "SUCCESS", actor_id=identity.user.id, target_id=identity.user.id)
    await db.commit()
    return success(request, {
        "next_step": SessionState.AUTHENTICATED.value,
        "csrf_token": rotated.csrf_token,
        "recovery_codes": codes,
        "user": user_view(identity.user),
    })


@router.post("/mfa/verify")
async def mfa_verify(
    payload: MfaCodeRequest,
    request: Request,
    response: Response,
    identity: Identity = Depends(require_csrf),
    db: AsyncSession = Depends(get_db),
    sessions: SessionStore = Depends(get_session_store),
) -> dict:
    if (
        identity.session.state != SessionState.MFA_REQUIRED.value
        or identity.user.role != UserRole.ADMIN
    ):
        raise ApiProblem(403, "MFA_NOT_REQUIRED", "MFA verification unavailable", "This session does not require MFA.")
    factor = await db.get(MfaFactor, identity.user.id)
    valid = False
    if factor is not None and factor.confirmed_at is not None and payload.method == "totp":
        valid = verify_totp(decrypt_secret(factor.secret_encrypted), payload.code)
    elif payload.method == "recovery_code":
        code = await db.scalar(select(MfaRecoveryCode).where(
            MfaRecoveryCode.user_id == identity.user.id,
            MfaRecoveryCode.code_hash == token_digest(payload.code),
            MfaRecoveryCode.used_at.is_(None),
        ))
        if code is not None:
            code.used_at = datetime.now(UTC)
            valid = True
    if not valid:
        audit(db, request, "auth.mfa.verified", "FAILURE", target_id=identity.user.id)
        await db.commit()
        raise ApiProblem(401, "MFA_INVALID", "Invalid authentication code", "Check the code and try again.")
    rotated = await rotate_session(
        request, response, sessions, identity.session, SessionState.AUTHENTICATED
    )
    audit(db, request, "auth.mfa.verified", "SUCCESS", actor_id=identity.user.id, target_id=identity.user.id)
    await db.commit()
    return success(request, {
        "next_step": SessionState.AUTHENTICATED.value,
        "csrf_token": rotated.csrf_token,
        "user": user_view(identity.user),
    })
