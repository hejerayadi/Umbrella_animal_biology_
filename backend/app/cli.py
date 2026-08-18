from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from datetime import UTC, datetime

from sqlalchemy import delete, select

from .core.config import get_settings
from .core.security import hash_password, normalize_email, validate_password
from .db.session import create_database
from .models import MfaFactor, MfaRecoveryCode, User, UserRole, UserStatus
from .services.audit import purge_old_audit_events


def configure_event_loop_policy() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def create_admin(email_arg: str | None) -> None:
    email = normalize_email(email_arg or input("Admin email: "))
    password = getpass.getpass("Admin password: ")
    if password != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords do not match.")
    validate_password(password, email=email)
    engine, factory = create_database(get_settings().database_url)
    async with factory() as db:
        existing = await db.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise SystemExit("That email already exists. Use promote-admin instead.")
        db.add(User(
            email=email,
            password_hash=hash_password(password),
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            email_verified_at=datetime.now(UTC),
        ))
        await db.commit()
    await engine.dispose()
    print(f"Created administrator {email}. MFA enrollment is required at first login.")


async def promote_admin(email_arg: str) -> None:
    email = normalize_email(email_arg)
    engine, factory = create_database(get_settings().database_url)
    async with factory() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if user is None or user.status != UserStatus.ACTIVE:
            raise SystemExit("Only an active existing user can be promoted.")
        user.role = UserRole.ADMIN
        await db.commit()
    await engine.dispose()
    print(f"Promoted {email}. Revoke existing sessions from the admin dashboard.")


async def reset_mfa(email_arg: str) -> None:
    email = normalize_email(email_arg)
    engine, factory = create_database(get_settings().database_url)
    async with factory() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if user is None or user.role != UserRole.ADMIN:
            raise SystemExit("Administrator not found.")
        await db.execute(delete(MfaRecoveryCode).where(MfaRecoveryCode.user_id == user.id))
        await db.execute(delete(MfaFactor).where(MfaFactor.user_id == user.id))
        await db.commit()
    await engine.dispose()
    print(f"Reset MFA for {email}. Re-enrollment is required at next login.")


async def purge_audit() -> None:
    settings = get_settings()
    engine, factory = create_database(settings.database_url)
    async with factory() as db:
        await purge_old_audit_events(db, settings.audit_retention_days)
    await engine.dispose()
    print("Expired audit events removed.")


def main() -> None:
    configure_event_loop_policy()
    parser = argparse.ArgumentParser(description="Umbrella user-management administration")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-admin")
    create.add_argument("--email")
    promote = commands.add_parser("promote-admin")
    promote.add_argument("email")
    reset = commands.add_parser("reset-mfa")
    reset.add_argument("email")
    commands.add_parser("purge-audit")
    args = parser.parse_args()
    if args.command == "create-admin":
        asyncio.run(create_admin(args.email))
    elif args.command == "promote-admin":
        asyncio.run(promote_admin(args.email))
    elif args.command == "reset-mfa":
        asyncio.run(reset_mfa(args.email))
    else:
        asyncio.run(purge_audit())


if __name__ == "__main__":
    main()
