from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pyotp
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.image_store import IMAGE_STORE
from backend.app.core.config import get_settings
from backend.app.core.security import hash_password
from backend.app.main import create_app
from backend.app.models import User, UserRole, UserStatus


class RecordingEmail:
    verification_token: str | None = None
    invitation_password: str | None = None
    application_recipient: str | None = None

    def verification(self, recipient: str, token: str) -> None:
        self.verification_token = token

    def invitation(self, recipient: str, password: str) -> None:
        self.invitation_password = password

    def password_reset(self, recipient: str, token: str) -> None:
        pass

    def new_biologist_application(self, recipient: str, **_: object) -> None:
        self.application_recipient = recipient

    def send(self, recipient: str, subject: str, text: str, html: str | None = None) -> None:
        pass


def _create_admin(email: str, password: str) -> None:
    engine = create_engine(get_settings().database_url)
    with Session(engine) as db:
        if db.scalar(select(User).where(User.email == email)) is None:
            db.add(User(
                email=email,
                password_hash=hash_password(password),
                role=UserRole.ADMIN,
                status=UserStatus.ACTIVE,
                email_verified_at=datetime.now(UTC),
            ))
            db.commit()
    engine.dispose()


def _data(response):
    body = response.json()
    assert body["error"] is None, body
    return body["data"]


def test_registration_approval_mfa_and_invitation_flow() -> None:
    suffix = uuid.uuid4().hex[:10]
    admin_email = f"admin-{suffix}@example.org"
    candidate_email = f"biologist-{suffix}@example.org"
    invited_email = f"invited-{suffix}@example.org"
    admin_password = "Admin-Strong-Password-47!"
    candidate_password = "Candidate-Strong-Password-82!"
    _create_admin(admin_email, admin_password)

    app = create_app()
    with TestClient(app) as admin_client:
        email = RecordingEmail()
        app.state.email_service = email

        login = _data(admin_client.post("/api/v1/auth/login", json={
            "email": admin_email, "password": admin_password,
        }))
        assert login["next_step"] == "MFA_ENROLLMENT_REQUIRED"
        csrf = login["csrf_token"]
        enrollment = _data(admin_client.post(
            "/api/v1/auth/mfa/enrollment", headers={"X-CSRF-Token": csrf}
        ))
        code = pyotp.TOTP(enrollment["secret"]).now()
        confirmed = _data(admin_client.post(
            "/api/v1/auth/mfa/enrollment/confirm",
            headers={"X-CSRF-Token": csrf}, json={"code": code, "method": "totp"},
        ))
        assert len(confirmed["recovery_codes"]) == 10
        csrf = confirmed["csrf_token"]

        registration_payload = {
            "email": candidate_email,
            "password": candidate_password,
            "full_name": "Nadia Researcher",
            "institution": "Marine Biology Institute",
            "professional_title": "Senior Biologist",
            "country": "TN",
            "motivation": "I study biodiversity change across protected coastal ecosystems.",
            "specialties": ["Biodiversity Analysis"],
        }
        rejected_registration = admin_client.post(
            "/api/v1/auth/register",
            json={**registration_payload, "password": "Secure-Nadia-Research-47!"},
        )
        assert rejected_registration.status_code == 422
        rejected_body = rejected_registration.json()
        assert rejected_body["error"]["code"] == "WEAK_PASSWORD"
        assert "name or email identifier" in rejected_body["error"]["detail"]

        registration = admin_client.post("/api/v1/auth/register", json=registration_payload)
        assert registration.status_code == 202
        assert email.verification_token

        # Re-registering an address that was never verified must re-issue the
        # link. Doing nothing would strand an applicant whose first email was
        # lost, with no way to ever finish signing up.
        first_token = email.verification_token
        email.verification_token = None
        repeat = admin_client.post("/api/v1/auth/register", json=registration_payload)
        assert repeat.status_code == 202
        assert email.verification_token and email.verification_token != first_token

        verified = _data(admin_client.post(
            "/api/v1/auth/email-verification/confirm",
            json={"token": email.verification_token},
        ))
        assert verified["status"] == "PENDING_APPROVAL"
        assert email.application_recipient == admin_email
        rows = _data(admin_client.get("/api/v1/admin/applications"))
        candidate = next(item for item in rows if item["email"] == candidate_email)
        approved = admin_client.post(
            f"/api/v1/admin/applications/{candidate['id']}/approve",
            headers={"X-CSRF-Token": csrf},
        )
        assert approved.status_code == 200

        invited = _data(admin_client.post(
            "/api/v1/admin/invitations",
            headers={"X-CSRF-Token": csrf},
            json={
                "email": invited_email,
                "full_name": "Invited Biologist",
                "institution": "Genome Lab",
                "professional_title": "Biologist",
                "country": "FR",
                "specialties": ["Genome Reconstruction"],
            },
        ))
        assert invited["status"] == "INVITED"
        assert email.invitation_password

        with TestClient(create_app()) as invited_client:
            invited_login = _data(invited_client.post("/api/v1/auth/login", json={
                "email": invited_email, "password": email.invitation_password,
            }))
            assert invited_login["next_step"] == "PASSWORD_CHANGE_REQUIRED"
            blocked = invited_client.post(
                "/api/v1/chat",
                headers={"X-CSRF-Token": invited_login["csrf_token"]},
                json={"query": "hello", "context": {}},
            )
            assert blocked.status_code == 403
            changed = _data(invited_client.post(
                "/api/v1/auth/password/change",
                headers={"X-CSRF-Token": invited_login["csrf_token"]},
                json={"new_password": "Permanent-Secure-Password-53!"},
            ))
            assert changed["next_step"] == "AUTHENTICATED"
            assert invited_client.get("/api/v1/health").status_code == 200

        with TestClient(create_app()) as candidate_client:
            signed_in = _data(candidate_client.post("/api/v1/auth/login", json={
                "email": candidate_email, "password": candidate_password,
            }))
            assert signed_in["next_step"] == "AUTHENTICATED"
            assert candidate_client.get("/api/v1/health").status_code == 200

        all_users = _data(admin_client.get("/api/v1/admin/users?page_size=100"))
        current_admin = next(item for item in all_users if item["email"] == admin_email)
        self_delete = admin_client.delete(
            f"/api/v1/admin/users/{current_admin['id']}",
            headers={"X-CSRF-Token": csrf},
        )
        assert self_delete.status_code == 409
        assert self_delete.json()["error"]["code"] == "SELF_DELETE_FORBIDDEN"

        stored = IMAGE_STORE.add(
            b"\x89PNG\r\n\x1a\npreview",
            "preview.png",
            owner_id=candidate["id"],
        )
        deleted = _data(admin_client.delete(
            f"/api/v1/admin/users/{candidate['id']}",
            headers={"X-CSRF-Token": csrf},
        ))
        assert deleted == {
            "deleted": True,
            "user_id": candidate["id"],
            "removed_images": 1,
        }
        assert IMAGE_STORE.get(stored.image_id) is None
        remaining_users = _data(admin_client.get("/api/v1/admin/users?page_size=100"))
        assert not any(item["id"] == candidate["id"] for item in remaining_users)
        deletion_events = _data(admin_client.get(
            "/api/v1/admin/audit-events?event_type=admin.user.deleted"
        ))
        assert any(event["details"]["role"] == "BIOLOGIST" for event in deletion_events)

        deleted_login = admin_client.post("/api/v1/auth/login", json={
            "email": candidate_email,
            "password": candidate_password,
        })
        assert deleted_login.status_code == 401

        stats = _data(admin_client.get("/api/v1/admin/stats"))
        assert stats["total_users"] >= 2
        assert stats["active_biologists"] >= 1
        assert "status_counts" in stats

        disabled_invited = admin_client.post(
            f"/api/v1/admin/users/{invited['id']}/disable",
            headers={"X-CSRF-Token": csrf},
        )
        assert disabled_invited.status_code == 200
        reactivated = _data(admin_client.post(
            "/api/v1/admin/bulk-users/reactivate",
            headers={"X-CSRF-Token": csrf},
            json={"user_ids": [invited["id"]]},
        ))
        assert reactivated["reactivated"] == 1

        bulk_deleted = _data(admin_client.post(
            "/api/v1/admin/bulk-users/delete",
            headers={"X-CSRF-Token": csrf},
            json={"user_ids": [invited["id"]]},
        ))
        assert bulk_deleted["deleted"] == 1
        assert bulk_deleted["user_ids"] == [invited["id"]]

        read_all = _data(admin_client.post(
            "/api/v1/admin/notifications/read-all",
            headers={"X-CSRF-Token": csrf},
        ))
        assert read_all["read"] >= 1
