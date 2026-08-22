from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from .models import User, UserRole, UserStatus


class RegistrationRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    full_name: str = Field(min_length=2, max_length=160)
    institution: str = Field(min_length=2, max_length=240)
    professional_title: str = Field(min_length=2, max_length=160)
    country: str = Field(min_length=2, max_length=2)
    orcid: str | None = Field(default=None, pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
    motivation: str = Field(min_length=20, max_length=4000)
    specialties: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, value: str) -> str:
        return value.upper()


class EmailRequest(BaseModel):
    email: EmailStr


class TokenRequest(BaseModel):
    token: str = Field(min_length=20, max_length=512)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class MfaCodeRequest(BaseModel):
    code: str = Field(min_length=6, max_length=32)
    method: Literal["totp", "recovery_code"] = "totp"


class PasswordChangeRequest(BaseModel):
    new_password: str = Field(min_length=12, max_length=128)
    current_password: str | None = None


class PasswordResetRequest(BaseModel):
    token: str = Field(min_length=20, max_length=512)
    new_password: str = Field(min_length=12, max_length=128)


class ProfileUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=160)
    institution: str | None = Field(default=None, min_length=2, max_length=240)
    professional_title: str | None = Field(default=None, min_length=2, max_length=160)
    country: str | None = Field(default=None, min_length=2, max_length=2)
    orcid: str | None = Field(default=None, pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
    motivation: str | None = Field(default=None, min_length=20, max_length=4000)
    specialties: list[str] | None = Field(default=None, max_length=20)


class RejectRequest(BaseModel):
    reason: str = Field(min_length=5, max_length=2000)


class BulkUserRequest(BaseModel):
    user_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)


class InvitationRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=160)
    institution: str = Field(min_length=2, max_length=240)
    professional_title: str = Field(min_length=2, max_length=160)
    country: str = Field(min_length=2, max_length=2)
    orcid: str | None = Field(default=None, pattern=r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
    specialties: list[str] = Field(default_factory=list, max_length=20)


class UserView(BaseModel):
    id: uuid.UUID
    email: str
    role: UserRole
    status: UserStatus
    email_verified_at: datetime | None
    last_login_at: datetime | None
    created_at: datetime
    full_name: str
    institution: str
    professional_title: str
    country: str
    orcid: str | None
    motivation: str
    specialties: list[str]


def user_view(user: User) -> dict[str, object]:
    profile = user.profile
    return UserView(
        id=user.id,
        email=user.email,
        role=user.role,
        status=user.status,
        email_verified_at=user.email_verified_at,
        last_login_at=user.last_login_at,
        created_at=user.created_at,
        full_name=profile.full_name if profile else user.email.split("@", 1)[0],
        institution=profile.institution if profile else "",
        professional_title=profile.professional_title if profile else "",
        country=profile.country if profile else "",
        orcid=profile.orcid if profile else None,
        motivation=profile.motivation if profile else "",
        specialties=profile.specialties if profile else [],
    ).model_dump(mode="json")
