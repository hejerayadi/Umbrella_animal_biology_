from __future__ import annotations

import hashlib
import re
import secrets
import string
import unicodedata

import pyotp
from cryptography.fernet import Fernet
from pwdlib import PasswordHash

from .config import get_settings

password_hasher = PasswordHash.recommended()

COMMON_PASSWORDS = {
    "123456789012",
    "admin123456",
    "letmein123456",
    "password",
    "password123",
    "password1234",
    "qwerty123456",
    "umbrella1234",
    "welcome123456",
}

PERSONAL_IDENTIFIER_MIN_LENGTH = 4


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def _comparison_key(value: str) -> str:
    """Create a case/accent/separator-insensitive key for policy comparisons."""
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in normalized if character.isalnum())


def _personal_identifiers(*, email: str, name: str) -> set[str]:
    email_identifier = normalize_email(email).partition("@")[0]
    candidates = [
        email_identifier,
        *re.split(r"[\W_]+", email_identifier, flags=re.UNICODE),
        name,
        *re.split(r"[\W_]+", name, flags=re.UNICODE),
    ]
    # Very short fragments (for example "li" or "ma") would match many
    # unrelated passwords. Full identifiers use the same explicit minimum.
    return {
        key
        for candidate in candidates
        if len(key := _comparison_key(candidate)) >= PERSONAL_IDENTIFIER_MIN_LENGTH
    }


def validate_password(password: str, *, email: str = "", name: str = "") -> None:
    if not 12 <= len(password) <= 128:
        raise ValueError("Password must contain between 12 and 128 characters.")
    if password.strip().casefold() in COMMON_PASSWORDS:
        raise ValueError("Choose a less common password.")
    password_key = _comparison_key(password)
    if any(identifier in password_key for identifier in _personal_identifiers(email=email, name=name)):
        raise ValueError("Password must not contain your name or email identifier.")


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        return password_hasher.verify(password, encoded)
    except Exception:
        return False


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def temporary_password() -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(20))
        if any(c.islower() for c in password) and any(c.isupper() for c in password) and any(
            c.isdigit() for c in password
        ):
            return password


def encrypt_secret(secret: str) -> str:
    return Fernet(get_settings().auth_encryption_key.encode("ascii")).encrypt(
        secret.encode("ascii")
    ).decode("ascii")


def decrypt_secret(ciphertext: str) -> str:
    return Fernet(get_settings().auth_encryption_key.encode("ascii")).decrypt(
        ciphertext.encode("ascii")
    ).decode("ascii")


def new_totp_secret() -> str:
    return pyotp.random_base32()


def verify_totp(secret: str, code: str) -> bool:
    return pyotp.TOTP(secret).verify(code.strip(), valid_window=1)


def recovery_codes() -> list[str]:
    return [f"{secrets.token_hex(4)}-{secrets.token_hex(4)}" for _ in range(10)]
