from cryptography.fernet import Fernet

from backend.app.core.config import get_settings
from backend.app.core.security import (
    decrypt_secret,
    encrypt_secret,
    hash_password,
    token_digest,
    validate_password,
    verify_password,
)


def test_passwords_are_argon2_hashed_and_verified() -> None:
    encoded = hash_password("Correct-Horse-47-Battery")
    assert encoded.startswith("$argon2")
    assert verify_password("Correct-Horse-47-Battery", encoded)
    assert not verify_password("wrong", encoded)


def test_password_policy_rejects_personal_data() -> None:
    try:
        validate_password("amira-super-secure", email="amira@example.org", name="Amira Doe")
    except ValueError as exc:
        assert "name or email identifier" in str(exc)
    else:
        raise AssertionError("personal password should be rejected")


def test_password_policy_normalizes_accents_case_and_separators() -> None:
    rejected = (
        ("Secure-Elodie-2026!", "researcher@example.org", "\u00c9lodie Martin"),
        ("Safe-Alex-Morgan-47!", "alex.morgan@example.org", "Research User"),
        ("Strong-JEAN-LUC-92!", "captain@example.org", "Jean-Luc Picard"),
    )
    for password, email, name in rejected:
        try:
            validate_password(password, email=email, name=name)
        except ValueError as exc:
            assert "name or email identifier" in str(exc)
        else:
            raise AssertionError(f"personal identifier should be rejected for {email}")


def test_password_policy_avoids_short_identifier_false_positives() -> None:
    validate_password(
        "Major-Research-Result-47!",
        email="ma@example.org",
        name="Li Wu",
    )


def test_token_digest_is_stable_and_secret_encryption_round_trips(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "auth_encryption_key", Fernet.generate_key().decode())
    assert token_digest("one-time") == token_digest("one-time")
    encrypted = encrypt_secret("TOTPSECRET")
    assert encrypted != "TOTPSECRET"
    assert decrypt_secret(encrypted) == "TOTPSECRET"
