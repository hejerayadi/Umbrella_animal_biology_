from __future__ import annotations

import logging
from email.message import EmailMessage

from backend.app.core.config import Settings
from backend.app.services.email import EmailService


class RecordingSmtp:
    """Stand-in for smtplib.SMTP / smtplib.SMTP_SSL that records the call order."""

    extensions: set[str] = {"starttls", "auth"}
    calls: list[object] = []
    message: EmailMessage | None = None

    def __init__(self, host: str, port: int, timeout: int, context: object = None) -> None:
        self.calls.append(("connect", host, port, timeout))

    def __enter__(self) -> RecordingSmtp:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def ehlo(self) -> None:
        self.calls.append("ehlo")

    def starttls(self, context: object = None) -> None:
        self.calls.append("starttls")

    def has_extn(self, extension: str) -> bool:
        self.calls.append(("has_extn", extension))
        return extension in self.extensions

    def login(self, username: str, password: str) -> None:
        self.calls.append(("login", username, password))

    def send_message(self, message: EmailMessage) -> None:
        type(self).message = message
        self.calls.append(("send", message["To"], message["Subject"]))


def _reset(extensions: set[str]) -> None:
    RecordingSmtp.calls = []
    RecordingSmtp.extensions = extensions
    RecordingSmtp.message = None


def _settings(**updates: object) -> Settings:
    values: dict[str, object] = {
        "smtp_host": "smtp.example.org",
        "smtp_port": 587,
        "smtp_username": "sender@example.org",
        "smtp_password": "app-password",
        "smtp_from_email": "sender@example.org",
        "smtp_starttls": True,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_email_service_negotiates_tls_before_authentication(monkeypatch) -> None:
    _reset({"starttls", "auth"})
    monkeypatch.setattr("backend.app.services.email.smtplib.SMTP", RecordingSmtp)

    EmailService(_settings()).send("recipient@example.org", "Subject", "Body")

    assert RecordingSmtp.calls == [
        ("connect", "smtp.example.org", 587, 15),
        "ehlo",
        ("has_extn", "starttls"),
        "starttls",
        "ehlo",
        ("has_extn", "auth"),
        ("login", "sender@example.org", "app-password"),
        ("send", "recipient@example.org", "Subject"),
    ]


def test_email_service_upgrades_to_tls_even_when_starttls_is_disabled(monkeypatch) -> None:
    """Credentials must never be sent over a cleartext channel, so an operator
    leaving SMTP_STARTTLS off must not downgrade an authenticated session."""
    _reset({"starttls", "auth"})
    monkeypatch.setattr("backend.app.services.email.smtplib.SMTP", RecordingSmtp)

    EmailService(_settings(smtp_starttls=False)).send("recipient@example.org", "Subject", "Body")

    assert "starttls" in RecordingSmtp.calls
    starttls_index = RecordingSmtp.calls.index("starttls")
    login_index = next(
        index
        for index, call in enumerate(RecordingSmtp.calls)
        if isinstance(call, tuple) and call[0] == "login"
    )
    assert starttls_index < login_index


def test_email_service_uses_implicit_tls_on_port_465(monkeypatch) -> None:
    _reset({"auth"})
    monkeypatch.setattr("backend.app.services.email.smtplib.SMTP_SSL", RecordingSmtp)

    EmailService(_settings(smtp_port=465, smtp_starttls=False)).send(
        "recipient@example.org", "Subject", "Body"
    )

    assert ("connect", "smtp.example.org", 465, 15) in RecordingSmtp.calls
    assert "starttls" not in RecordingSmtp.calls
    assert ("login", "sender@example.org", "app-password") in RecordingSmtp.calls
    assert ("send", "recipient@example.org", "Subject") in RecordingSmtp.calls


def test_email_service_does_not_attempt_login_without_auth_extension(
    monkeypatch,
    caplog,
) -> None:
    _reset(set())
    monkeypatch.setattr("backend.app.services.email.smtplib.SMTP", RecordingSmtp)
    caplog.set_level(logging.ERROR, logger="umbrella.email")

    EmailService(_settings()).send("recipient@example.org", "Subject", "Body")

    assert not any(isinstance(call, tuple) and call[0] == "login" for call in RecordingSmtp.calls)
    assert not any(isinstance(call, tuple) and call[0] == "send" for call in RecordingSmtp.calls)
    record = next(record for record in caplog.records if record.name == "umbrella.email")
    assert record.exc_info is not None
    assert "does not offer SMTP AUTH" in str(record.exc_info[1])


def test_registration_verification_email_has_plain_text_and_premium_html(monkeypatch) -> None:
    _reset({"starttls", "auth"})
    monkeypatch.setattr("backend.app.services.email.smtplib.SMTP", RecordingSmtp)

    EmailService(_settings(frontend_url="https://umbrella.example")).verification(
        "biologist@example.org", "verification-token"
    )

    message = RecordingSmtp.message
    assert message is not None
    assert message["Subject"] == "Verify your Umbrella email"
    plain = message.get_body(preferencelist=("plain",))
    html = message.get_body(preferencelist=("html",))
    assert plain is not None and html is not None
    assert "expires in 30 minutes" in plain.get_content()
    assert "https://umbrella.example/verify-email?token=verification-token" in plain.get_content()
    html_content = html.get_content()
    assert "Umbrella Animal BioHub" in html_content
    assert "Verify email address" in html_content
    assert "Application &middot; Step 1 of 2" in html_content
    assert "#b53b2a" in html_content
