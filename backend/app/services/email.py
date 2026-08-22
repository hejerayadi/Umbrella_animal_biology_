from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from ..core.config import Settings
from ..email.templates import new_biologist_application_email, verification_email

logger = logging.getLogger("umbrella.email")

# Implicit-TLS submission port: the socket is wrapped in TLS from the first byte
# instead of being upgraded later with STARTTLS.
IMPLICIT_TLS_PORT = 465


class EmailService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _connect(self) -> smtplib.SMTP:
        if self.settings.smtp_port == IMPLICIT_TLS_PORT:
            return smtplib.SMTP_SSL(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=15,
                context=ssl.create_default_context(),
            )
        return smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=15)

    def send(self, recipient: str, subject: str, text: str, html: str | None = None) -> None:
        message = EmailMessage()
        message["From"] = f"{self.settings.smtp_from_name} <{self.settings.smtp_from_email}>"
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(text)
        if html:
            message.add_alternative(html, subtype="html")
        try:
            with self._connect() as smtp:
                smtp.ehlo()
                needs_auth = bool(self.settings.smtp_username)
                # Credentials must never cross the wire in cleartext, so upgrade
                # whenever we intend to authenticate and the server offers STARTTLS -
                # even if SMTP_STARTTLS was left off by mistake. On port 465 the
                # connection is already encrypted and STARTTLS is not advertised.
                if smtp.has_extn("starttls") and (self.settings.smtp_starttls or needs_auth):
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()
                if needs_auth:
                    if not smtp.has_extn("auth"):
                        raise RuntimeError(
                            f"{self.settings.smtp_host}:{self.settings.smtp_port} does not offer "
                            "SMTP AUTH on an encrypted channel. Use port 587 with STARTTLS or "
                            "port 465 for implicit TLS, and verify SMTP_HOST/SMTP_PORT."
                        )
                    smtp.login(self.settings.smtp_username, self.settings.smtp_password)
                smtp.send_message(message)
        except Exception:
            # Every caller dispatches this through BackgroundTasks, so the HTTP
            # response has already been sent and there is nobody left to inform.
            # The failure is logged here with recipient context instead.
            logger.exception(
                "Email delivery failed", extra={"recipient": recipient, "subject": subject}
            )

    def verification(self, recipient: str, token: str) -> None:
        url = f"{self.settings.frontend_url.rstrip('/')}/verify-email?token={token}"
        text, html = verification_email(
            recipient, url, self.settings.email_verification_ttl_minutes
        )
        self.send(recipient, "Verify your Umbrella email", text, html)

    def new_biologist_application(
        self,
        recipient: str,
        *,
        applicant_email: str,
        full_name: str,
        institution: str,
        professional_title: str,
        country: str,
        orcid: str | None,
        motivation: str,
        specialties: list[str],
    ) -> None:
        review_url = f"{self.settings.frontend_url.rstrip('/')}/admin"
        text, html = new_biologist_application_email(
            applicant_email=applicant_email,
            full_name=full_name,
            institution=institution,
            professional_title=professional_title,
            country=country,
            orcid=orcid,
            motivation=motivation,
            specialties=specialties,
            review_url=review_url,
        )
        self.send(recipient, "New Umbrella biologist application", text, html)

    def password_reset(self, recipient: str, token: str) -> None:
        url = f"{self.settings.frontend_url.rstrip('/')}/reset-password?token={token}"
        self.send(recipient, "Reset your Umbrella password", f"Reset your password within 30 minutes: {url}")

    def invitation(self, recipient: str, password: str) -> None:
        url = f"{self.settings.frontend_url.rstrip('/')}/signin"
        self.send(
            recipient,
            "Your Umbrella invitation",
            f"Sign in at {url} with this one-time password (expires in 72 hours): {password}",
        )
