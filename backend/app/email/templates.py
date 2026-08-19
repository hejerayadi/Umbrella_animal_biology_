from __future__ import annotations

from html import escape


def new_biologist_application_email(
    *,
    applicant_email: str,
    full_name: str,
    institution: str,
    professional_title: str,
    country: str,
    orcid: str | None,
    motivation: str,
    specialties: list[str],
    review_url: str,
) -> tuple[str, str]:
    """Return a detailed, email-client-safe application alert for administrators."""

    display_name = full_name.strip() or applicant_email
    specialty_text = ", ".join(specialties) if specialties else "Not provided"
    orcid_text = orcid or "Not provided"
    text = (
        "New verified biologist application\n\n"
        "A candidate has verified their email and is waiting for administrator review.\n\n"
        f"Name: {display_name}\n"
        f"Email: {applicant_email}\n"
        f"Professional title: {professional_title or 'Not provided'}\n"
        f"Institution: {institution or 'Not provided'}\n"
        f"Country: {country or 'Not provided'}\n"
        f"ORCID: {orcid_text}\n"
        f"Specialties: {specialty_text}\n\n"
        "Motivation\n"
        f"{motivation or 'Not provided'}\n\n"
        f"Review this application: {review_url}\n\n"
        "This is an administrative notification from Umbrella Animal BioHub."
    )

    safe_name = escape(display_name)
    safe_email = escape(applicant_email)
    safe_title = escape(professional_title or "Not provided")
    safe_institution = escape(institution or "Not provided")
    safe_country = escape(country or "Not provided")
    safe_orcid = escape(orcid_text)
    safe_specialties = escape(specialty_text)
    safe_motivation = escape(motivation or "Not provided").replace("\n", "<br>")
    safe_review_url = escape(review_url, quote=True)

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light">
    <title>New biologist application</title>
  </head>
  <body style="margin:0;padding:0;background:#f7f3ef;color:#182126;font-family:'IBM Plex Sans',Arial,sans-serif;">
    <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">
      {safe_name} completed email verification and is ready for review.
    </div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f7f3ef;">
      <tr>
        <td align="center" style="padding:36px 16px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="max-width:640px;">
            <tr>
              <td style="padding:0 6px 22px;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0">
                  <tr>
                    <td width="44" height="44" align="center" style="width:44px;height:44px;border-radius:8px;background:#b53b2a;color:#fff;font-family:'Space Grotesk',Arial,sans-serif;font-size:22px;font-weight:700;">U</td>
                    <td style="padding-left:12px;">
                      <div style="font-family:'Space Grotesk',Arial,sans-serif;font-size:22px;line-height:24px;font-weight:700;color:#182126;">Umbrella</div>
                      <div style="padding-top:3px;font-size:12px;line-height:15px;font-weight:600;color:#b53b2a;">Animal BioHub · Administration</div>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr>
              <td style="overflow:hidden;border:1px solid #e6dcd5;border-radius:12px;background:#fff;box-shadow:0 18px 48px rgba(70,46,35,.10);">
                <div style="height:4px;background:#b53b2a;line-height:4px;font-size:4px;">&nbsp;</div>
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                  <tr>
                    <td style="padding:40px 44px 22px;">
                      <div style="font-size:11px;line-height:16px;font-weight:700;letter-spacing:1.4px;color:#b53b2a;text-transform:uppercase;">New application · Review required</div>
                      <h1 style="margin:14px 0 10px;font-family:'Space Grotesk',Arial,sans-serif;font-size:32px;line-height:39px;color:#182126;">{safe_name}</h1>
                      <p style="margin:0;font-size:15px;line-height:24px;color:#615a56;">A new biologist has completed email verification and is waiting for your decision.</p>
                      <div style="display:inline-block;margin-top:16px;padding:6px 10px;border-radius:999px;background:#edf7f0;color:#267044;font-size:12px;font-weight:700;">✓ Email verified</div>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:0 44px 28px;">
                      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;border:1px solid #eee5df;border-radius:8px;background:#fbf9f7;">
                        <tr><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#7b736f;font-size:12px;width:34%;">Email</td><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#2c2927;font-size:14px;font-weight:600;word-break:break-word;">{safe_email}</td></tr>
                        <tr><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#7b736f;font-size:12px;">Professional title</td><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#2c2927;font-size:14px;">{safe_title}</td></tr>
                        <tr><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#7b736f;font-size:12px;">Institution</td><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#2c2927;font-size:14px;">{safe_institution}</td></tr>
                        <tr><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#7b736f;font-size:12px;">Country / ORCID</td><td style="padding:14px 18px;border-bottom:1px solid #eee5df;color:#2c2927;font-size:14px;">{safe_country} · {safe_orcid}</td></tr>
                        <tr><td style="padding:14px 18px;color:#7b736f;font-size:12px;">Specialties</td><td style="padding:14px 18px;color:#2c2927;font-size:14px;">{safe_specialties}</td></tr>
                      </table>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:0 44px 30px;">
                      <div style="font-size:12px;font-weight:700;letter-spacing:.6px;color:#7b736f;text-transform:uppercase;">Motivation</div>
                      <div style="margin-top:9px;padding:16px 18px;border-left:3px solid #b53b2a;background:#fdfbf9;color:#4f4945;font-size:14px;line-height:22px;">{safe_motivation}</div>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:0 44px 40px;">
                      <table role="presentation" cellspacing="0" cellpadding="0" border="0">
                        <tr>
                          <td align="center" style="border-radius:8px;background:#b53b2a;box-shadow:0 9px 22px rgba(181,59,42,.22);">
                            <a href="{safe_review_url}" style="display:inline-block;padding:15px 24px;color:#fff;text-decoration:none;font-size:15px;font-weight:700;">Review application&nbsp;&nbsp;&rarr;</a>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:22px 44px 28px;border-top:1px solid #eee5df;background:#fdfbf9;color:#7b736f;font-size:12px;line-height:19px;">Review the candidate's profile before approving or rejecting the application. This notification was sent only to active Umbrella administrators.</td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr>
              <td style="padding:20px 8px 0;text-align:center;color:#918984;font-size:11px;line-height:18px;">Umbrella Animal BioHub · Administrative notification</td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""
    return text, html


def verification_email(
    recipient: str, verification_url: str, expires_in_minutes: int
) -> tuple[str, str]:
    """Return a plain-text and email-client-safe HTML registration message."""

    safe_recipient = escape(recipient)
    safe_url = escape(verification_url, quote=True)
    text = (
        "Welcome to Umbrella Animal BioHub.\n\n"
        "Verify your email address to continue your biologist application:\n"
        f"{verification_url}\n\n"
        f"This secure link expires in {expires_in_minutes} minutes. After verification, your application "
        "will be sent to an Umbrella administrator for review.\n\n"
        "If you did not create this account, you can safely ignore this email."
    )
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="light">
    <title>Verify your Umbrella email</title>
  </head>
  <body style="margin:0;padding:0;background:#f7f3ef;color:#182126;font-family:'IBM Plex Sans',Arial,sans-serif;">
    <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">
      Verify your email to continue your Umbrella biologist application.
    </div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#f7f3ef;">
      <tr>
        <td align="center" style="padding:36px 16px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="max-width:620px;">
            <tr>
              <td style="padding:0 6px 22px;">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0">
                  <tr>
                    <td width="44" height="44" align="center" valign="middle" style="width:44px;height:44px;border-radius:8px;background:#b53b2a;color:#ffffff;font-family:'Space Grotesk',Arial,sans-serif;font-size:22px;font-weight:700;">U</td>
                    <td style="padding-left:12px;">
                      <div style="font-family:'Space Grotesk',Arial,sans-serif;font-size:22px;line-height:24px;font-weight:700;color:#182126;">Umbrella</div>
                      <div style="padding-top:3px;font-size:12px;line-height:15px;font-weight:600;color:#b53b2a;">Animal BioHub</div>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr>
              <td style="overflow:hidden;border:1px solid #e6dcd5;border-radius:12px;background:#ffffff;box-shadow:0 18px 48px rgba(70,46,35,0.10);">
                <div style="height:4px;background:#b53b2a;line-height:4px;font-size:4px;">&nbsp;</div>
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                  <tr>
                    <td style="padding:44px 48px 24px;">
                      <div style="font-size:11px;line-height:16px;font-weight:700;letter-spacing:1.4px;color:#b53b2a;text-transform:uppercase;">Application &middot; Step 1 of 2</div>
                      <h1 style="margin:14px 0 14px;font-family:'Space Grotesk',Arial,sans-serif;font-size:34px;line-height:40px;letter-spacing:0;color:#182126;">Verify your email</h1>
                      <p style="margin:0;font-size:16px;line-height:26px;color:#615a56;">Welcome to Umbrella. Confirm <strong style="color:#2c2927;">{safe_recipient}</strong> to continue your application to the research community.</p>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:8px 48px 34px;">
                      <table role="presentation" cellspacing="0" cellpadding="0" border="0">
                        <tr>
                          <td align="center" style="border-radius:8px;background:#b53b2a;box-shadow:0 9px 22px rgba(181,59,42,0.22);">
                            <a href="{safe_url}" style="display:inline-block;padding:15px 25px;font-size:15px;line-height:18px;font-weight:700;color:#ffffff;text-decoration:none;">Verify email address&nbsp;&nbsp;&rarr;</a>
                          </td>
                        </tr>
                      </table>
                      <p style="margin:16px 0 0;font-size:13px;line-height:20px;color:#8a817c;">This secure link expires in {expires_in_minutes} minutes.</p>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:0 48px 34px;">
                      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border:1px solid #eee5df;border-radius:8px;background:#fbf9f7;">
                        <tr>
                          <td width="50%" valign="top" style="padding:18px 16px 18px 20px;border-right:1px solid #eee5df;">
                            <div style="font-size:11px;line-height:15px;font-weight:700;color:#b53b2a;">01 &nbsp; EMAIL</div>
                            <div style="padding-top:6px;font-size:13px;line-height:18px;font-weight:600;color:#2c2927;">Verify your address</div>
                          </td>
                          <td width="50%" valign="top" style="padding:18px 20px 18px 16px;">
                            <div style="font-size:11px;line-height:15px;font-weight:700;color:#9b938f;">02 &nbsp; REVIEW</div>
                            <div style="padding-top:6px;font-size:13px;line-height:18px;color:#6f6864;">Admin approval follows</div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:24px 48px 34px;border-top:1px solid #eee5df;background:#fdfbf9;">
                      <p style="margin:0 0 8px;font-size:13px;line-height:19px;font-weight:700;color:#2c2927;">Secure registration</p>
                      <p style="margin:0;font-size:13px;line-height:21px;color:#7b736f;">Umbrella will never ask you to share your password or verification code. If you did not create this account, no action is required.</p>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr>
              <td style="padding:22px 8px 0;text-align:center;font-size:11px;line-height:18px;color:#918984;">
                <p style="margin:0 0 8px;">Button not working? Open this secure link:</p>
                <p style="margin:0;word-break:break-all;"><a href="{safe_url}" style="color:#a23a2b;text-decoration:underline;">{safe_url}</a></p>
                <p style="margin:18px 0 0;">Umbrella Animal BioHub &middot; One platform. All animal genomics. Powered by AI.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""
    return text, html
