from __future__ import annotations

from html import escape


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
