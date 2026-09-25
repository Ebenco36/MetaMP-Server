import os

from flask import current_app

from src.utils.common import TokenGenerator
from src.utils.emailService import EmailService


def send_forgot_password_email(request, user):
    mail_subject = "Reset your password"
    domain = os.environ.get("API_URL") or request.url_root.rstrip("/")
    uid = user.id
    token = TokenGenerator.encode_token(user)
    email_service = EmailService()
    sender = (
        current_app.config.get("MAIL_DEFAULT_SENDER")
        or os.environ.get("MAIL_DEFAULT_SENDER")
        or "admin@admin.com"
    )
    email_service.send_email(
        sender=sender,
        subject=mail_subject,
        recipients=[user.email],
        body="Reset your password",
        html=(
            "Please click on the link to reset your password: "
            f"{domain}/pages/auth/reset-password/{uid}/{token}"
        ),
    )
