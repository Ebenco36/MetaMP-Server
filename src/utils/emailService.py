# email_service.py
from flask_mail import Mail, Message
from flask import current_app

class EmailService:
    def __init__(self, app=None):
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self.mail = Mail(app)

    def send_email(self, sender="admin@admin.com", subject="test title", recipients=[], body="", html=None):
        """Sends an email with the provided details."""
        msg = Message(
            subject=subject,
            sender=sender,
            recipients=recipients,
            body=body,
            html=html
        )
        try:
            self.mail.send(msg)
            return {"status": "success", "message": "Email sent successfully"}
        except Exception as e:
            current_app.logger.error(f"Failed to send email: {e}")
            return {"status": "error", "message": str(e)}

    def send_email_from_data(self, data: dict):
        """Validates and sends an email using the provided data dictionary."""
        subject = data.get('subject')
        sender = data.get('sender')
        recipients = data.get('recipients')
        body = data.get('body')
        html = data.get('html', None)

        if not subject or not recipients or not body:
            return {"status": "error", "message": "Subject, recipients, and body are required fields"}

        # Use the send_email method to actually send the email
        return self.send_email(sender, subject, recipients, body, html)
