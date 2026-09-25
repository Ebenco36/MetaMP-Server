from __future__ import annotations

import os

from flask import current_app

from database.db import db
from src.Contact.models import Contact
from src.Contact.serializers import ContactSchema
from src.utils.emailService import EmailService


class ContactRepository:
    def create(self, contact: Contact) -> Contact:
        db.session.add(contact)
        db.session.commit()
        return contact

    def list_all(self) -> list[Contact]:
        return Contact.query.order_by(Contact.created_at.desc()).all()


class ContactNotificationService:
    DEFAULT_RECIPIENTS = (
        "ola@gmail.com",
        "ebenco94@gmail.com",
    )

    def __init__(self, email_service: EmailService | None = None):
        self.email_service = email_service or EmailService()

    def notify(self, contact: Contact) -> None:
        recipients = self._get_recipients()
        if not recipients:
            current_app.logger.warning(
                "Skipping contact notification because no recipients were configured."
            )
            return

        payload = {
            "subject": f"Message from: {contact.name} from {contact.company or 'Unknown company'}",
            "sender": contact.email,
            "recipients": recipients,
            "body": f"Message from: {contact.name}",
            "html": contact.message,
        }
        result = self.email_service.send_email_from_data(payload)
        if result.get("status") != "success":
            current_app.logger.warning(
                "Contact notification failed for email=%s: %s",
                contact.email,
                result.get("message"),
            )

    def _get_recipients(self) -> list[str]:
        configured = current_app.config.get("CONTACT_NOTIFICATION_RECIPIENTS") or os.getenv(
            "CONTACT_NOTIFICATION_RECIPIENTS",
            ",".join(self.DEFAULT_RECIPIENTS),
        )
        return [
            recipient.strip()
            for recipient in configured.split(",")
            if recipient.strip()
        ]


class ContactService:
    def __init__(
        self,
        repository: ContactRepository | None = None,
        notifier: ContactNotificationService | None = None,
        schema: ContactSchema | None = None,
    ):
        self.repository = repository or ContactRepository()
        self.notifier = notifier or ContactNotificationService()
        self.schema = schema or ContactSchema()

    def submit_contact(self, contact_data: dict) -> Contact:
        validated_data = self.schema.load(contact_data)
        contact = Contact(**validated_data)
        persisted_contact = self.repository.create(contact)
        self.notifier.notify(persisted_contact)
        return persisted_contact

    def get_contacts(self):
        contacts = self.repository.list_all()
        return [
            {
                'name': contact.name,
                'email': contact.email,
                'company': contact.company,
                'message': contact.message,
            }
            for contact in contacts
        ]
