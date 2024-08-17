from src.Contact.models import Contact
from flask_mail import Mail, Message
from src.utils.emailService import EmailService
from src.Contact.serializers import ContactSchema
from flask import current_app
from database.db import db

contact_schema = ContactSchema()

class ContactService:
    def add_contact(self, contact_data):
        validated_data = contact_schema.load(contact_data)
        new_contact = Contact(**validated_data)
        db.session.add(new_contact)
        db.session.commit()

        # Notify admins via email
        # self.notify_admins(new_contact)

    def get_contacts(self):
        contacts = Contact.query.all()
        return [
            {'name': contact.name, 'email': contact.email, 'company': contact.company, 'message': contact.message}
            for contact in contacts
        ]

    def notify_admins(self, new_contact):
        admins = ['admin1@example.com', 'admin2@example.com']  # Update with your admin emails
        # Create an instance of EmailService
        email_service = EmailService(current_app)
        body = (
            f"New contact added:\n"
            f"Name: {new_contact.name}\n"
            f"Email: {new_contact.email}\n"
            f"Company: {new_contact.company}\n"
            f"Message: {new_contact.message}"
        )
        email_service.send_email(
            subject='Hello from Flask-Mail',
            sender='your_email@example.com',
            recipients=admins,
            body=body
        )