from flask import Flask, request, current_app as app
from flask_restful import Api, Resource
from src.Contact.serializers import ContactSchema
from src.Contact.services import ContactService
from src.utils.emailService import EmailService
from src.utils.response import ApiResponse
from marshmallow import ValidationError

class ContactResource(Resource):
    def __init__(self):
        self.contact_service = ContactService()
        
        
    def get(self):
        contacts = self.contact_service.get_contacts()
        return ApiResponse.success(contacts, "Fetched contact list successfully.", 200)

    def post(self):
        contact_schema = ContactSchema()
        data = request.get_json()
        try:
            contact_schema.load(data)
        except ValidationError as e:
            return ApiResponse.error("Validation Error", 400, e.messages)

        self.contact_service.add_contact(data)
        # Initialize EmailService with Flask app
        email_service = EmailService(app)
        data = {
            "subject": "Message from: " + data['name'] + " from " + data['company'],
            "sender": data['email'],
            "recipients": ["ola@gmail.com", "ebenco94@gmail.com"], 
            "body": "Message from: " + data['name'],
            "html": data['message']
        }
        # Use the EmailService to send the email
        email_service.send_email_from_data(data)
        
        return ApiResponse.success([], "Thank you for getting in touch. We will certainly respond shortly.", 201)