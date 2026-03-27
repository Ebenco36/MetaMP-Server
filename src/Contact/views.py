from flask import request
from flask_restful import Resource
from src.Contact.services import ContactService
from src.utils.response import ApiResponse
from marshmallow import ValidationError


class ContactResource(Resource):
    def __init__(self):
        self.contact_service = ContactService()

    def get(self):
        contacts = self.contact_service.get_contacts()
        return ApiResponse.success(contacts, "Fetched contact list successfully.", 200)

    def post(self):
        try:
            data = request.get_json() or {}
            self.contact_service.submit_contact(data)
        except ValidationError as e:
            return ApiResponse.error("Validation Error", 400, e.messages)

        return ApiResponse.success(
            [],
            "Thank you for getting in touch. We will certainly respond shortly.",
            201,
        )
