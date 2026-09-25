from marshmallow import Schema, fields, validates, ValidationError

class ContactSchema(Schema):
    name = fields.Str(required=True)
    email = fields.Email(required=True)
    company = fields.Str()
    message = fields.Str(required=True)

    @validates('message')
    def validate_message_length(self, value):
        if value and len(value) > 500:
            raise ValidationError('Message length exceeds the maximum allowed (500 characters).')
