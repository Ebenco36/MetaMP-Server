from marshmallow import Schema, fields, validates, ValidationError

class SubmissionSchema(Schema):
    id = fields.Int(dump_only=True)  # Read-only field
    name = fields.Str(required=True, validate=lambda x: x.strip() != "", error_messages={"required": "Name cannot be blank!"})
    email = fields.Str(required=True, validate=lambda x: x.strip() != "", error_messages={"required": "Email cannot be blank!"})
    institution = fields.Str(required=True, validate=lambda x: x.strip() != "", error_messages={"required": "Institution cannot be blank!"})
    request_type = fields.Str(required=True, validate=lambda x: x.strip() != "", error_messages={"required": "Request type cannot be blank!"})
    protein_code_or_name = fields.Str(required=True, validate=lambda x: x.strip() != "", error_messages={"required": "Protein code or name cannot be blank!"})
    description = fields.Str(required=True, validate=lambda x: x.strip() != "", error_messages={"required": "Description cannot be blank!"})

    @validates('email')
    def validate_email(self, value):
        if '@' not in value:
            raise ValidationError("Invalid email address.")
