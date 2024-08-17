from flask import current_app as app
from flask import request, jsonify
from database.db import db
from src.Submission.models import Submission
from flask_restful import Resource
from src.Submission.schema import SubmissionSchema
from src.utils.emailService import EmailService
from src.utils.response import ApiResponse
from marshmallow import ValidationError

submission_schema = SubmissionSchema()
submissions_schema = SubmissionSchema(many=True)

class SubmissionAPI(Resource):

    def get(self, id=None):
        if id:
            submission = Submission.query.get(id)
            if not submission:
                return {"message": "Submission not found"}, 404
            serialized_data = submission_schema.dump(submission)
            return jsonify(serialized_data)
        else:
            submissions = Submission.query.all()
            serialized_data = submissions_schema.dump(submissions)
            return jsonify(serialized_data)

    def post(self):
        try:
            # Parse and validate input data using the schema
            data = submission_schema.load(request.get_json())
        except ValidationError as err:
            # Return validation errors with clear messages
            return {"errors": err.messages}, 400

        new_submission = Submission(
            name=data['name'],
            email=data['email'],
            institution=data['institution'],
            request_type=data['request_type'],
            protein_code_or_name=data['protein_code_or_name'],
            description=data['description']
        )
        db.session.add(new_submission)
        db.session.commit()
        serialized_data = submission_schema.dump(new_submission)
        
        # Initialize EmailService with Flask app
        email_service = EmailService(app)
        data = {
            "subject": "Message from: " + data['name'] + " with email: " + data['email'],
            "sender": data['email'],
            "recipients": ["ola@gmail.com", "ebenco94@gmail.com"], 
            "body": "Message from: " + data['name'],
            "html": data['description']
        }
        # Use the EmailService to send the email
        email_service.send_email_from_data(data)

        return ApiResponse.success(
            serialized_data, 'Submission created successfully', 201
        )

    def put(self, id):
        submission = Submission.query.get(id)
        if not submission:
            return {"message": "Submission not found"}, 404

        try:
            # Parse and validate input data using the schema
            data = submission_schema.load(request.get_json(), partial=True)  # partial=True for partial updates
        except ValidationError as err:
            # Return validation errors with clear messages
            return {"errors": err.messages}, 400

        # Update submission fields with validated data
        if 'name' in data:
            submission.name = data['name']
        if 'email' in data:
            submission.email = data['email']
        if 'institution' in data:
            submission.institution = data['institution']
        if 'request_type' in data:
            submission.request_type = data['request_type']
        if 'protein_code_or_name' in data:
            submission.protein_code_or_name = data['protein_code_or_name']
        if 'description' in data:
            submission.description = data['description']

        db.session.commit()
        serialized_data = submission_schema.dump(submission)
        return jsonify(serialized_data)

    def delete(self, id):
        submission = Submission.query.get(id)
        if not submission:
            return {"message": "Submission not found"}, 404

        db.session.delete(submission)
        db.session.commit()
        serialized_data = submission_schema.dump(submission)
        return jsonify(serialized_data)
