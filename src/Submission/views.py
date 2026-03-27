from flask import request
from flask_restful import Resource
from marshmallow import ValidationError

from src.Submission.schema import SubmissionSchema
from src.Submission.services import (
    EmailSubmissionNotifier,
    SQLAlchemySubmissionRepository,
    SubmissionNotFoundError,
    SubmissionService,
)
from src.utils.emailService import EmailService
from src.utils.response import ApiResponse

submission_schema = SubmissionSchema()
submissions_schema = SubmissionSchema(many=True)


def build_submission_service() -> SubmissionService:
    repository = SQLAlchemySubmissionRepository()
    notifier = EmailSubmissionNotifier(EmailService())
    return SubmissionService(repository=repository, notifier=notifier)


class SubmissionAPI(Resource):
    def __init__(self):
        self.service = build_submission_service()

    def get(self, id=None):
        try:
            if id is not None:
                submission = self.service.get_submission(id)
                return ApiResponse.success(submission_schema.dump(submission))

            submissions = self.service.list_submissions()
            return ApiResponse.success(submissions_schema.dump(submissions))
        except SubmissionNotFoundError as error:
            return ApiResponse.error(str(error), 404)

    def post(self):
        try:
            data = submission_schema.load(request.get_json())
        except ValidationError as err:
            return ApiResponse.error("Validation Error", 400, err.messages)

        new_submission = self.service.create_submission(data)
        return ApiResponse.success(
            submission_schema.dump(new_submission),
            "Submission created successfully",
            201,
        )

    def put(self, id):
        try:
            data = submission_schema.load(request.get_json(), partial=True)
        except ValidationError as err:
            return ApiResponse.error("Validation Error", 400, err.messages)

        try:
            submission = self.service.update_submission(id, data)
            return ApiResponse.success(submission_schema.dump(submission))
        except SubmissionNotFoundError as error:
            return ApiResponse.error(str(error), 404)

    def delete(self, id):
        try:
            submission = self.service.delete_submission(id)
            return ApiResponse.success(submission_schema.dump(submission))
        except SubmissionNotFoundError as error:
            return ApiResponse.error(str(error), 404)
