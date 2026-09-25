from flask import g, request
from marshmallow import ValidationError
from flask_restful import Resource

from src.Feedbacks.serializers import (
    FeedbackQuestionSchema,
    FeedbackQuestionWithAnswersSchema,
    FeedbackSchema,
)
from src.Feedbacks.services import (
    FeedbackQuestionService,
    StructureExpertNoteService,
    UserFeedbackService,
)
from src.middlewares.auth_middleware import token_required
from src.utils.response import ApiResponse


feedbacks_schema = FeedbackSchema(many=True)

class FeedbackQuestionResourceAPI(Resource):
    def __init__(self):
        self.feedback_question_service = FeedbackQuestionService()

    def get(self, question_id=None):
        if question_id is None:
            questions = self.feedback_question_service.list_questions()
            return ApiResponse.success(
                FeedbackQuestionSchema(many=True).dump(questions), 
                "Fetch records successfully."
            )

        question = self.feedback_question_service.get_question(question_id)
        if question:
            return ApiResponse.success(
                FeedbackQuestionWithAnswersSchema().dump(question), 
                "Fetch records successfully."
            )
        return ApiResponse.error("Question not found", 404)

    def post(self):
        try:
            question = self.feedback_question_service.create_question(request.get_json() or {})
        except ValidationError as e:
            return ApiResponse.error("Validation Error", 400, e.messages)

        return ApiResponse.success(
            FeedbackQuestionWithAnswersSchema().dump(question), 
            "Fetch records successfully.", 201
        )


class FeedbackQuestionUpdateResourceAPI(Resource):
    def __init__(self):
        self.feedback_question_service = FeedbackQuestionService()

    def put(self, question_id):
        try:
            question = self.feedback_question_service.replace_question_options(
                question_id,
                request.get_json() or {},
            )
        except ValidationError as error:
            return ApiResponse.error("Validation Error", 400, error.messages)

        if question is None:
            return ApiResponse.error("Question not found", 404)
        return ApiResponse.success(
            FeedbackQuestionWithAnswersSchema().dump(question), 
            "Fetch records successfully.", 200
        )

    def delete(self, question_id):
        deleted = self.feedback_question_service.delete_question(question_id)
        if not deleted:
            return ApiResponse.error("Question not found", 404)
        return ApiResponse.success(
            "", f"Question '{question_id}' deleted successfully", 200
        )


class FeedbackResource(Resource):
    def __init__(self):
        self.feedback_schema = FeedbackSchema()
        self.user_feedback_service = UserFeedbackService()

    @token_required
    def get(self):
        current_user = g.current_user
        feedbacks = self.user_feedback_service.get_user_feedbacks(current_user.id)
        return feedbacks
    
    @token_required
    def post(self):
        current_user = g.current_user
        try:
            json_data = request.get_json()
            json_data['user_id'] = current_user.id
            data = self.feedback_schema.load(json_data)
        except ValidationError as err:
            return ApiResponse.error(
                'Error encountered submitting feedback. Error might have occur as a result of: ' + str(err), 400, []
            )

        response = self.user_feedback_service.store_user_feedback(data)

        return ApiResponse.success(
            response, 'Feedback submitted successfully', 200
        )


class FeedbackListResource(Resource):
    def __init__(self):
        self.user_feedback_service = UserFeedbackService()

    @token_required
    def get(self):
        result = self.user_feedback_service.get_all_feedbacks()

        return ApiResponse.success(
            result, 'Feedback list fetched successfully', 200
        )


class UserFeedbackListResource(Resource):
    def __init__(self):
        self.user_feedback_service = UserFeedbackService()

    @token_required
    def get(self):
        current_user = g.current_user
        result = self.user_feedback_service.get_user_feedbacks(current_user.id)
        
        return ApiResponse.success(
            result, 'Feedback list fetched successfully', 200
        )


class StructureExpertNoteResource(Resource):
    def __init__(self):
        self.structure_expert_note_service = StructureExpertNoteService()

    @token_required
    def get(self, pdb_code):
        limit = request.args.get("limit", default=StructureExpertNoteService.DEFAULT_LIST_LIMIT, type=int)
        try:
            payload = self.structure_expert_note_service.list_notes_for_record(
                pdb_code,
                limit=limit,
            )
        except LookupError as error:
            return ApiResponse.error(str(error), 404)

        return ApiResponse.success(
            payload,
            "Structure expert notes fetched successfully",
            200,
        )

    @token_required
    def post(self, pdb_code):
        current_user = g.current_user
        try:
            payload = self.structure_expert_note_service.create_note_for_record(
                pdb_code,
                request.get_json() or {},
                current_user,
            )
        except ValidationError as error:
            return ApiResponse.error("Validation Error", 400, error.messages)
        except LookupError as error:
            return ApiResponse.error(str(error), 404)

        return ApiResponse.success(
            payload,
            "Structure expert note submitted successfully",
            201,
        )
