from database.db import db
from flask import g, jsonify, request
from marshmallow import ValidationError
from src.utils.response import ApiResponse
from flask_restful import Resource, reqparse
from src.Feedbacks.services import UserFeedbackService
from src.middlewares.auth_middleware import token_required
from src.Feedbacks.models import Feedback, FeedbackOption, FeedbackQuestion
from src.Feedbacks.serializers import (
    FeedbackQuestionSchema, FeedbackSchema,
    FeedbackQuestionWithAnswersSchema
)

feedbacks_schema = FeedbackSchema(many=True)


class FeedbackQuestionResourceAPI(Resource):
    def get(self, question_id=None):
        if question_id is None:
            # Get all questions
            questions = FeedbackQuestion.query.all()
            return ApiResponse.success(
                FeedbackQuestionSchema(many=True).dump(questions), 
                "Fetch records successfully."
            )
        else:
            # Get a specific question by ID
            question = FeedbackQuestion.query.get(question_id)
            if question:
                return ApiResponse.success(
                    FeedbackQuestionWithAnswersSchema().dump(question), 
                    "Fetch records successfully."
                )
            else:
                return ApiResponse.error("Question not found", 404)

    def post(self):
        data = request.json
        try:
            validated_data = FeedbackQuestionSchema().load(data)
        except ValidationError as e:
            return ApiResponse.error(str(e), 400)

        question_text = validated_data['question_text']
        options = validated_data['options']

        new_question = FeedbackQuestion(question_text=question_text)
        for option_data in options:
            new_option = FeedbackOption(value=option_data['value'])
            new_question.options.append(new_option)
        db.session.add(new_question)
        db.session.commit()

        return ApiResponse.success(
            FeedbackQuestionWithAnswersSchema().dump(new_question), 
            "Fetch records successfully.", 201
        )


class FeedbackQuestionUpdateResourceAPI(Resource):
    
    def put(self, question_id):
        data = request.json
        new_options = data.get('options')

        if not new_options:
            return ApiResponse.error("Invalid request format", 400)

        question = FeedbackQuestion.query.get(question_id)
        if question:
            FeedbackOption.query.filter_by(question_id=question_id).delete()
            for option_data in new_options:
                new_option = FeedbackOption(value=option_data['value'])
                question.options.append(new_option)
            db.session.commit()
            return ApiResponse.success(
                FeedbackQuestionWithAnswersSchema().dump(question), 
                "Fetch records successfully.", 201
            )
        else:
            return ApiResponse.error("Question not found", 404)

    def delete(self, question_id):
        question = FeedbackQuestion.query.get(question_id)
        if question:
            db.session.delete(question)
            db.session.commit()
            return ApiResponse.success(
                "", f"Question '{question_id}' deleted successfully", 200
            )
        else:
            return ApiResponse.error("Question not found", 404)


class FeedbackResource(Resource):
    feedback_schema = FeedbackSchema()

    @token_required
    def get(self, user_id):
        feedbacks = UserFeedbackService.get_user_feedbacks(user_id)
        return feedbacks
    
    @token_required
    def post(self):
        current_user = g.current_user
        try:
            # Get the raw JSON data
            json_data = request.get_json()
            # Add the user_id to the data
            json_data['user_id'] = current_user.id
            # Validate and deserialize input
            data = self.feedback_schema.load(json_data)
        except ValidationError as err:
            return ApiResponse.error(
                'Error encountered submitting feedback. Error might have occur as a result of: ' + str(err), 400, []
            )

        # Proceed to save the feedback data
        response = UserFeedbackService.store_user_feedback(data)

        return ApiResponse.success(
            response, 'Feedback submitted successfully', 200
        )


class FeedbackListResource(Resource):
    @token_required
    def get(self):
        feedback_list = Feedback.query.all()

        # Serialize the list of feedbacks using the schema
        result = feedbacks_schema.dump(feedback_list)

        return ApiResponse.success(
            result, 'Feedback list fetched successfully', 200
        )


class UserFeedbackListResource(Resource):
    @token_required
    def get(self):
        current_user = g.current_user
        feedback_list = Feedback.query.filter_by(user=current_user)

        # Serialize the list of feedbacks using the schema
        result = feedbacks_schema.dump(feedback_list)
        
        return ApiResponse.success(
            result, 'Feedback list fetched successfully', 200
        )