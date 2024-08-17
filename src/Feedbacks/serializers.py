# serializers.py
from marshmallow import Schema, fields

class UserSchema(Schema):
    class Meta:
        fields = ('id', 'username', 'email', 'name')  # Add other user-related fields

class OptionSchema(Schema):
    id = fields.Int(required=True)
    value = fields.Int(required=True)
    
class ResponseSchema(Schema):
    questionId = fields.Int(required=True)
    questionText = fields.Str(required=True)
    option = fields.Nested(OptionSchema, required=True)
    
class FeedbackSchema(Schema):
    id = fields.Int(dump_only=True)
    comment = fields.Str(required=True)
    responses = fields.List(fields.Nested(ResponseSchema), required=True)
    name = fields.Str(required=True)
    gender = fields.Str(required=False)
    domain = fields.Str(required=False)
    years_of_experience = fields.Str(required=True)
    is_student = fields.Int(required=False)
    
    user_id = fields.Int(required=True)  # Add user_id field
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    user = fields.Nested(UserSchema, only=('id',))

    def serialize(self, feedback):
        return {
            'id': feedback.id,
            'comment': feedback.comment,
            'responses': feedback.responses,
            'name': feedback.name,
            'gender': feedback.gender,
            'domain': feedback.domain,
            'is_student': feedback.is_student,
            'years_of_experience': feedback.years_of_experience,
            'user': UserSchema().dump(feedback.user),
            'created_at': feedback.created_at,
            'updated_at': feedback.updated_at,
        }

class FeedbackOptionSchema(Schema):
    id = fields.Int()
    value = fields.Int()
    text = fields.Str()

class FeedbackQuestionSchema(Schema):
    id = fields.Int()
    question_text = fields.Str()
    options = fields.List(fields.Nested(FeedbackOptionSchema)) # fields.Nested(FeedbackOptionSchema, many=True)

class FeedbackQuestionWithAnswersSchema(Schema):
    id = fields.Int()
    question_text = fields.Str()
    answers = fields.List(fields.Nested(FeedbackOptionSchema))