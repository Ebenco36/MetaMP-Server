# serializers.py

from flask import Flask
from datetime import datetime
from src.Training.models import Category
from flask_marshmallow import Marshmallow
from marshmallow import Schema, fields, validates, ValidationError

app = Flask(__name__)
with app.app_context():
    ma = Marshmallow(app)

class OptionSchema(Schema):
    id = fields.Int()
    text = fields.Str()
    is_correct = fields.Bool()
    question_id = fields.Int()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()


class FilterToolOptionSchema(Schema):
    id = fields.Int()
    text = fields.Str()
    value = fields.Str()
    filter_tool_id = fields.Int()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
     
class FilterToolSchema(Schema):
    id = fields.Int()
    title = fields.Str()
    name = fields.Str()
    parent = fields.Str()
    selected_option = fields.Str()
    question_id = fields.Int()
    filter_tool_options = fields.Nested(FilterToolOptionSchema, many=True, data_key='filter_tool_options')
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
   
class QuestionSchema(Schema):
    id = fields.Int()
    text = fields.Str()
    category_id = fields.Int()
    item_order = fields.Int()
    question_type = fields.Str()
    instruction = fields.Str()
    hints = fields.Str()
    created_at = fields.DateTime()
    updated_at = fields.DateTime()
    options = fields.Nested(OptionSchema, many=True, data_key='options')
    filter_tools = fields.Nested(FilterToolSchema, many=True)
    
    @validates('options')
    def validate_options(self, value):
        # Ensure at least one option is marked as correct
        correct_options = [option for option in value if option.get('is_correct')]
        if not correct_options:
            raise ValidationError('At least one option must be marked as correct.')

class CategorySchema(ma.SQLAlchemyAutoSchema):
    name = fields.Str()
    description = fields.Str()
    questions = fields.Nested(QuestionSchema, many=True)

    class Meta:
        model = Category

class UserResponseSerializer(Schema):
    id = fields.Integer(dump_only=True)
    session_id = fields.String(required=True)
    user_id = fields.Integer(required=True)
    question_id = fields.Integer(required=True)
    option_id = fields.Integer(required=True)
    is_correct = fields.Boolean(required=True, default=False)
    created_at = fields.DateTime(dump_only=True, default=datetime.utcnow)
    updated_at = fields.DateTime(dump_only=True, default=datetime.utcnow)