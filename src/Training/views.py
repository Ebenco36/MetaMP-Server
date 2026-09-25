import uuid
from flask import g, request
from http import HTTPStatus
from flask_restful import Resource, reqparse
from src.Training.serializers import (
    FilterToolSchema, QuestionSchema, CategorySchema
)
from src.middlewares.auth_middleware import token_required
from src.utils.response import ApiResponse


def _training_services():
    from src.Training.services import (
        CategoryService,
        FilterToolService,
        OptionService,
        QuestionService,
        TrainingAnalyticsService,
        TrainingPresenter,
        UserResponseService,
        get_records_on_method,
    )

    return {
        "CategoryService": CategoryService,
        "FilterToolService": FilterToolService,
        "OptionService": OptionService,
        "QuestionService": QuestionService,
        "TrainingAnalyticsService": TrainingAnalyticsService,
        "TrainingPresenter": TrainingPresenter,
        "UserResponseService": UserResponseService,
        "get_records_on_method": get_records_on_method,
    }


def build_category_parser():
    parser = reqparse.RequestParser()
    parser.add_argument("name", type=str, required=True, help="Name cannot be blank")
    parser.add_argument("description", type=str, required=False)
    return parser


def build_question_parser():
    parser = reqparse.RequestParser()
    parser.add_argument("text", type=str, required=True, help="Question text")
    parser.add_argument("category_id", type=int, required=True, help="Question category")
    parser.add_argument("item_order", type=int, required=True, help="Question order")
    parser.add_argument("instructions", type=str, required=False, help="Question Instructions")
    parser.add_argument("hints", type=str, required=False, help="Question Hints")
    parser.add_argument("question_type", type=str, required=False, help="Question Type")
    return parser


def build_option_parser():
    parser = reqparse.RequestParser()
    parser.add_argument("text", type=str, required=True, help="Text of the answer")
    parser.add_argument("question_id", type=int, required=True, help="question id")
    parser.add_argument("is_correct", type=bool, required=False, default=False, help="Is the answer correct")
    return parser


def build_filter_tool_parser():
    parser = reqparse.RequestParser()
    parser.add_argument("title", type=str, required=True, help="Title is required")
    parser.add_argument("name", type=str, required=True, help="Name is required")
    parser.add_argument("parent", type=str, required=False)
    parser.add_argument("selected_option", type=str, required=False)
    parser.add_argument("question_id", type=int, required=True, help="Question ID is required")
    parser.add_argument("options", type=list, location="json", required=False, default=[])
    return parser

class CategoryResource(Resource):
    @token_required
    def get(self, category_id):
        services = _training_services()
        category = services["CategoryService"].get_category(category_id)
        category_schema = CategorySchema()
        return category_schema.dump(category)
    
    @token_required
    def put(self, category_id):
        args = build_category_parser().parse_args()
        services = _training_services()

        # Check if the category with the given ID exists
        existing_category = services["CategoryService"].get_category(category_id)
        if not existing_category:
            return {"message": "Category not found"}, 404

        # Update the category
        updated_category = services["CategoryService"].update_category(category_id, args['name'], args['description'])

        if updated_category:
            category_schema = CategorySchema()  # Replace with your actual CategorySchema
            return category_schema.dump(updated_category)
        else:
            return {"message": "Cannot update category. Another category with the same name may exist."}, 400
        
    @token_required
    def delete(self, category_id):
        services = _training_services()
        deleted = services["CategoryService"].delete_category(category_id)
        if deleted:
            return {"message": f"Category deleted successfully."}
        else:
            return {"message": "Cannot delete category. Questions are attached."}, 400

class CategoryListResource(Resource):
    # @token_required
    def get(self):
        categories = _training_services()["CategoryService"].get_all_categories()

        # Filter categories based on the specified question_type
        """
        filtered_categories = [category for category in categories if any(
            question.question_type == question_type for question in category.questions
        )]
        """

        category_schema = CategorySchema(many=True)
        return category_schema.dump(categories)
        return filter_questions_in_sets(question_sets=category_schema.dump(categories), class_name=question_type)

    @token_required
    def post(self):
        args = build_category_parser().parse_args()
        services = _training_services()

        new_category = services["CategoryService"].create_category(args['name'], args['description'])
        if new_category:
            category_schema = CategorySchema()
            return category_schema.dump(new_category), 201
        else:
            return {"message": "Category with this name already exists."}, 400

class QuestionResource(Resource):
    @token_required
    def get(self, question_id):
        services = _training_services()
        question = services["QuestionService"].get_question_by_id(question_id)
        if not question:
            return {"message": "Question not found"}, HTTPStatus.NOT_FOUND
        question_schema = QuestionSchema()
        return question_schema.dump(question), 200
    
    @token_required
    def put(self, question_id):
        args = build_question_parser().parse_args()
        services = _training_services()
        question = services["QuestionService"].get_question_by_id(question_id)
        if question:
            services["QuestionService"].update_question(
                question_id,
                args['text'],
                args['category_id'],
                args['item_order'],
                args['question_type'],
                args['instructions'],
                args['hints'],
            )
            return {'message': 'Question updated successfully'}
        return {'message': 'Question not found'}, 404

    @token_required
    def delete(self, question_id):
        services = _training_services()
        question = services["QuestionService"].get_question_by_id(question_id)
        if question:
            services["QuestionService"].delete_question(question_id)
            return {'message': 'Question deleted successfully'}
        return {'message': 'Question not found'}, 404

class QuestionsResource(Resource):
    @token_required
    def get(self):
        questions = _training_services()["QuestionService"].get_all_questions()
        question_schema = QuestionSchema(many=True)
        return question_schema.dump(questions), 200

    @token_required
    def post(self):
        args = build_question_parser().parse_args()
        services = _training_services()
        question = services["QuestionService"].create_question(
            args['text'], 
            args['category_id'],
            args['item_order'], 
            args['question_type'], 
            args['instructions'],
            args['hints']
        )
        return services["TrainingPresenter"].question_payload(question), 201

class OptionResource(Resource):
    
    @token_required
    def get(self, option_id):
        services = _training_services()
        option = services["OptionService"].get_option_by_id(option_id)
        if(option):
            return services["TrainingPresenter"].option_payload(option), 200
        else: 
            return {'message': 'Option not found'}, 404

    @token_required
    def put(self, option_id):
        args = build_option_parser().parse_args()
        services = _training_services()
        answer = services["OptionService"].get_option_by_id(option_id)
        if answer:
            services["OptionService"].update_option(option_id, args['text'], args['is_correct'])
            return {'message': 'Option updated successfully'}
        return {'message': 'Option not found'}, 404

    @token_required
    def delete(self, option_id):
        services = _training_services()
        answer = services["OptionService"].get_option_by_id(option_id)
        if answer:
            services["OptionService"].delete_option(option_id)
            return {'message': 'Option deleted successfully'}
        return {'message': 'Option not found'}, 404

class OptionsResource(Resource):
    @token_required
    def get(self):
        options = _training_services()["OptionService"].get_all_options()
        return [{'id': a.id, 'text': a.text, 'is_correct': a.is_correct, 'question_id': a.question_id} for a in options]

    @token_required
    def post(self):
        args = build_option_parser().parse_args()
        services = _training_services()
        option = services["OptionService"].create_option(args['text'], args['question_id'], args['is_correct'])
        return services["TrainingPresenter"].option_payload(option), 201

# FilterToolView
class FilterToolResource(Resource):
    def post(self):
        args = build_filter_tool_parser().parse_args()
        services = _training_services()

        # Check if the question with the given ID exists
        question = services["QuestionService"].get_question_by_id(args['question_id'])
        if not question:
            return {"message": "Question not found"}, 404

        # Create a new filter tool
        filter_tool = services["FilterToolService"].create_filter_tool(
            args['title'], args['name'], args['parent'], args['selected_option'], args['question_id'], args['options']
        )

        filter_tool_schema = FilterToolSchema()
        return filter_tool_schema.dump(filter_tool), 201

    def get(self, filter_tool_id):
        # Get filter tool with options
        filter_tool = _training_services()["FilterToolService"].get_filter_tool(filter_tool_id)

        filter_tool_schema = FilterToolSchema()
        return filter_tool_schema.dump(filter_tool), 200

    def put(self, filter_tool_id):
        args = build_filter_tool_parser().parse_args()
        services = _training_services()

        # Update filter tool
        filter_tool = services["FilterToolService"].update_filter_tool(
            filter_tool_id, args['title'], args['name'], args['parent'], args['selected_option'], args['options']
        )

        filter_tool_schema = FilterToolSchema()
        return filter_tool_schema.dump(filter_tool), 200

    def delete(self, filter_tool_id):
        # Delete filter tool
        _training_services()["FilterToolService"].delete_filter_tool(filter_tool_id)
        return {"message": "Filter Tool deleted successfully"}, 200

    
    
class UserAnswerResource(Resource):
    @token_required
    def post(self):
        data = request.get_json()
        session_id = request.headers.get('X-Session-ID') or str(uuid.uuid4())
        current_user = g.current_user
        response = _training_services()["UserResponseService"].check_user_response(current_user.id, session_id, data)
        return response
    
class UserResponsesResource(Resource):
    @token_required
    def get(self, user_id):
        services = _training_services()
        user_responses = services["UserResponseService"].get_user_responses(user_id)
        return services["TrainingPresenter"].user_response_payloads(user_responses)


class TestingResources(Resource):
    def get(self):
        data = _training_services()["get_records_on_method"]()
        return data
    
    
def chartForTraining(
    chart_data, x="resolution", y="rcsentinfo_molecular_weight", 
    tooltips=["rcsentinfo_molecular_weight", 'resolution', 'group', 'pdb_code' ], color="group"
):
    import altair

    from src.services.graphs.helpers import convert_chart

    chart = altair.Chart(chart_data).mark_point().encode(
        x=altair.Y(x), 
        y=altair.Y(y),
        color=color,
        tooltip=[altair.Tooltip(tooltip, title=tooltip.capitalize()) for tooltip in tooltips]
    ).properties(
        width="container"
    ).interactive().configure_legend(
        orient='bottom'
    )
    return convert_chart(chart)

class generateChartForQuestions(Resource):
    def __init__(self):
        self.service = _training_services()["TrainingAnalyticsService"]()
        
    def post(self):
        try:
            return self.service.generate_response(request.get_json())
        except ValueError as exc:
            return {"message": str(exc)}, HTTPStatus.BAD_REQUEST
            
            
