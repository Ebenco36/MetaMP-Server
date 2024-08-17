from src.Training.views import (
    CategoryResource, 
    CategoryListResource, 
    QuestionResource, 
    QuestionsResource, 
    OptionResource, 
    OptionsResource, 
    TestingResources, 
    UserAnswerResource, 
    UserResponsesResource,
    generateChartForQuestions,
    FilterToolResource
)

def training_routes(api):
    api.add_resource(CategoryResource, '/admin/categories/<int:category_id>')
    api.add_resource(CategoryListResource, '/admin/categories')
    api.add_resource(QuestionResource, '/admin/question/<int:question_id>')
    api.add_resource(QuestionsResource, '/admin/questions')
    api.add_resource(OptionResource, '/admin/option/<int:option_id>')
    api.add_resource(OptionsResource, '/admin/options')
    # api.add_resource(FilterToolResource, '/filter_tools')
    # api.add_resource(FilterToolResource, '/filter_tools/<int:question_id>')
    api.add_resource(UserAnswerResource, '/save-user-answer')
    api.add_resource(UserResponsesResource, '/user-responses/<int:user_id>')
    api.add_resource(TestingResources, '/api/test')
    #chart for question
    api.add_resource(generateChartForQuestions, '/chart/questions')