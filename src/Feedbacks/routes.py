from src.Feedbacks.views import ( 
    FeedbackResource, 
    FeedbackListResource, 
    UserFeedbackListResource, 
    FeedbackQuestionResourceAPI,
    FeedbackQuestionUpdateResourceAPI
)

def feedback_routes(api):
    api.add_resource(FeedbackResource, '/feedback')
    api.add_resource(UserFeedbackListResource, '/my-feedback')
    api.add_resource(FeedbackListResource, '/feedback/all')
    api.add_resource(FeedbackQuestionResourceAPI, '/admin/feedback-questions')
    api.add_resource(FeedbackQuestionUpdateResourceAPI, '/admin/feedback-questions/<int:question_id>')