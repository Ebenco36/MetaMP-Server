from src.Submission.views import (
    SubmissionAPI, 
)

def submission_routes(api):
    api.add_resource(SubmissionAPI, '/submission')