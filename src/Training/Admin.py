from app import app
from database.db import db
from flask_admin import Admin
from src.Training.models import Question
from flask_admin.contrib.sqla import ModelView

admin = Admin(app)

class QuestionAdminView(ModelView):
    column_list = ('id', 'text', 'answers')

admin.add_view(QuestionAdminView(Question, db.session))
