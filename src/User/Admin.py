from app import app
from database.db import db
from flask_admin import Admin
from src.User.model import UserModel
from flask_admin.contrib.sqla import ModelView

admin = Admin(app)

class UserAdminView(ModelView):
    column_list = ('id', 'text', 'answers')

admin.add_view(UserAdminView(UserModel, db.session))
