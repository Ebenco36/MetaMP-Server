from flask_admin import Admin
from flask_babel import Babel
from flask_cors import CORS
from flask_mail import Mail
from flask_socketio import SocketIO

from database.db import db

cors = CORS()
mail = Mail()
admin = Admin()
babel = Babel()
socketio = SocketIO()
