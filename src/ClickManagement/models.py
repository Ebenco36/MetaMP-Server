from datetime import datetime
from database.db import db
from src.User.model import UserModel

class Click(db.Model):
    __tablename__ = 'clicks'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    clicked_at = db.Column(db.DateTime, default=datetime.utcnow)
    element_id = db.Column(db.String(50))
    data = db.Column(db.Text)
    page_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=True)

    user = db.relationship(UserModel, backref=db.backref('clicks', lazy=True))
