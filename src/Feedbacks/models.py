import datetime
from sqlalchemy import JSON
from database.db import db
from src.User.model import UserModel

class FeedbackOption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Integer, nullable=False)
    text = db.Column(db.String(255), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('feedback_question.id'), nullable=False)


class FeedbackQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.String(255), nullable=False)
    options = db.relationship(
        'FeedbackOption',
        backref='feedback_question',
        lazy=True,
        cascade='all, delete-orphan',
        passive_deletes=True,
        order_by='FeedbackOption.value.desc()',
    )

    def to_dict(self):
        return {
            "id": self.id,
            "question_text": self.question_text,
            "options": [option.value for option in self.options]
        }

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.Text)
    responses = db.Column(JSON)
    name = db.Column(db.Text, nullable=False)
    gender = db.Column(db.Text, nullable=False)
    domain = db.Column(db.Text, nullable=False)
    is_student = db.Column(db.Integer, nullable=False)
    years_of_experience = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=True)
    user = db.relationship(UserModel, backref='feedbacks')


class DiscrepancyReview(db.Model):
    __tablename__ = "discrepancy_reviews"

    id = db.Column(db.Integer, primary_key=True)
    pdb_code = db.Column(db.Text, nullable=False, unique=True, index=True)
    canonical_pdb_code = db.Column(db.Text, nullable=True, index=True)
    status = db.Column(db.Text, nullable=False, default="open")
    rationale = db.Column(db.Text, nullable=True)
    reviewer_note = db.Column(db.Text, nullable=True)
    reviewed_group = db.Column(db.Text, nullable=True)
    reviewed_tm_count = db.Column(db.Integer, nullable=True)
    source_snapshot = db.Column(JSON, nullable=True)
    reviewed_by_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=True)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=True,
    )

    reviewer = db.relationship(UserModel, backref='discrepancy_reviews')
