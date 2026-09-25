# models.py

from datetime import datetime
from database.db import db

# Submission Model
class Submission(db.Model):
    __tablename__ = 'submission'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    institution = db.Column(db.String(100), nullable=False)
    request_type = db.Column(db.String(50), nullable=False)
    protein_code_or_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)

    def __init__(self, name, email, institution, request_type, protein_code_or_name, description):
        self.name = name
        self.email = email
        self.institution = institution
        self.request_type = request_type
        self.protein_code_or_name = protein_code_or_name
        self.description = description