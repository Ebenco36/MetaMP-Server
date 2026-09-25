# models.py

from datetime import datetime
from database.db import db

class Category(db.Model):
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text(), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    questions = db.relationship('Question', backref='category', lazy=True)

class Question(db.Model):
    __tablename__ = 'questions'
    
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text(), nullable=False)
    item_order = db.Column(db.Integer, nullable=False)
    question_type = db.Column(db.String(50), nullable=False)
    instruction = db.Column(db.Text, nullable=True)
    hints = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define a foreign key relationship with categories
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)
    
    # Define a relationship with options (assuming 'Option' is the model name)
    options = db.relationship('Option', backref='question', lazy=True)
    
    # Define a relationship with filter_tools
    filter_tools = db.relationship('FilterTool', backref='question', lazy=True)



class Option(db.Model):
    __tablename__ = 'options'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(255), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Define a foreign key relationship with questions
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)

class FilterToolOption(db.Model):
    __tablename__ = 'filter_tool_options'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text(), nullable=False)
    value = db.Column(db.Text(), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Define a foreign key relationship with filter_tools
    filter_tool_id = db.Column(db.Integer, db.ForeignKey('filter_tools.id'), nullable=False)

class FilterTool(db.Model):
    __tablename__ = 'filter_tools'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    parent = db.Column(db.String(50), nullable=True)
    selected_option = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    # Define a foreign key relationship with questions
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    # Define a relationship with filter_tool_options
    filter_tool_options = db.relationship('FilterToolOption', backref='filter_tool', lazy=True)
    
    
class UserResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, nullable=False) 
    duration = db.Column(db.String(50), nullable=True)
    endTime = db.Column(db.String(50), nullable=True)  
    startTime = db.Column(db.String(50), nullable=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    answer_id = db.Column(db.Integer, db.ForeignKey('options.id'), nullable=False)
    is_correct = db.Column(db.Boolean, default=False, nullable=False)
    time_taken = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)