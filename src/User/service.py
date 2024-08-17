import json
import jwt
import datetime
from os import environ
from database.db import db
from sqlalchemy import or_
from flask import g, jsonify, make_response
from src.Feedbacks.serializers import FeedbackSchema
from src.Training.serializers import QuestionSchema, UserResponseSerializer
from src.User.helper import send_forgot_password_email
from src.User.model import UserModel
from sqlalchemy.orm import class_mapper
from src.Feedbacks.models import Feedback
from src.Training.models import Question, UserResponse
from src.middlewares.auth_middleware import token_required
from src.utils.common import generate_response, TokenGenerator
from src.User.validation import (
    CreateLoginInputSchema,
    CreateResetPasswordEmailSendInputSchema,
    CreateSignupInputSchema, ResetPasswordInputSchema,
)
from src.utils.http_code import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
from http import HTTPStatus

def create_user(request, input_data):
    """
    It creates a new user
    :param request: The request object
    :param input_data: This is the data that is passed to the function
    :return: A response object
    """
    # Validate input_data using a schema (replace 'YourValidationSchema' with the actual schema class)
    create_validation_schema = CreateSignupInputSchema()
    errors = create_validation_schema.validate(input_data)
    
    if errors:
        return generate_response(message=errors, status=HTTPStatus.BAD_REQUEST)

    # Check if username or email already exists
    check_username_exist = UserModel.query.filter_by(username=input_data.get("username")).first()
    check_email_exist = UserModel.query.filter_by(email=input_data.get("email")).first()
    if check_username_exist:
        return generate_response(message="Username already exists", status=HTTPStatus.BAD_REQUEST)
    elif check_email_exist:
        return generate_response(message="Email already taken", status=HTTPStatus.BAD_REQUEST)

    # Create a new user instance
    new_user = UserModel(**input_data)
    new_user.password = generate_password_hash(input_data.get("password"), method='sha256')
    # Add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    # Response data without the password
    del input_data["password"]
    return generate_response(data=input_data, message="Registration successful", status=HTTPStatus.CREATED)


def login_user(request, input_data):
    """
    It takes in a request and input data, validates the input data, checks if the user exists, checks if
    the password is correct, and returns a response
    :param request: The request object
    :param input_data: The data that is passed to the function
    :return: A dictionary with the keys: data, message, status
    """
    # Validate input_data using a schema (replace 'YourValidationSchema' with the actual schema class)
    create_validation_schema = CreateLoginInputSchema()
    errors = create_validation_schema.validate(input_data)

    if errors:
        current_app.logger.info(errors)
        return generate_response(message=errors, status=HTTPStatus.BAD_REQUEST)

    email_or_username = input_data.get("email")
    
    # Use 'filter' instead of 'filter_by' and 'or_' to create an OR condition
    get_user = UserModel.query.filter(
        (UserModel.email == email_or_username) | (UserModel.username == email_or_username)
    ).first()

    current_app.logger.info(get_user)

    if get_user is None:
        return generate_response(message="User not found", status=HTTPStatus.NOT_FOUND)
    # check password 
    if check_password_hash(get_user.password, input_data.get("password")):
        token = jwt.encode(
            {
                "id": get_user.id,
                "email": get_user.email,
                "username": get_user.username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
            },
            current_app.config["SECRET_KEY"],
        )
        response_data = {
            "user": {
                "id": get_user.id,
                "email": get_user.email,
                "username": get_user.username,
                "name": get_user.name,
                'location': get_user.location, 
                'institute': get_user.institute,
                "has_taken_tour": get_user.has_taken_tour
            },
            "token": token
        }
        return generate_response(
            data=response_data, message="User login successfully", status=HTTPStatus.CREATED
        )
    else:
        return generate_response(message="Password is wrong", status=HTTPStatus.NOT_FOUND)
    
    
def reset_password_email_send(request, input_data):
    """
    It takes an email address as input, checks if the email address is registered in the database, and
    if it is, sends a password reset email to that address
    :param request: The request object
    :param input_data: The data that is passed to the function
    :return: A response object with a message and status code.
    """
    create_validation_schema = CreateResetPasswordEmailSendInputSchema()
    errors = create_validation_schema.validate(input_data)
    if errors:
        return generate_response(message=errors)
    user = UserModel.query.filter_by(email=input_data.get("email")).first()
    if user is None:
        return generate_response(
            message="No record found with this email. please signup first.",
            status=HTTP_400_BAD_REQUEST,
        )
    send_forgot_password_email(request, user)
    return generate_response(
        message="Link sent to the registered email address.", status=HTTP_200_OK
    )


def reset_password(request, input_data, token):
    create_validation_schema = ResetPasswordInputSchema()
    errors = create_validation_schema.validate(input_data)
    if errors:
        return generate_response(message=errors)
    if not token:
        return generate_response(
            message="Token is required!",
            status=HTTP_400_BAD_REQUEST,
        )
    token = TokenGenerator.decode_token(token)
    user = UserModel.query.filter_by(id=token.get('id')).first()
    if user is None:
        return generate_response(
            message="No record found with this email. please signup first.",
            status=HTTP_400_BAD_REQUEST,
        )
    user = UserModel.query.filter_by(id=token['id']).first()
    user.password = generate_password_hash(input_data.get('password')).decode("utf8")
    db.session.commit()
    return generate_response(
        message="New password SuccessFully set.", status=HTTP_200_OK
    )



def logout_user(request):
    """
    Logs out the user by revoking the JWT token
    :param request: The request object
    :return: A dictionary with the keys: message, status
    """
    # Ensure the user is authenticated with a valid token
    token = request.headers.get("Authorization")
    token = token.replace("Bearer ", "")
    
    if not token:
        return generate_response(message="Missing token", status=HTTPStatus.UNAUTHORIZED)

    try:
        decoded_token = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return generate_response(message="Token has expired", status=HTTPStatus.UNAUTHORIZED)
    except jwt.InvalidTokenError as e:
        return generate_response(message="Invalid token", status=HTTPStatus.UNAUTHORIZED)

    # Log the user out by revoking the JWT token
    # Note: You may want to implement a token blacklist mechanism for better security

    current_app.logger.info(f"User {decoded_token['id']} logged out successfully")

    # Create a response with unset cookies (assuming you are using cookies for JWT)
    response = make_response()
    response.delete_cookie("access_token")

    return generate_response(message="User logged out successfully", status=HTTPStatus.OK)

def current_user():
    if g.current_user:
        user_responses = UserResponse.query.filter_by(user_id=g.current_user.id).all()

        # Fetch associated questions based on question_id values
        question_ids = [user_response.question_id for user_response in user_responses]
        questions = Question.query.filter(Question.id.in_(question_ids)).all()

        # Create a dictionary to map question_id to the corresponding Question instance
        question_dict = {question.id: question for question in questions}

        # Serialize UserResponses with nested question information
        user_response_schema = UserResponseSerializer(many=True)

        # Manually include question information in the serialized data
        serialized_data = user_response_schema.dump(user_responses)
        
        for user_response in serialized_data:
            question_id = user_response.get('question_id')
            if question_id is not None:
                user_response['question'] = QuestionSchema().dump(question_dict.get(question_id))


        user_feedbacks = Feedback.query.filter_by(user_id=g.current_user.id).all()
        user_feedbacks_schema = FeedbackSchema(many=True)
        feedback_dicts = user_feedbacks_schema.dump(user_feedbacks)
        
        return {
            "id": g.current_user.id,
            "name": g.current_user.name,
            "phone": g.current_user.phone,
            "email": g.current_user.email,
            'user_responses': serialized_data,
            'user_feedbacks': feedback_dicts,
            "username": g.current_user.username,
            'location': g.current_user.location, 
            'institute': g.current_user.institute,
            "has_taken_tour": g.current_user.has_taken_tour
        }, 200
    else:
        return {'message': 'Unauthorized'}, 401
    
class UserService:
    @staticmethod
    def get_all_users():
        return UserModel.query.all()

    @staticmethod
    def create_user(name, phone, username, email, password, status=True, is_admin=False, location="Berlin", institute="RKI"):
        new_user = UserModel(
            name = name,
            email = email,
            phone = phone,
            location=location,
            username = username, 
            is_admin = is_admin,
            institute=institute
        )
        if(password):
            new_user.password = generate_password_hash(password, method='sha256')
        db.session.add(new_user)
        db.session.commit()
        return new_user

    @staticmethod
    def get_user_by_id(user_id):
        return UserModel.query.get(user_id)
    
    
    @staticmethod
    def get_user_by_username(username):
        return UserModel.query.filter_by(username=username)

    @staticmethod
    def update_user(user_id, name=None, phone=None, status=True, username=None, email=None, is_admin=False, has_taken_tour=False, location=None, institute=None):
        user = UserModel.query.filter_by(id=user_id).first()
        if(username):
            user.username = username
        if(email):
            user.email = email
        if(is_admin):
            user.is_admin = is_admin
        if(name):
            user.name = name
        if(phone):
            user.phone = phone
        if(status):
            user.status = status
        if(location):
            user.location = location
        if(institute):
            user.institute = institute
        if(has_taken_tour):
            user.has_taken_tour = has_taken_tour
        
        db.session.commit()

    @staticmethod
    def delete_user(user_id):
        user = UserModel.query.filter_by(id=user_id).first()
        db.session.delete(user)
        db.session.commit()