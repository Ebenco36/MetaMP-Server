import json
import jwt
import datetime
from database.db import db
from flask import g, make_response
from src.Feedbacks.serializers import FeedbackSchema
from src.Training.serializers import QuestionSchema, UserResponseSerializer
from src.User.helper import send_forgot_password_email
from src.User.model import UserModel
from src.Feedbacks.models import Feedback
from src.Training.models import Question, UserResponse
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


class UserRepository:
    def list_all(self):
        return UserModel.query.order_by(UserModel.id.asc()).all()

    def get_by_id(self, user_id):
        return db.session.get(UserModel, user_id)

    def get_by_username(self, username):
        return UserModel.query.filter_by(username=username)

    def get_one_by_username(self, username):
        return UserModel.query.filter_by(username=username).first()

    def get_by_email(self, email):
        return UserModel.query.filter_by(email=email).first()

    def get_by_login_identifier(self, identifier):
        return UserModel.query.filter(
            (UserModel.email == identifier) | (UserModel.username == identifier)
        ).first()

    def create(self, user):
        db.session.add(user)
        db.session.commit()
        return user

    def save(self):
        db.session.commit()

    def delete(self, user):
        db.session.delete(user)
        db.session.commit()


class UserPresenter:
    @staticmethod
    def admin_list_item(user):
        return {
            'id': user.id,
            'name': user.name,
            'phone': user.phone,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin,
        }

    @staticmethod
    def detail(user):
        return {
            'id': user.id,
            'name': user.name,
            'phone': user.phone,
            'email': user.email,
            'username': user.username,
            'is_admin': user.is_admin,
            'location': user.location,
            'institute': user.institute,
            'has_taken_tour': user.has_taken_tour,
        }

    @staticmethod
    def auth_payload(user):
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "name": user.name,
            'location': user.location,
            'institute': user.institute,
            "has_taken_tour": user.has_taken_tour,
        }


class AuthenticationService:
    def __init__(self, repository=None):
        self.repository = repository or UserRepository()

    def register(self, input_data):
        errors = CreateSignupInputSchema().validate(input_data)
        if errors:
            return generate_response(message=errors, status=HTTPStatus.BAD_REQUEST)

        if self.repository.get_one_by_username(input_data.get("username")):
            return generate_response(
                message="Username already exists",
                status=HTTPStatus.BAD_REQUEST,
            )
        if self.repository.get_by_email(input_data.get("email")):
            return generate_response(
                message="Email already taken",
                status=HTTPStatus.BAD_REQUEST,
            )

        new_user = UserModel(**input_data)
        new_user.password = generate_password_hash(
            input_data.get("password"),
            method='sha256',
        )
        self.repository.create(new_user)

        response_data = {
            key: value
            for key, value in input_data.items()
            if key not in {"password", "cpassword"}
        }
        return generate_response(
            data=response_data,
            message="Registration successful",
            status=HTTPStatus.CREATED,
        )

    def login(self, input_data):
        errors = CreateLoginInputSchema().validate(input_data)
        if errors:
            current_app.logger.info(errors)
            return generate_response(message=errors, status=HTTPStatus.BAD_REQUEST)

        user = self.repository.get_by_login_identifier(input_data.get("email"))
        if user is None:
            return generate_response(
                message="User not found",
                status=HTTPStatus.NOT_FOUND,
            )

        if not check_password_hash(user.password, input_data.get("password")):
            return generate_response(
                message="Password is wrong",
                status=HTTPStatus.NOT_FOUND,
            )

        token = jwt.encode(
            {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
            },
            current_app.config["SECRET_KEY"],
        )
        response_data = {
            "user": UserPresenter.auth_payload(user),
            "token": token,
        }
        return generate_response(
            data=response_data,
            message="User login successfully",
            status=HTTPStatus.CREATED,
        )

    def logout(self, request):
        token = request.headers.get("Authorization")
        token = token.replace("Bearer ", "") if token else None

        if not token:
            return generate_response(
                message="Missing token",
                status=HTTPStatus.UNAUTHORIZED,
            )

        try:
            decoded_token = jwt.decode(
                token,
                current_app.config["SECRET_KEY"],
                algorithms=["HS256"],
            )
        except jwt.ExpiredSignatureError:
            return generate_response(
                message="Token has expired",
                status=HTTPStatus.UNAUTHORIZED,
            )
        except jwt.InvalidTokenError:
            return generate_response(
                message="Invalid token",
                status=HTTPStatus.UNAUTHORIZED,
            )

        current_app.logger.info(
            "User %s logged out successfully",
            decoded_token["id"],
        )
        response = make_response()
        response.delete_cookie("access_token")
        return generate_response(
            message="User logged out successfully",
            status=HTTPStatus.OK,
        )

    def current_user_payload(self):
        if not g.current_user:
            return {'message': 'Unauthorized'}, 401

        user_responses = UserResponse.query.filter_by(user_id=g.current_user.id).all()
        question_ids = [user_response.question_id for user_response in user_responses]
        questions = Question.query.filter(Question.id.in_(question_ids)).all()
        question_dict = {question.id: question for question in questions}

        serialized_data = UserResponseSerializer(many=True).dump(user_responses)
        for user_response in serialized_data:
            question_id = user_response.get('question_id')
            if question_id is not None:
                user_response['question'] = QuestionSchema().dump(
                    question_dict.get(question_id)
                )

        user_feedbacks = Feedback.query.filter_by(user_id=g.current_user.id).all()
        feedback_dicts = FeedbackSchema(many=True).dump(user_feedbacks)

        return {
            **UserPresenter.detail(g.current_user),
            'user_responses': serialized_data,
            'user_feedbacks': feedback_dicts,
        }, 200


class PasswordResetService:
    def __init__(self, repository=None):
        self.repository = repository or UserRepository()

    def send_reset_email(self, request, input_data):
        errors = CreateResetPasswordEmailSendInputSchema().validate(input_data)
        if errors:
            return generate_response(message=errors)

        user = self.repository.get_by_email(input_data.get("email"))
        if user is None:
            return generate_response(
                message="No record found with this email. please signup first.",
                status=HTTP_400_BAD_REQUEST,
            )
        send_forgot_password_email(request, user)
        return generate_response(
            message="Link sent to the registered email address.",
            status=HTTP_200_OK,
        )

    def reset_password(self, input_data, token):
        errors = ResetPasswordInputSchema().validate(input_data)
        if errors:
            return generate_response(message=errors)
        if not token:
            return generate_response(
                message="Token is required!",
                status=HTTP_400_BAD_REQUEST,
            )

        decoded_token = TokenGenerator.decode_token(token)
        user = self.repository.get_by_id(decoded_token.get('id'))
        if user is None:
            return generate_response(
                message="No record found with this email. please signup first.",
                status=HTTP_400_BAD_REQUEST,
            )

        user.password = generate_password_hash(input_data.get('password'))
        self.repository.save()
        return generate_response(
            message="New password SuccessFully set.",
            status=HTTP_200_OK,
        )


def create_user(request, input_data):
    return AuthenticationService().register(input_data)


def login_user(request, input_data):
    return AuthenticationService().login(input_data)
    
    
def reset_password_email_send(request, input_data):
    return PasswordResetService().send_reset_email(request, input_data)


def reset_password(request, input_data, token):
    return PasswordResetService().reset_password(input_data, token)



def logout_user(request):
    return AuthenticationService().logout(request)

def current_user():
    return AuthenticationService().current_user_payload()
    
class UserService:
    _repository = UserRepository()

    @staticmethod
    def get_all_users():
        return UserService._repository.list_all()

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
        return UserService._repository.create(new_user)

    @staticmethod
    def get_user_by_id(user_id):
        return UserService._repository.get_by_id(user_id)
    
    
    @staticmethod
    def get_user_by_username(username):
        return UserService._repository.get_by_username(username)

    @staticmethod
    def update_user(user_id, name=None, phone=None, status=True, username=None, email=None, is_admin=False, has_taken_tour=False, location=None, institute=None):
        user = UserService._repository.get_by_id(user_id)
        if user is None:
            return None
        if username is not None:
            user.username = username
        if email is not None:
            user.email = email
        if is_admin is not None:
            user.is_admin = is_admin
        if name is not None:
            user.name = name
        if phone is not None:
            user.phone = phone
        if status is not None:
            user.status = status
        if location is not None:
            user.location = location
        if institute is not None:
            user.institute = institute
        if has_taken_tour is not None:
            user.has_taken_tour = has_taken_tour
        
        UserService._repository.save()
        return user

    @staticmethod
    def delete_user(user_id):
        user = UserService._repository.get_by_id(user_id)
        if user is None:
            return False
        UserService._repository.delete(user)
        return True
