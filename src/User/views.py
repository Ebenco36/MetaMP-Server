from flask import Response
from src.User.service import (
    create_user, 
    reset_password_email_send, 
    login_user, reset_password, 
    current_user, logout_user, 
    UserService
)
from flask_restful import Resource
from flask import request, make_response
from src.utils.response import ApiResponse
from src.middlewares.auth_middleware import token_required
from src.User.requests import UserRequest, UserUpdateRequest
from src.User.service import UserPresenter

class SignUpApi(Resource):
    @staticmethod
    def post() -> Response:
        """
        POST response method for creating user.
        :return: JSON object
        """
        input_data = request.get_json()
        response, status = create_user(request, input_data)
        return make_response(response, status)


class LoginApi(Resource):
    @staticmethod
    def post() -> Response:
        """
        POST response method for login user.
        :return: JSON object
        """
        input_data = request.get_json()
        response, status = login_user(request, input_data)
        return make_response(response, status)


class LogoutApi(Resource):
    @staticmethod
    def get() -> Response:
        response, status = logout_user(request)
        return make_response(response, status)

class ForgotPassword(Resource):
    @staticmethod
    def post() -> Response:
        """
        POST response method for forgot password email send user.
        :return: JSON object
        """
        input_data = request.get_json()
        response, status = reset_password_email_send(request, input_data)
        return make_response(response, status)


class ResetPassword(Resource):
    @staticmethod
    def post(token) -> Response:
        """
        POST response method for save new password.
        :return: JSON object
        """
        input_data = request.get_json()
        response, status = reset_password(request, input_data, token)
        return make_response(response, status)
        
class CurrentUserResource(Resource):
    @token_required
    def get(self):
        response, status = current_user()
        return ApiResponse.success(response, "User fetched successfully", status)
    


class UserResource(Resource):
    @token_required
    def get(self):
        users = UserService.get_all_users()
        user_list = [UserPresenter.admin_list_item(user) for user in users]
        return {'users': user_list, "message": "fetch records successfully."}

    def post(self):
        args = UserRequest.parse_args()
        user = UserService.create_user(
            args['name'], 
            args['phone'], 
            args['username'], 
            args['email'], 
            args['password'], 
            args['status'], 
            args['is_admin'],
            args.get('location'),
            args.get('institute'),
        )
        return {
            'message': 'User created successfully', 
            'user': UserPresenter.admin_list_item(user),
        }

class UserDetailResource(Resource):
    @token_required
    def get(self, user_id):
        user = UserService.get_user_by_id(user_id)
        if user:
            return ApiResponse.success(
                UserPresenter.detail(user),
                "User fetched successfully",
                200,
            )
        else:
            return ApiResponse.error("User not found", 404, "User not found")
        
    @token_required
    def put(self, user_id):
        args = UserUpdateRequest.parse_args()
        updated_user = UserService.update_user(
            user_id, 
            args['name'], 
            args['phone'], 
            args['status'], 
            args['username'], 
            args['email'], 
            args['is_admin'],
            args['has_taken_tour'],
            args.get('location'),
            args.get('institute'),
        )
        if updated_user is None:
            return ApiResponse.error("User not found", 404, "User not found")
        return ApiResponse.success([], "User updated successfully", 201)

    @token_required
    def delete(self, user_id):
        deleted = UserService.delete_user(user_id)
        if not deleted:
            return ApiResponse.error("User not found", 404, "User not found")
        return ApiResponse.success([], "User deleted successfully", 201)
