from flask_restful import Api
from src.User.views import (
    LoginApi, 
    LogoutApi,
    SignUpApi, 
    UserResource, 
    ResetPassword, 
    ForgotPassword, 
    UserDetailResource, 
    CurrentUserResource, 
)


def create_authentication_routes(api: Api):
    """Adds resources to the api.
    :param api: Flask-RESTful Api Object
    """
    api.add_resource(LoginApi, "/auth/login")
    api.add_resource(LogoutApi, "/auth/logout")
    api.add_resource(SignUpApi, "/auth/register")  
    api.add_resource(UserResource, '/admin/user')
    api.add_resource(ForgotPassword, "/auth/forgot-password")
    api.add_resource(CurrentUserResource, '/auth/current-user')
    api.add_resource(ResetPassword, "/auth/reset-password/<token>")
    api.add_resource(UserDetailResource, '/admin/user/<int:user_id>') 