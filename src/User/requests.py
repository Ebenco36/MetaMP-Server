# requests.py
from flask_restful import reqparse

class UserRequest:
    @staticmethod
    def get_parser():
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str, required=True, help='Name is required')
        parser.add_argument('phone', type=str, required=False, help='Phone is required')
        parser.add_argument('username', type=str, required=True, help='Username is required')
        parser.add_argument('email', type=str, required=False, help='Email is required')
        parser.add_argument('institute', type=str, required=False, help='Institute is required')
        parser.add_argument('location', type=str, required=False, help='Location is required')
        parser.add_argument('password', type=str, required=True, help='Password is required')
        parser.add_argument('cpassword', type=str, required=True, help='Confirm Password is required')
        parser.add_argument('status', type=bool, default=True)
        parser.add_argument('is_admin', type=bool, default=False)
        return parser

    @staticmethod
    def parse_args():
        return UserRequest.get_parser().parse_args()


class UserUpdateRequest:
    @staticmethod
    def get_parser():
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str, required=False, help='Name is required')
        parser.add_argument('phone', type=str, required=False, help='Phone is required')
        parser.add_argument('username', type=str, required=False, help='Username is required')
        parser.add_argument('institute', type=str, required=False, help='Institute is required')
        parser.add_argument('location', type=str, required=False, help='Location is required')
        parser.add_argument('email', type=str, required=False, help='Email is required')
        parser.add_argument('password', type=str, required=False, help='Password is required')
        parser.add_argument('status', type=bool, default=False)
        parser.add_argument('is_admin', type=bool, default=False)
        parser.add_argument('has_taken_tour', type=bool, default=False)
        return parser

    @staticmethod
    def parse_args():
        return UserUpdateRequest.get_parser().parse_args()
