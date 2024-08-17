# requests.py
from flask_restful import reqparse

class UserClickRequest:
    @staticmethod
    def get_parser():
        parser = reqparse.RequestParser()
        parser.add_argument('event', type=str, required=True, help='event is required')
        return parser

    @staticmethod
    def parse_args():
        return UserClickRequest.get_parser().parse_args()
