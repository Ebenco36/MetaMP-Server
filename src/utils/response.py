from flask import jsonify

class ApiResponse:
    @staticmethod
    def success(data=None, message='Success', status_code=200):
        response = {
            'status': 'success',
            'message': message,
            'data': data
        }
        return response, status_code

    @staticmethod
    def error(message='Error', status_code=400, errors=None):
        response = {
            'status': 'error',
            'message': message,
            'errors': errors
        }
        return response, status_code
