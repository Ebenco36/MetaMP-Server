
from flask import request
from flask_restful import Resource
from src.utils.response import ApiResponse
from src.sql2db.services import SqlGenerationService

class SqlGenerator(Resource):
    def post(self):
        data = request.get_json(force=True)
        try:
            result = SqlGenerationService.generate(data.get('query'))
        except ValueError as e:
            return ApiResponse.error(message=str(e), status_code=400)
        except Exception as e:
            return ApiResponse.error(message=f"An error occurred {e}", status_code=400)

        return ApiResponse.success(
            message="SQL generated successfully",
            data=result,
            status_code=200
        )
