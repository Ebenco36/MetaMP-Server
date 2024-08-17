from flask import jsonify
from flask_restful import Resource
from src.MP.services import DataService

class DataMerge(Resource):
    def get(self):
        return 'Check your console for output'
    
    
class newImplementation(Resource):
    def get(self):
        result = DataService.get_data_by_column_search_download(column_name=None, value=None)
        #print(result)
        return jsonify(result.get("data", {}).to_dict(orient='records'))