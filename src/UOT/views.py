from flask import request
from flask_restful import Resource
from flask import jsonify
from src.utils.response import ApiResponse


class MachineLearningView(Resource):
    def __init__(self):
        from src.UOT.services import UOTWorkflowService

        self.workflow = UOTWorkflowService()

    def get(self, dataset, action):
        return jsonify('Welcome')
    
    def post(self, dataset, action):
        try:
            response = self.workflow.run_action(
                dataset,
                action,
                request.get_json() or {},
            )
        except ValueError as exc:
            return ApiResponse.error(str(exc), 400)
        return ApiResponse.success(response, "Fetch successfully", 200)
    

class UseCases(Resource):
    def __init__(self):
        pass
    
    def get(self):
        data = {
            "em_features": [
                'emt_molecular_weight', 
                'reconstruction_num_particles',
                'processed_resolution'
            ],
            "x_ray_features": [
                "cell_length_a", 
                "cell_length_b", 
                "cell_length_c", 
                "crystal_density_matthews",
                "molecular_weight", 
                "processed_resolution"
            ],
            "outlier_detection_algorithms": [
                "LocalOutlierFactor",
                "IsolationForest",
                "DBSCAN"
            ]
        }
        
        return ApiResponse.success(data, "Fetch successfully")
    
    def post(self):
        from src.UOT.services import UOTUseCaseService, get_use_cases

        data = request.get_json() or {}
        data["groupby"] = data.get("category", "group")
        payload = UOTUseCaseService.build_use_case_payload(data)
        response = get_use_cases(payload)
        
        return ApiResponse.success(response, "Fetch successfully")
