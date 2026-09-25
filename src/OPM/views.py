from flask import jsonify
from flask_restful import Resource
from src.OPM.services import OPMDatasetService, OPMStatusService


class OPMMergeStatusResource(Resource):
    def get(self):
        return jsonify(OPMStatusService.get_merge_status())


class OPMDatasetResource(Resource):
    def get(self):
        return jsonify(OPMDatasetService.get_records())


DataMerge = OPMMergeStatusResource
newImplementation = OPMDatasetResource
