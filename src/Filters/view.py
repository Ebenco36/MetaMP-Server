from flask import jsonify, request
from flask_restful import Resource

from src.Filters.services import FilterKitService, FilterQueryService


class BaseFilterResource(Resource):
    service = FilterKitService()

    def respond(self, payload):
        return jsonify(payload)

class Filters(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_filters_payload())


class MissingFilterKit(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_missing_filter_options())


class allowMissingPerc(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_missing_percentage_options())


class normalizationOptions(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_normalization_options())


class dimensionalityReductionOptions(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_dimensionality_reduction_options())


class dataSplitPercOptions(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_data_split_options())


class PCAComponentsOptions(BaseFilterResource):
    def get(self):
        features = FilterQueryService.parse_feature_count(
            request.args.get("n_features", 2)
        )
        return self.respond(self.service.get_pca_component_options(features))


class MachineLearningOptions(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_machine_learning_options())


class GraphOptions(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_graph_options())


"""
    We need the following function to test_or_train_kit
"""
class trainAndTestSplitOptions(BaseFilterResource):
    def get(self):
        return self.respond(self.service.get_train_and_test_split_options())
