from src.UOT.views import MachineLearningView, UseCases


def UOT_routes(api):
    api.add_resource(MachineLearningView, '/<string:dataset>/<string:action>')
    api.add_resource(UseCases, '/use-cases')
    
    