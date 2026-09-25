def UOT_routes(api):
    from src.UOT.views import MachineLearningView, UseCases

    api.add_resource(MachineLearningView, '/<string:dataset>/<string:action>')
    api.add_resource(UseCases, '/use-cases')
    
    
