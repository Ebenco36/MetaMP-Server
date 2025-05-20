from src.MP.views import (
    DataResource, UsupervisedResource, 
    DataFilterResource, getFilterBasedOnMethodResource,
    GroupSubGroupResource, MLPrediction, MLPredictionPost,
    MLPredictionAccuracy, MLDimensionalityReductionCharts,
    GenerateRealSampleDataTest, SequenceTMResource
)

def MP_routes(api):
    api.add_resource(DataResource, '/data-view')
    api.add_resource(MLPrediction, '/ml-predictions')
    api.add_resource(SequenceTMResource, '/predict_tm')
    api.add_resource(UsupervisedResource, '/data-view-ML')
    api.add_resource(MLPredictionPost, '/ml-predictions-post')
    api.add_resource(DataFilterResource, '/data-view-filters')
    api.add_resource(MLPredictionAccuracy, '/ml-predictions-accuracy')
    api.add_resource(GroupSubGroupResource, '/group-sub-group-resource')
    api.add_resource(getFilterBasedOnMethodResource, '/method-based-filter')
    api.add_resource(MLDimensionalityReductionCharts, '/machine-learning-dm-charts')
    api.add_resource(GenerateRealSampleDataTest, '/ml-get-sample-real-data')