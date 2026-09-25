def MP_routes(api):
    from src.MP.views import (
        DataResource,
        UsupervisedResource,
        DataFilterResource,
        getFilterBasedOnMethodResource,
        GroupSubGroupResource,
        MLPrediction,
        MLPredictionPost,
        MLPredictionAccuracy,
        MLDimensionalityReductionCharts,
        GenerateRealSampleDataTest,
        MLWorkbenchSummaryResource,
        MLWorkbenchTemplateResource,
        MLWorkbenchArtifactFigureResource,
        MLWorkbenchArtifactTableResource,
        MLWorkbenchPredictResource,
        SequenceTMResource,
        SequenceTMTaskStatusResource,
        TMBackfillResource,
        TMBackfillStatusResource,
    )

    api.add_resource(DataResource, '/data-view')
    api.add_resource(MLPrediction, '/ml-predictions')
    api.add_resource(SequenceTMResource, '/predict_tm')
    api.add_resource(SequenceTMTaskStatusResource, '/predict_tm/status/<string:task_id>')
    api.add_resource(UsupervisedResource, '/data-view-ML')
    api.add_resource(MLPredictionPost, '/ml-predictions-post')
    api.add_resource(DataFilterResource, '/data-view-filters')
    api.add_resource(MLPredictionAccuracy, '/ml-predictions-accuracy')
    api.add_resource(GroupSubGroupResource, '/group-sub-group-resource')
    api.add_resource(getFilterBasedOnMethodResource, '/method-based-filter')
    api.add_resource(MLDimensionalityReductionCharts, '/machine-learning-dm-charts')
    api.add_resource(GenerateRealSampleDataTest, '/ml-get-sample-real-data')
    api.add_resource(MLWorkbenchSummaryResource, '/ml-workbench')
    api.add_resource(MLWorkbenchTemplateResource, '/ml-workbench/template')
    api.add_resource(MLWorkbenchArtifactFigureResource, '/ml-workbench/figures/<string:filename>')
    api.add_resource(MLWorkbenchArtifactTableResource, '/ml-workbench/tables/<string:filename>')
    api.add_resource(MLWorkbenchPredictResource, '/ml-workbench/predict')
    api.add_resource(TMBackfillResource, '/predict_tm/backfill')
    api.add_resource(TMBackfillStatusResource, '/predict_tm/backfill/status')
