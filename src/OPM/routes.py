from src.OPM.views import OPMDatasetResource, OPMMergeStatusResource

def OPM_routes(api):
    api.add_resource(OPMMergeStatusResource, '/data-merge')
    api.add_resource(OPMDatasetResource, '/data-new-implementation')
