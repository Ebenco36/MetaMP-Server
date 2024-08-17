from src.OPM.views import DataMerge, newImplementation

def OPM_routes(api):
    api.add_resource(DataMerge, '/data-merge')
    api.add_resource(newImplementation, '/data-new-implementation')