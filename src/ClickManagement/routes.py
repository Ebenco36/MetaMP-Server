from src.ClickManagement.views import ClickResource

def click_routes(api):
    api.add_resource(ClickResource, '/record-click')