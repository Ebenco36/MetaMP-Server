
from src.Contact.views import ContactResource

def Contact_routes(api):
    api.add_resource(ContactResource, '/contacts')