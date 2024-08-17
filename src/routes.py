from flask_restful import Api
from flask import Flask, Blueprint
from src.MP.routes import MP_routes
from src.UOT.routes import UOT_routes
from src.OPM.routes import OPM_routes
from src.Dashboard.routes import routes
from src.Contact.routes import Contact_routes
from src.Training.routes import training_routes
from src.Feedbacks.routes import feedback_routes
from src.ClickManagement.routes import click_routes
from src.Submission.routes import submission_routes
from src.User.routes import create_authentication_routes

class RouteInitialization:
    def __init__(self):
        self.blueprints = [
            {
                "name": "auth", 
                "blueprint": Blueprint('auth', __name__, static_url_path="assets"), 
                "register_callback": create_authentication_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "bp", 
                "blueprint": Blueprint('api', __name__, static_url_path="assets"), 
                "register_callback": routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "UOT", 
                "blueprint":  Blueprint('UOT', __name__, static_url_path="assets"), 
                "register_callback": UOT_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "training", 
                "blueprint":  Blueprint('Training', __name__, static_url_path="assets"), 
                "register_callback": training_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "click_management", 
                "blueprint":  Blueprint('Click', __name__, static_url_path="assets"), 
                "register_callback": click_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "feedbacks", 
                "blueprint":  Blueprint('Feedback', __name__, static_url_path="assets"), 
                "register_callback": feedback_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "MP_routes", 
                "blueprint":  Blueprint('MP_routes', __name__, static_url_path="assets"), 
                "register_callback": MP_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "OPM_routes", 
                "blueprint":  Blueprint('OPM_routes', __name__, static_url_path="assets"), 
                "register_callback": OPM_routes, 
                "url_prefix": "/api/v1"
            },
            
            {
                "name": "Contact_routes", 
                "blueprint":  Blueprint('Contact_routes', __name__, static_url_path="assets"), 
                "register_callback": Contact_routes, 
                "url_prefix": "/api/v1"
            },
            {
                "name": "Submission_routes", 
                "blueprint":  Blueprint('Submission_routes', __name__, static_url_path="assets"), 
                "register_callback": submission_routes, 
                "url_prefix": "/api/v1"
            },
        ]


    def init_app(self, flask_app: Flask):
        for blueprint in self.blueprints:
            init_route = Api(blueprint.get("blueprint"))
            blueprint.get("register_callback")(init_route)
            flask_app.register_blueprint(blueprint.get("blueprint"), url_prefix=blueprint.get("url_prefix"))

